from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pickle
import os
import json
import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from transformers import pipeline

app = FastAPI()

# === Load environment ===
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# === Initialize models ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatMistralAI(api_key=api_key)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Categories for research classification ===
CATEGORIES = ["technical_explanation", "research", "non_technical", "casual_talk", "joke"]

# === Load or create FAISS index ===
dimension = 384  # for MiniLM
index_path = "research_doc_index.faiss"
meta_path = "research_doc_metadata.pkl"
debug_path = "debug_vectors.json"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatIP(dimension)
    metadata = []

# === Load or init debug vector log ===
if os.path.exists(debug_path):
    with open(debug_path, "r") as f:
        debug_vectors = json.load(f)
else:
    debug_vectors = []

# === Session memory ===
user_last_queries = {}

# === Input schema ===
class Query(BaseModel):
    id: int
    input: str

# === Helper: Research query check ===
def is_research_prompt(text: str) -> bool:
    result = classifier(text, CATEGORIES)
    top_labels = result["labels"][:3]
    return any(label in ["technical_explanation", "research"] for label in top_labels)

# === Helper: Generic follow-up checker ===
def is_generic_followup(text: str) -> bool:
    followups = ["more", "continue", "elaborate", "go on", "further"]
    return any(p in text.lower() for p in followups)

# === Main endpoint ===
@app.post("/q")
async def query_endpoint(query: Query):
    try:
        input_text = query.input.strip()

        # Handle vague input
        if is_generic_followup(input_text) and query.id in user_last_queries:
            input_text = user_last_queries[query.id] + "\n" + input_text

        # Validate research relevance
        if not is_research_prompt(input_text):
            return {
                "response": "👋 I'm here to help with research and technical queries. Please try asking something tech-related.",
                "isResearchRelated": False,
                "isInVector": False
            }

        # === Embed query ===
        query_embedding = embedding_model.embed_query(input_text)
        query_embedding_np = np.array([query_embedding], dtype=np.float32)

        # === Search in FAISS ===
        k = 3
        distances, indices = index.search(query_embedding_np, k)

        matched_contexts = []
        isInVector = False
        for i, dist in zip(indices[0], distances[0]):
            if i >= 0 and dist > 0.75 and i < len(metadata):
                matched_contexts.append(metadata[i]["text"])
                isInVector = True  # mark as found in vector

        # === If not found, ask directly ===
        if matched_contexts:
            context_str = "\n".join(matched_contexts)
            prompt = f"Context:\n{context_str}\n\nQuestion: {input_text}\nAnswer:"
        else:
            prompt = f"Answer the following technical question:\n\n{input_text}\nAnswer:"

        # === LLM Response ===
        response = llm.invoke(prompt).content

        # === Save vector, metadata, debug ===
        index.add(query_embedding_np)
        metadata.append({
            "text": f"Q: {input_text}\nA: {response}",
            "source": "user_query",
            "user": str(query.id),
            "date": str(datetime.datetime.now())
        })
        debug_vectors.append({
            "query": input_text,
            "vector": list(map(float, query_embedding))  # convert to plain float list
        })

        # === Save to disk ===
        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        with open(debug_path, "w") as f:
            json.dump(debug_vectors, f, indent=2)

        user_last_queries[query.id] = input_text

        return {
            "response": response,
            "isResearchRelated": True,
            "isInVector": isInVector
        }

    except Exception as e:
        return {"error": f"Query failed: {str(e)}"} 
