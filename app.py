from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pickle
import os
import datetime
from dotenv import load_dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI

app = FastAPI()

# Load .env for Mistral API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Track session memory per user
user_last_queries = {}

# HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Mistral AI initialization
llm = ChatMistralAI(api_key=api_key)

# Load Faiss index and metadata
try:
    index = faiss.read_index("research_doc_index.faiss")
    with open("research_doc_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    raise Exception(f"Failed to load Faiss index or metadata: {e}")

# Zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define research-related categories
CATEGORIES = [
    "technical_explanation", "research", "general_greeting",
    "casual_talk", "joke", "non_technical"
]

def is_research_prompt(text: str) -> bool:
    result = classifier(text, CATEGORIES)
    top_labels = result["labels"][:3]
    return any(label in ["technical_explanation", "research"] for label in top_labels)

def is_generic_followup(text: str) -> bool:
    generic_phrases = [
        "more details", "elaborate", "continue", "go on", "further info", 
        "explain more", "more explanation", "brief more", "add more"
    ]
    return text.strip().lower() in generic_phrases

# Input format
class Query(BaseModel):
    id: int
    input: str

@app.post("/q")
async def query_endpoint(query: Query):
    try:
        input_text = query.input.strip()

        # Step 1: Handle generic follow-ups
        if is_generic_followup(input_text) and query.id in user_last_queries:
            input_text = user_last_queries[query.id] + "\n" + input_text

        # Step 2: Classify input
        is_research_related = is_research_prompt(input_text)

        if not is_research_related:
            return {
                "response": "👋 Hi! I'm your research assistant. Please ask queries related to research, technical topics, or learning content so I can assist you better.",
                "isResearchRelated": False
            }

        # Step 3: Embed query
        query_embedding = embedding_model.embed_query(input_text)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Step 4: Search vector DB
        k = 5
        distances, indices = index.search(query_embedding, k)

        if len(indices[0]) == 0 or all(i < 0 for i in indices[0]):
            return {
                "response": "Sorry, I couldn't find related research context. Try asking a more detailed question.",
                "isResearchRelated": True
            }

        # Step 5: Build context
        contexts = [metadata[i]["text"] for i in indices[0] if i >= 0 and i < len(metadata)]
        context_str = "\n".join(contexts)
        print("Matched Contexts:", contexts)
        # Step 6: Ask Mistral
        prompt = f"Context:\n{context_str}\n\nQuestion: {query.input}\nAnswer:"
        response = llm.invoke(prompt).content

        # Step 7: Save to Faiss + metadata
        index.add(query_embedding)
        metadata.append({
            "text": f"Q: {input_text}\nA: {response}",
            "source": "user_query",
            "user": str(query.id),
            "date": str(datetime.datetime.now())
        })
        faiss.write_index(index, "research_doc_index.faiss")
        with open("research_doc_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        user_last_queries[query.id] = input_text

        return {
            "response": response,
            "isResearchRelated": True
        }

    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}
