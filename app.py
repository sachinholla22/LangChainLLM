from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pickle
import os
import json
import hashlib
import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from transformers import pipeline
import asyncio
import platform

app = FastAPI()

# === Load environment ===
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# === Initialize models ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatMistralAI(api_key=api_key)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

CATEGORIES = ["technical_explanation", "research", "non_technical", "casual_talk", "joke"]

# === Memory Stores ===
user_last_queries = {}
session_titles_file = "session_titles.json"
if os.path.exists(session_titles_file):
    with open(session_titles_file, "r") as f:
        session_titles = json.load(f)
else:
    session_titles = {}

recent_context_memory = {}      # stores up to 3 recent queries per session
input_hash_registry = {}        # stores hashes per session to skip duplicate embeddings

def save_session_titles():
    with open(session_titles_file, "w") as f:
        json.dump(session_titles, f, indent=2)

# === Request Schema ===
class Query(BaseModel):
    user_id: int
    id: int  # session id
    input: str

# === Helpers ===
def is_research_prompt(text: str) -> bool:
    result = classifier(text, CATEGORIES)
    top_labels = result["labels"][:3]
    return any(label in ["technical_explanation", "research"] for label in top_labels)

def is_generic_followup(text: str) -> bool:
    followups = ["more", "continue", "elaborate", "go on", "further"]
    return any(p in text.lower() for p in followups)

def extract_session_title(input_text: str) -> str:
    if isinstance(input_text, str):
        words = [w for w in input_text.lower().split() if not any(c in w for c in "@._") and w not in ["in", "with", "by"]]
        if not words:
            return "Untitled_Session"
        if len(words) > 1 and words[0] in ["explain", "describe", "tell"]:
            return f"{words[1]} {words[0]}tion"
        return " ".join(words[:2]) + " session"
    return "Invalid_Input_Session"

def get_session_paths(user_id, session_id):
    base_dir = "sessiondir"
    user_dir = os.path.join(base_dir, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    faiss_path = os.path.join(user_dir, f"session_{session_id}.faiss")
    meta_path = os.path.join(user_dir, f"session_{session_id}_meta.pkl")

    return faiss_path, meta_path

def load_or_create_session_index(user_id, session_id):
    faiss_path, meta_path = get_session_paths(user_id, session_id)
    if os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(384)
        metadata = []
    return index, metadata

def save_session_data(user_id, session_id, index, metadata):
    faiss_path, meta_path = get_session_paths(user_id, session_id)
    faiss.write_index(index, faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

# === Endpoint ===
@app.post("/q")
async def query_endpoint(query: Query):
    try:
        input_text = query.input.strip()
        session_id = query.id
        user_id = query.user_id

        index, metadata = load_or_create_session_index(user_id, session_id)

        # Setup memory for session
        session_key = f"{user_id}_{session_id}"
        if session_key not in recent_context_memory:
            recent_context_memory[session_key] = []
        if session_key not in input_hash_registry:
            input_hash_registry[session_key] = set()

        # Deduplication
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()
        if input_hash in input_hash_registry[session_key]:
            return {
                "response": "⚠️ You already asked this question in this session.",
                "isResearchRelated": True,
                "isInVector": True,
                "sessionTitle": session_titles.get(session_key, extract_session_title(input_text))
            }

        if is_generic_followup(input_text) and session_key in user_last_queries:
            input_text = user_last_queries[session_key] + "\n" + input_text

        if not is_research_prompt(input_text):
            return {
                "response": "👋 I'm here to help with research and technical queries. Please try asking something tech-related.",
                "isResearchRelated": False,
                "isInVector": False,
                "sessionTitle": session_titles.get(session_key, extract_session_title(input_text))
            }

        # === Embed input with context chain ===
        context_chain = "\n".join(recent_context_memory[session_key][-2:])
        combined_input = context_chain + "\n" + input_text if context_chain else input_text

        query_embedding = embedding_model.embed_query(combined_input)
        query_embedding_np = np.array([query_embedding], dtype=np.float32)

        # === Semantic Search ===
        k = 3
        distances, indices = index.search(query_embedding_np, k)

        matched_contexts = []
        isInVector = False
        for i, dist in zip(indices[0], distances[0]):
            if i >= 0 and dist > 0.75 and i < len(metadata):
                matched_contexts.append(metadata[i]["text"])
                isInVector = True

        # === Prompt Construction ===
        context_str = "\n".join(matched_contexts)
        prompt = f"Context:\n{context_str}\n\nQuestion: {input_text}\nAnswer:" if matched_contexts else f"Question: {input_text}\nAnswer:"
        response = llm.invoke(prompt).content

        # === Save title for first time
        if session_key not in session_titles:
            session_titles[session_key] = extract_session_title(input_text)
            save_session_titles()
        current_session_title = session_titles[session_key]

        # Save vector + metadata
        index.add(query_embedding_np)
        metadata.append({
            "text": f"Q: {input_text}\nA: {response}",
            "source": "user_query",
            "user": str(user_id),
            "date": str(datetime.datetime.now())
        })
        save_session_data(user_id, session_id, index, metadata)

        # Memory + dedup + context
        recent_context_memory[session_key].append(input_text)
        if len(recent_context_memory[session_key]) > 5:
            recent_context_memory[session_key] = recent_context_memory[session_key][-5:]
        input_hash_registry[session_key].add(input_hash)
        user_last_queries[session_key] = input_text

        return {
            "response": response,
            "isResearchRelated": True,
            "isInVector": isInVector,
            "sessionTitle": current_session_title
        }

    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}

# === Startup (optional) ===
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
