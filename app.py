from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
import pickle
import os
import re

app = FastAPI()

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatMistralAI(api_key="ve6zWkVaMmtmQF3hgW8OjP9cCz6m8TiG")  # Replace with your Mistral AI key

# Load Faiss index and metadata
try:
    index = faiss.read_index("research_doc_index.faiss")
    with open("research_doc_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    raise Exception(f"Failed to load Faiss index or metadata: {e}")

# Research-related keywords
research_keywords = r'\b(explain|why|what|how|demonstrate|is|will|when|brief)\b'

class Query(BaseModel):
    input: str

@app.post("/q")
async def query_endpoint(query: Query):
    try:
        # Check if query is research-related
        is_research_related = bool(re.search(research_keywords, query.input.lower()))

        # Embed query
        query_embedding = embedding_model.embed_query(query.input)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Search Faiss index
        k = 5  # Top 5 results
        distances, indices = index.search(query_embedding, k)
        contexts = [metadata[i]["text"] for i in indices[0]]

        # Prepare prompt for Mistral AI
        context_str = "\n".join(contexts)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query.input}\nAnswer:"
        
        # Query Mistral AI
        response = llm.invoke(prompt).content

        # Store research-related prompt + response
        if is_research_related:
            index.add(query_embedding)
            metadata.append({
                "text": f"Q: {query.input}\nA: {response}",
                "source": "user_query"
            })
            faiss.write_index(index, "research_doc_index.faiss")
            with open("research_doc_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

        return {
            "response": response,
            "isResearchRelated": is_research_related
        }
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}
