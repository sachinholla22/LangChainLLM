<<<<<<< HEAD
from fastapi import FastAPI
=======
from fastapi import FastAPI, Request
>>>>>>> 000edc16f6ba5c64e265cfbaf3d526925b71196e
from pydantic import BaseModel
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
<<<<<<< HEAD
import pickle
import os

app = FastAPI()

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatMistralAI(api_key="")
# Load Faiss index and metadata
try:
    index = faiss.read_index("research_doc_index.faiss")
    with open("research_doc_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    raise Exception(f"Failed to load Faiss index or metadata: {e}")

class Query(BaseModel):
    input: str

@app.post("/q")
async def query_endpoint(query: Query):
    try:
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

        # Determine if research-related
        is_research_related = any("squad" in meta["source"] for meta in [metadata[i] for i in indices[0]])

        return {
            "response": response,
            "isResearchRelated": is_research_related,
            "input": query.input
        }
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}
=======
from langchain_core.messages import HumanMessage
import weaviate

app = FastAPI()
client = weaviate.connect_to_local(port=8880, skip_init_checks=True)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatMistralAI(api_key="your-mistral-api-key", model="mistral-tiny")

class PromptRequest(BaseModel):
    input: str

class PromptResponse(BaseModel):
    response: str
    isResearchRelated: bool
    input: str

@app.post("/q", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    try:
        user_prompt = request.input
        query_vector = embedding_model.embed_query(user_prompt)
        collection = client.collections.get("ResearchDoc")
        results = collection.query.near_vector(near_vector=query_vector, limit=3, return_properties=["text"])
        docs = [obj.properties["text"] for obj in results.objects]
        if not docs:
            return PromptResponse(response="No research documents found", isResearchRelated=False, input=user_prompt)
        context = "\n".join(docs)
        full_prompt = f"Context:\n{context}\n\nUser Question: {user_prompt}"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        return PromptResponse(response=response.content, isResearchRelated=True, input=user_prompt)
    except Exception as e:
        return PromptResponse(response=f"Error: {str(e)}", isResearchRelated=False, input=user_prompt)
    finally:
        client.close()
>>>>>>> 000edc16f6ba5c64e265cfbaf3d526925b71196e
