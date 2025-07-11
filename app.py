from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
import pickle
import os
import datetime
from transformers import pipeline
from dotenv import load_dotenv

# Track the last research-related prompt per user
user_last_queries = {}

# Initialize classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
app = FastAPI()

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=api_key)

# Load Faiss index and metadata
try:
    index = faiss.read_index("research_doc_index.faiss")
    with open("research_doc_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    raise Exception(f"Failed to load Faiss index or metadata: {e}")

CATEGORIES = [
    "technical_explanation", "research", "general_greeting",
    "casual_talk", "joke", "non_technical"
]
def is_research_prompt(text: str) -> bool:
    result = classifier(text, CATEGORIES)
    top_labels = result["labels"][:3]  # Consider top 3 labels
    return any(label in ["technical_explanation", "research"] for label in top_labels)


def is_generic_followup(text: str) -> bool:
    generic_phrases = [
        "more details", "elaborate", "continue", "go on", "further info", 
        "explain more", "more explanation", "brief more", "add more"
    ]
    return text.strip().lower() in generic_phrases

class Query(BaseModel):
    id: int
    input: str

@app.post("/q")
async def query_endpoint(query: Query):
    try:
        input_text = query.input.strip()

        # Replace vague input with last query if needed
        if is_generic_followup(input_text) and query.id in user_last_queries:
            input_text = user_last_queries[query.id] + "\n" + input_text

        # Detect research-type query
        is_research_related = is_research_prompt(input_text)
        
        # If not research-related, return static message immediately
        if not is_research_related:
        
            return {
                "response": "👋 Hi! I'm your research assistant. Please ask queries related to research, technical topics, or learning content so I can assist you better.",
                "isResearchRelated": False
            }
        # Embed and search
        query_embedding = embedding_model.embed_query(input_text)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = index.search(query_embedding, 5)
        contexts = [metadata[i]["text"] for i in indices[0]]
        context_str = "\n".join(contexts)

        # Ask Mistral
        prompt = f"Context:\n{context_str}\n\nQuestion: {query.input}\nAnswer:"
        response = llm.invoke(prompt).content

        # Save to vector DB + session
        if is_research_related:
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

            # 🔥 ADD THIS LINE for session memory:
            user_last_queries[query.id] = input_text

        return {
            "response": response,
            "isResearchRelated": is_research_related
        }
        
        # If not research-related, send static message
        if not is_research_related:
             return {
                 "response": "👋 Hi! I'm your research assistant. Please ask queries related to research, technical topics, or learning content so I can assist you better.",
                 "isResearchRelated": False
             }



    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}
