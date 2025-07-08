import faiss
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

def upload_static_data():
    try:
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Static data
        texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "AI CI/CD automates the development, testing, and deployment of AI models."
        ]
        metadata = [{"text": t, "source": "static"} for t in texts]  # Use 't' to avoid 'text' scope issues
        
        # Embed texts
        embeddings = embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Load Faiss index
        index = faiss.read_index("research_doc_index.faiss")
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Save updated index and metadata
        faiss.write_index(index, "research_doc_index.faiss")
        with open("research_doc_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        print("Static data uploaded to Faiss index.")
    except Exception as e:
        print(f"Error uploading static data: {e}")

if __name__ == "__main__":
    upload_static_data()