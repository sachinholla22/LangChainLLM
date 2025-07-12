import faiss
import pickle
import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

def upload_squad_data():
    try:
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        
        # Load SQuAD data
        squad_file = "data/dev-v1.1.json"
        if not os.path.exists(squad_file):
            raise FileNotFoundError(f"{squad_file} not found")
        
        with open(squad_file, "r") as f:
            squad_data = json.load(f)
        
        texts = []
        metadata = []
        for article in squad_data["data"]:
            for paragraph in article["paragraphs"]:
                texts.append(paragraph["context"])
                metadata.append({"text": paragraph["context"], "source": "squad"})
                print(f"Processed {len(texts)} contexts")  # Progress tracking
        
        # Embed texts
        embeddings = embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Load Faiss index
        index = faiss.read_index("research_doc_index.faiss")
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Load existing metadata if exists
        try:
            with open("research_doc_metadata.pkl", "rb") as f:
                existing_metadata = pickle.load(f)
        except FileNotFoundError:
            existing_metadata = []
        
        # Append new metadata
        existing_metadata.extend(metadata)
        
        # Save updated index and metadata
        faiss.write_index(index, "research_doc_index.faiss")
        with open("research_doc_metadata.pkl", "wb") as f:
            pickle.dump(existing_metadata, f)
        
        print("SQuAD data uploaded to Faiss index.")
    except Exception as e:
        print(f"Error uploading SQuAD data: {e}")

if __name__ == "__main__":
    upload_squad_data()