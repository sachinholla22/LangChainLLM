import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Faiss index
index = faiss.read_index("research_doc_index.faiss")

# Static texts
texts = [
    "LangChain is a framework for developing applications powered by language models.",
    "Faiss is an open-source library for efficient similarity search and clustering of dense vectors."
]
metadata = [
    {"text": text, "source": "static"} for text in texts
]

# Generate embeddings
embeddings = embedding_model.embed_documents(texts)
embeddings = np.array(embeddings, dtype=np.float32)

# Add to Faiss index
index.add(embeddings)

# Save updated index
faiss.write_index(index, "research_doc_index.faiss")

# Save metadata
with open("research_doc_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Static data uploaded to Faiss index.")