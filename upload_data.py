import faiss
import pickle
import numpy as np
import os
from langchain_huggingface import HuggingFaceEmbeddings

# === New static texts to embed ===
texts = [
    "LangChain helps build LLM-based applications with context and memory.",
    "CI/CD stands for Continuous Integration and Continuous Deployment.",
    "Docker is used to containerize applications for portability.",
    "RCB won the IPL cup in 2025."
]

# === Embed the texts ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = embedding_model.embed_documents(texts)
vectors = np.array(vectors, dtype=np.float32)

# === File paths ===
index_path = "research_doc_index.faiss"
meta_path = "research_doc_metadata.pkl"
dimension = 384  # for all-MiniLM-L6-v2

# === Load existing index or create new ===
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatIP(dimension)
    metadata = []

# === Append vectors and metadata ===
index.add(vectors)
metadata += [{"text": t, "source": "static"} for t in texts]

# === Save updated index and metadata ===
faiss.write_index(index, index_path)
with open(meta_path, "wb") as f:
    pickle.dump(metadata, f)

print("✅ Static data appended to FAISS index and metadata")
