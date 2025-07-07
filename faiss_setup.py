import faiss
import numpy as np

# Initialize Faiss index (FlatL2 for simplicity)
dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
index = faiss.IndexFlatL2(dimension)

# Save the empty index to disk
faiss.write_index(index, "research_doc_index.faiss")
print("Faiss index 'ResearchDoc' created and saved to research_doc_index.faiss.")