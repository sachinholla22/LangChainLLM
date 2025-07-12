import faiss

def create_faiss_index():
    try:
        dimension = 768  # Dimension for all-MiniLM-L6-v2 embeddings
        index = faiss.IndexFlatL2(dimension)
        faiss.write_index(index, "research_doc_index.faiss")
        print("Faiss index 'ResearchDoc' created and saved to research_doc_index.faiss.  with 768 dimension")
    except Exception as e:
        print(f"Error creating Faiss index: {e}")

import pickle
with open("research_doc_metadata.pkl", "wb") as f:
    pickle.dump([], f)
print("Refreshed FAISS index and metadata.")

if __name__ == "__main__":
    create_faiss_index()