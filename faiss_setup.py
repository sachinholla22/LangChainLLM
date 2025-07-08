import faiss

def create_faiss_index():
    try:
        dimension = 384  # Dimension for all-MiniLM-L6-v2 embeddings
        index = faiss.IndexFlatL2(dimension)
        faiss.write_index(index, "research_doc_index.faiss")
        print("Faiss index 'ResearchDoc' created and saved to research_doc_index.faiss.")
    except Exception as e:
        print(f"Error creating Faiss index: {e}")

if __name__ == "__main__":
    create_faiss_index()