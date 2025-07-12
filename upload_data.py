import faiss
import pickle
import numpy as np
import traceback
from langchain_huggingface import HuggingFaceEmbeddings

def upload_static_data():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "AI CI/CD automates the development, testing, and deployment of AI models.",
            "RCB won the ipl cup in the year 2025 under Rajath patidar's captaincy",
            "Current year running is 2025"
        ]
        new_metadata = [{"text": t, "source": "static"} for t in texts]

        embeddings = embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        index = faiss.read_index("research_doc_index.faiss")

        try:
            with open("research_doc_metadata.pkl", "rb") as f:
                existing_metadata = pickle.load(f)
        except:
            existing_metadata = []

        index.add(embeddings)
        existing_metadata.extend(new_metadata)
        faiss.write_index(index, "research_doc_index.faiss")
        with open("research_doc_metadata.pkl", "wb") as f:
            pickle.dump(existing_metadata, f)

        print("Static data uploaded to Faiss index.")
    except Exception as e:
        print("Error uploading static data:")
        traceback.print_exc()  # 🔥 This prints full error trace

if __name__ == "__main__":
    upload_static_data()
