import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import json
import pickle
import os

<<<<<<< HEAD
# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Faiss index
try:
    index = faiss.read_index("research_doc_index.faiss")
except Exception as e:
    print(f"Error loading Faiss index: {e}")
    exit(1)

# Load existing metadata or initialize
try:
    with open("research_doc_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
except FileNotFoundError:
    metadata = []

# Load SQuAD data
squad_file = "data/dev-v1.1.json"
if not os.path.exists(squad_file):
    print(f"Error: {squad_file} not found. Please ensure it exists in the data/ directory.")
    exit(1)

try:
    with open(squad_file, "r") as f:
        squad_data = json.load(f)
except Exception as e:
    print(f"Error loading {squad_file}: {e}")
    exit(1)

# Extract contexts
texts = []
new_metadata = []
try:
    for data in squad_data["data"]:
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            texts.append(context)
            new_metadata.append({"text": context, "source": "squad"})
except Exception as e:
    print(f"Error processing SQuAD data: {e}")
    exit(1)

# Generate embeddings
try:
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings, dtype=np.float32)
except Exception as e:
    print(f"Error generating embeddings: {e}")
    exit(1)

# Add to Faiss index
try:
    index.add(embeddings)
except Exception as e:
    print(f"Error adding to Faiss index: {e}")
    exit(1)

# Update metadata
metadata.extend(new_metadata)

# Save updated index and metadata
try:
    faiss.write_index(index, "research_doc_index.faiss")
    with open("research_doc_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("SQuAD data uploaded to Faiss index.")
except Exception as e:
    print(f"Error saving index or metadata: {e}")
    exit(1)
=======
client = weaviate.connect_to_local(port=8880, skip_init_checks=True)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    with open("data/dev-v1.1.json", "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    collection = client.collections.get("ResearchDoc")
    with collection.batch.dynamic() as batch:
        for article in squad_data["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    question = qa["question"]
                    answers = qa["answers"]
                    answer_text = answers[0]["text"] if answers else ""
                    full_text = f"Q: {question}\nA: {answer_text}\nContext: {context}"
                    embedding = embedding_model.embed_query(full_text)
                    batch.add_object(
                        properties={"text": full_text, "source": "SQuAD"},
                        vector=embedding
                    )
    print("✅ SQuAD JSON data uploaded to Weaviate!")
except Exception as e:
    print(f"Error uploading SQuAD data: {str(e)}")
finally:
    client.close()
>>>>>>> 000edc16f6ba5c64e265cfbaf3d526925b71196e
