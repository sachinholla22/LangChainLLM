<<<<<<< HEAD
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
=======
import weaviate
from langchain.embeddings import HuggingFaceEmbeddings

client = weaviate.connect_to_local(port=8880, skip_init_checks=True)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    texts = [
        "Spring Boot is used to build microservices and REST APIs in Java.",
        "LangChain enables chaining of LLMs with memory and tools.",
        "Vector databases like Weaviate store and search embeddings based on similarity.",
        "React is a frontend JavaScript library for building user interfaces.",
        "PDF documents can be used as sources for research QA systems."
    ]
    collection = client.collections.get("ResearchDoc")
    with collection.batch.dynamic() as batch:
        for text in texts:
            embedding = embedding_model.embed_query(text)
            batch.add_object(
                properties={"text": text, "source": "Static"},
                vector=embedding
            )
    print("✅ Uploaded research docs to Weaviate")
finally:
    client.close()
>>>>>>> 000edc16f6ba5c64e265cfbaf3d526925b71196e
