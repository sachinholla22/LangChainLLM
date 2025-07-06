import weaviate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Step 1: Connect
client = weaviate.Client("http://localhost:8880")


#Step2 :Embedding Model
embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Step 3: Your research texts
texts = [
    "Spring Boot is used to build microservices and REST APIs in Java.",
    "LangChain enables chaining of LLMs with memory and tools.",
    "Vector databases like Weaviate store and search embeddings based on similarity.",
    "React is a frontend JavaScript library for building user interfaces.",
    "PDF documents can be used as sources for research QA systems."
]


for text in texts:
    embedding=embedding_model.embed_query(text)
    client.data_object.create(
        data_object={"text":text},
        class_name="ResearchDoc",
        vector=embedding
    )

print("✅ Uploaded research docs to Weaviate")    