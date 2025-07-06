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