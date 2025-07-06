import weaviate
from langchain.embeddings import HuggingFaceEmbeddings

client=weaviate.Client("http://localhost:8880")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

query = "What is LangChain?"
query_embedding = embedding_model.embed_query(query)

response=client.query.get("ResearchDoc",["text"])\
    .with_near_vector({"vector":query_embedding})\
    .with_limit(2)\
    .do()

print("🔍 Search results:")
for item in response["data"]["Get"]["ResearchDoc"]:
    print("-", item["text"])