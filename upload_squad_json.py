import json
import weaviate
from langchain.embeddings import HuggingFaceEmbeddings

client = weaviate.Client("http://localhost:8880")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with open("data/dev-v1.1.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

for article in squad_data["data"]:
    for para in article["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            question = qa["question"]
            answers = qa["answers"]
            answer_text = answers[0]["text"] if answers else ""
            
            full_text = f"Q: {question}\nA: {answer_text}\nContext: {context}"

            embedding = embedding_model.embed_query(full_text)
            client.data_object.create(
                data_object={"text": full_text},
                class_name="ResearchDoc",
                vector=embedding
            )

print("✅ SQuAD JSON data uploaded to Weaviate!")
