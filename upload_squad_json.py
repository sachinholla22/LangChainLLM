import json
import weaviate
from langchain.embeddings import HuggingFaceEmbeddings

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