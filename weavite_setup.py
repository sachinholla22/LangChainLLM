import weaviate
from weaviate.classes.config import Property, DataType

client = weaviate.connect_to_local(port=8880, skip_init_checks=True)

try:
    schema = {
        "class": "ResearchDoc",
        "description": "Stores research-related text",
        "vectorizer": "none",
        "properties": [
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT)
        ]
    }
    if not client.collections.exists("ResearchDoc"):
        client.collections.create(
            name="ResearchDoc",
            description="Stores research-related text",
            vectorizer_config=None,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT)
            ]
        )
    print("Schema created successfully")
finally:
    client.close()