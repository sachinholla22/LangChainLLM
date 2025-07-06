import weaviate

#connect to weavite
client=weavite.Client("http://localhost:8880")


#define schema doe the documents

schema={
    "class":"ResearchDoc",
    "description": "Stores research-related text",
    "vectorize":"none",
    "properties":[
        {
            "name":"text",
            "datatype":["text"],
        }
    ]
}


#create the schema if doesnt exist
if not client.schema.contains({"classes":[schema]}):
    client.schema.create_class(schema)

print('Schema  created successfully')