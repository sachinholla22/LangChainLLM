from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
import weaviate

app = FastAPI()
client = weaviate.connect_to_local(port=8880, skip_init_checks=True)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatMistralAI(api_key="your-mistral-api-key", model="mistral-tiny")

class PromptRequest(BaseModel):
    input: str

class PromptResponse(BaseModel):
    response: str
    isResearchRelated: bool
    input: str

@app.post("/q", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    try:
        user_prompt = request.input
        query_vector = embedding_model.embed_query(user_prompt)
        collection = client.collections.get("ResearchDoc")
        results = collection.query.near_vector(near_vector=query_vector, limit=3, return_properties=["text"])
        docs = [obj.properties["text"] for obj in results.objects]
        if not docs:
            return PromptResponse(response="No research documents found", isResearchRelated=False, input=user_prompt)
        context = "\n".join(docs)
        full_prompt = f"Context:\n{context}\n\nUser Question: {user_prompt}"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        return PromptResponse(response=response.content, isResearchRelated=True, input=user_prompt)
    except Exception as e:
        return PromptResponse(response=f"Error: {str(e)}", isResearchRelated=False, input=user_prompt)
    finally:
        client.close()