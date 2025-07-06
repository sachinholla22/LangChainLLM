from fastapi import FastApi,Request
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from  langchain_core.messages import HumanMessage
import weaviate

app=FastApi()

weaviate_client=weaviate.Client("http://localhost:8880")

#embedding Model

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM model (use your key and model name)
llm = ChatMistralAI(api_key="your-mistral-api-key", model="mistral-tiny")

# Request input model
class PromptRequest(BaseModel):
    input: str

# Response model
class PromptResponse(BaseModel):
    response: str
    isResearchRelated: bool

@app.post("/q",response_model=PromptResponse)
def generate_response(request:PromptResponse):
    user_prompt = request.input  
    query_vector=embedding_model.embed_query(user_prompt)

    #Search Weaviate
    results=weaviate_client.query.get("ResearchDoc",["text"])\
        .with_near_vector({"vector":query_vector})\
        .with_limit(3).do()
    

    docs = results.get("data", {}).get("Get", {}).get("ResearchDoc", [])

    if not docs:
        return PromptResponse(
            response="👋 Hello! I'm a research assistant. Please ask a research-related question or upload a research document.",
            isResearchRelated=False
        )
    
     # Join top matches as context
    context = "\n".join([doc["text"] for doc in docs])

    full_prompt = f"Context:\n{context}\n\nUser Question: {user_prompt}"

    # Ask the LLM
    response = llm.invoke([HumanMessage(content=full_prompt)])

    return PromptResponse(response=response.content, isResearchRelated=True)