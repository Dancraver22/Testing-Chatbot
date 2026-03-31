from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tools import all_tools

app = FastAPI(title="Global Vision AI Backend")
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
llm_with_tools = llm.bind_tools(all_tools)

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: List[dict] = []
    data_context: str = "No file uploaded."

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # This replaces the logic currently in your app.py execution block
    # It allows any frontend (Web, Mobile, Bot) to use your AI brain
    pass 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
