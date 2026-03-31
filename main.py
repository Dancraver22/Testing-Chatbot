from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from tools import all_tools
from database import index_any_csv, search_data_vault

app = FastAPI(title="Global Vision AI: Multi-tool Backend")
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
llm_with_tools = llm.bind_tools(all_tools)

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: List[dict] = []

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Endpoint to ingest any CSV/Excel into the Vector DB."""
    content = await file.read()
    result = index_any_csv(content, file.filename)
    return result

@app.post("/chat")
async def chat_logic(request: ChatRequest):
    # 1. RETRIEVE: Check the Vector DB for relevant info related to the message
    data_context = search_data_vault(request.message)

    # 2. CONSTRUCT: Build the professional system prompt with your persona rules
    sys_prompt = f"PERSONA: {request.persona}\nCONTEXT FROM DATA: {data_context}\n"
    sys_prompt += "INSTRUCTIONS: Use the provided DATA_CONTEXT to answer accurately. If the info isn't there, use your tools."

    # 3. EXECUTE: Standard tool-calling loop (Simplified for brevity)
    # In production, you would handle the tool loop here and return the final text
    return {"response": "This is where the processed AI text returns to Streamlit", "context_used": data_context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
