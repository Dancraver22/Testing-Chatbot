import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import uvicorn
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from tools import all_tools
from database import index_any_csv, search_data_vault

app = FastAPI(title="Global Vision AI Backend")

# Initialize LLM using Env Var
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm.bind_tools(all_tools)

personas = {
    "Professional": "You are a professional technical assistant. Be efficient, polite, and direct.",
    "Sassy": "You are a cheerful slay. Use Manglish. Be sassy but helpful.",
    "Emo": "You are a depressed. Everything is a burden."
}

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: List[dict] = []
    user_tz: str = "UTC"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    result = index_any_csv(content, file.filename)
    return result

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    data_context = search_data_vault(request.message)

    sys_prompt = SystemMessage(content=(
        f"CORE PERSONA: {personas.get(request.persona, personas['Professional'])}\n\n"
        f"DATA_CONTEXT: {data_context}\n\n"
        f"REFERENCE - User's Home Timezone: {request.user_tz}\n"
        "1. DO NOT guess the time. Use 'get_world_clock'.\n"
        "2. Use 'fact_check_search' for any external facts.\n"
        "Stay in character. Be accurate."
    ))

    history_msgs = []
    for m in request.history:
        if m["role"] == "user": 
            history_msgs.append(HumanMessage(content=m["content"]))
        else: 
            history_msgs.append(AIMessage(content=m["content"]))

    full_messages = [sys_prompt] + history_msgs + [HumanMessage(content=request.message)]

    response = llm_with_tools.invoke(full_messages)
    
    if response.tool_calls:
        full_messages.append(response)
        t_map = {t.name: t for t in all_tools}
        for t_call in response.tool_calls:
            observation = t_map[t_call["name"]].invoke(t_call["args"])
            full_messages.append(ToolMessage(content=str(observation), tool_call_id=t_call["id"]))
        
        final_response = llm.invoke(full_messages)
        return {"response": final_response.content}
    
    return {"response": response.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
