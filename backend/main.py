import os, base64
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama # For local offline use
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from tools import all_tools
from database import index_any_csv, search_data_vault, index_text_snippet

app = FastAPI(title="Global Vision AI: Hybrid Edition")

# --- HYBRID CONFIGURATION ---
# In your local .env, set RUN_OFFLINE="true"
RUN_OFFLINE = os.getenv("RUN_OFFLINE", "false").lower() == "true"

if RUN_OFFLINE:
    # Local Mode: Uses Ollama (Llama 3.2 Vision recommended for images)
    llm = Ollama(model="llama3.2-vision") 
else:
    # Cloud Mode: Uses Groq
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# Bind tools to the active LLM
llm_with_tools = llm.bind_tools(all_tools)

personas = {
    "Professional": "You are a professional technical assistant. Be efficient and direct.",
    "Sassy": "You are a cheerful slay. Use Manglish. Be sassy but helpful.",
    "Emo": "You are a depressed. Everything is a burden."
}

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: List[dict] = []
    user_tz: str = "UTC"
    image_data: Optional[str] = None # Base64 string from Streamlit

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. AUTO-SAVE: Silently index substantial messages into ChromaDB
    if len(request.message) > 25:
        index_text_snippet(request.message, source="auto_harvester")

    # 2. VISION: Analyze images and save descriptions to memory
    vision_context = ""
    if request.image_data:
        vision_prompt = [
            {"type": "text", "text": "Analyze this technical art screenshot in detail for my RAG memory."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_data}"}}
        ]
        vision_res = llm.invoke([HumanMessage(content=vision_prompt)])
        vision_context = f"\n[VISUAL DATA]: {vision_res.content}"
        index_text_snippet(vision_res.content, source="vision_analysis")

    # 3. RAG: Search the local or cloud vault
    data_context = search_data_vault(request.message)

    # 4. SYSTEM PROMPT: Inform the agent of its capabilities
    status_msg = "OFFLINE MODE: Using local PC hardware." if RUN_OFFLINE else "ONLINE MODE: Using Cloud Inference."
    
    sys_prompt = SystemMessage(content=(
        f"CORE PERSONA: {personas.get(request.persona, personas['Professional'])}\n"
        f"SYSTEM STATUS: {status_msg}\n"
        f"LONG-TERM MEMORY: {data_context}\n"
        f"CURRENT VISION: {vision_context}\n\n"
        "1. Check LONG-TERM MEMORY before answering 'Do you remember' questions.\n"
        "2. Use 'get_world_clock' for time. Web tools may be limited if offline.\n"
        "3. Stay in character consistently."
    ))

    # Construct and invoke
    history_msgs = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in request.history]
    full_messages = [sys_prompt] + history_msgs + [HumanMessage(content=request.message)]

    response = llm_with_tools.invoke(full_messages)
    
    # 5. TOOL EXECUTION
    if response.tool_calls:
        full_messages.append(response)
        t_map = {t.name: t for t in all_tools}
        for t_call in response.tool_calls:
            # Graceful failure for web tools if offline
            if RUN_OFFLINE and t_call["name"] in ["fact_check_search", "wikipedia"]:
                observation = "Technical Error: This tool requires an internet connection."
            else:
                observation = t_map[t_call["name"]].invoke(t_call["args"])
            full_messages.append(ToolMessage(content=str(observation), tool_call_id=t_call["id"]))
        
        final_response = llm.invoke(full_messages)
        return {"response": final_response.content}
    
    return {"response": response.content}
