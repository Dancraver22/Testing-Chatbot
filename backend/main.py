import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# Import your custom logic
from tools import all_tools
from database import index_any_csv, search_data_vault

app = FastAPI(title="Global Vision AI Backend")

# Initialize LLM
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
llm_with_tools = llm.bind_tools(all_tools)

# --- YOUR PERSONAS (Preserved exactly) ---
personas = {
    "Professional": "You are a professional technical assistant. Be efficient, polite, and direct.",
    "Sassy": "You are a cheerful slay. Use Manglish. Be sassy but helpful.",
    "Emo": "You are a depressed. Everything is a burden. Low energy, no hope, but you'll answer if you must."
}

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: List[dict] = []
    user_tz: str = "UTC"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload any CSV to the Vector DB for analysis."""
    content = await file.read()
    result = index_any_csv(content, file.filename)
    return result

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Retrieve Data Context from ChromaDB
    data_context = search_data_vault(request.message)

    # 2. YOUR EXACT SYSTEM PROMPT (Preserved from your original script)
    sys_prompt = SystemMessage(content=(
        f"CORE PERSONA: {personas.get(request.persona, personas['Professional'])}\n\n"
        f"DATA_CONTEXT: {data_context}\n\n"
        "INSTRUCTIONS:\n"
        "1. You DO NOT know the current time or date. Do not guess.\n"
        "2. If the user asks for the time, you MUST use the 'get_world_clock' tool.\n"
        "3. If the user asks for facts, use 'fact_check_search'.\n"
        f"REFERENCE - User's Home Timezone: {request.user_tz}\n"
        "Dont assume the time without checking the tool first. "
        "Stay in character. Be grounded. "
        "FACT-CHECKING RULES:\n"
        "1. Response MUST be based ONLY on search results.\n"
        "2. DO NOT use internal knowledge to 'correct' results.\n"
        "3. Use specific names or dates EXACTLY as written.\n\n"
        "Always check fact on wiki or Google before telling."
    ))

    # 3. CONSTRUCT CONVERSATION
    # Convert history dicts back to LangChain objects
    history_msgs = []
    for m in request.history:
        if m["role"] == "user": history_msgs.append(HumanMessage(content=m["content"]))
        else: history_msgs.append(AIMessage(content=m["content"]))

    full_messages = [sys_prompt] + history_msgs + [HumanMessage(content=request.message)]

    # 4. TOOL EXECUTION LOOP (The 'Tavily' and 'Clock' logic)
    response = llm_with_tools.invoke(full_messages)
    
    if response.tool_calls:
        full_messages.append(response)
        for t_call in response.tool_calls:
            t_map = {t.name: t for t in all_tools}
            # Execute the tool (Tavily/Clock/etc)
            observation = t_map[t_call["name"]].invoke(t_call["args"])
            full_messages.append(ToolMessage(content=str(observation), tool_call_id=t_call["id"]))
        
        # Final call after tool results
        final_response = llm.invoke(full_messages)
        return {"response": final_response.content}
    
    return {"response": response.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
