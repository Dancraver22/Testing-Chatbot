import streamlit as st
import os, requests, pytz
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Hybrid AI Agent", page_icon="🧠", layout="wide")

# 1. TIME ENGINE
def get_verified_time():
    myt = pytz.timezone("Asia/Kuala_Lumpur")
    return datetime.now(myt).strftime('%I:%M %p, %A, %B %d, %Y')

# 2. CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, streaming=True)

# 3. SIDEBAR
with st.sidebar:
    st.title("⚙️ AI Logic")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a factual, elite assistant. Be polite.",
        "Sassy": "You are witty and sarcastic. Don't be a boring robot.",
        "Emo": "You are moody and deep. Everything is gray."
    }

st.title(f"🤖 {persona} Assistant")

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 4. HYBRID LOGIC: Casual vs Fact-Check
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        curr_time = get_verified_time()
        
        # GATEKEEPER: Detect if search is actually needed
        # We don't search for "how are you", "hi", "yo", etc.
        needs_fact_check = any(k in user_input.lower() for k in [
            "who", "what", "where", "news", "price", "brand", "restaurant", "naknak", "weather", "is it"
        ])
        
        search_data = ""
        sources_text = ""
        
        if needs_fact_check and tavily_api_key:
            with st.status("🔍 Fact-checking your request...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                response = tavily.search(query=user_input, search_depth="advanced", max_results=3)
                search_data = "\n".join([res['content'] for res in response['results']])
                sources_text = "\n".join([f"- {res['title']}: {res['url']}" for res in response['results']])

        # THE DYNAMIC PROMPT
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"LOCAL_TIME: {curr_time}\n"
            f"SEARCH_DATA: {search_data}\n"
            f"SOURCES: {sources_text}\n\n"
            "RULES:\n"
            "1. If SEARCH_DATA is empty, be casual and don't cite anything.\n"
            "2. If SEARCH_DATA has info, use it and LIST the sources at the bottom.\n"
            "3. If the user argues about facts (like Naknak), rely ONLY on SEARCH_DATA.\n"
            "4. For casual 'hi' or 'how are you', do NOT talk about articles or databases."
        )

        full_response = st.write_stream(llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=full_response))
