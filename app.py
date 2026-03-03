import streamlit as st
import os, pytz
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# API Keys setup
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Global Hybrid AI", page_icon="🌍", layout="wide")

# 1. NEW 2026 GLOBAL TIME ENGINE
def get_device_time():
    try:
        # Fetches actual timezone from the user's browser/phone
        user_tz_name = st.context.timezone or "UTC"
        user_tz = pytz.timezone(user_tz_name)
    except:
        user_tz = pytz.timezone("Asia/Kuala_Lumpur") # Fallback
        user_tz_name = "Asia/Kuala_Lumpur"
    
    now = datetime.now(user_tz)
    return now.strftime('%I:%M %p, %A, %B %d, %Y'), user_tz_name

# 2. CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, streaming=True)

# 3. SIDEBAR (Preserving your existing functions)
with st.sidebar:
    st.title("⚙️ AI Logic")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a factual, elite assistant. Be polite and precise.",
        "Sassy": "You are witty and sarcastic. Be funny but helpful.",
        "Emo": "You are moody and deep. Everything is gray and meaningless."
    }

st.title(f"🤖 {persona} Assistant")

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 4. HYBRID LOGIC: Global Time & Persona-Based Fact Check
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        curr_time, tz_name = get_device_time()
        
        # GATEKEEPER: Smarter detection for global facts and time
        needs_fact_check = any(k in user_input.lower() for k in [
            "who", "what", "where", "news", "price", "is it", "weather", "time in", "clock in"
        ])
        
        search_data = ""
        sources_text = ""
        
        if needs_fact_check and tavily_api_key:
            with st.status("🔍 Verifying global facts...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                # Force search for 2026 to ensure accuracy
                query = f"{user_input} current info March 2026"
                response = tavily.search(query=query, search_depth="advanced", max_results=3)
                search_data = "\n".join([res['content'] for res in response['results']])
                sources_text = "\n".join([f"- {res['title']}: {res['url']}" for res in response['results']])

        # THE DYNAMIC PROMPT (Preserves Persona + Global Info)
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"USER_TIMEZONE: {tz_name}\n"
            f"USER_LOCAL_TIME: {curr_time}\n"
            f"SEARCH_DATA: {search_data}\n"
            f"SOURCES: {sources_text}\n\n"
            "INSTRUCTIONS:\n"
            f"1. Stay in your {persona} character at all times, even when citing facts.\n"
            "2. If asked 'what time is it', use USER_LOCAL_TIME as the baseline.\n"
            "3. If asked for time in ANOTHER city, use SEARCH_DATA to find its current time.\n"
            "4. If you use SEARCH_DATA, mention your sources briefly at the end."
        )

        full_response = st.write_stream(llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=full_response))
