import streamlit as st
import os, requests, pytz
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

# 1. CLEAN STARTUP
load_dotenv()
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Pro AI Agent", page_icon="🌍")

# 2. FAST TOOLS (Using lightweight Requests instead of heavy libraries)
def get_weather(city):
    try:
        # wttr.in is much faster than geopy + weather APIs
        return requests.get(f"https://wttr.in/{city}?format=3", timeout=3).text
    except:
        return "Weather currently unavailable."

def get_time():
    # Instant local calculation
    return datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime('%I:%M %p, %A')

# 3. ENGINE SETUP
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, streaming=True)

# 4. UI
with st.sidebar:
    st.title("Agent Tools")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    user_city = st.text_input("Location:", "Kuala Lumpur")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.title(f"🤖 {persona} Assistant")

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 5. SMART RESPONSE LOOP
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        # INSTANT DATA
        curr_time = get_time()
        
        # CONDITIONAL DATA (Only runs if user asks for it)
        weather_info = ""
        search_info = ""
        
        if any(word in user_input.lower() for word in ["weather", "temperature", "hot", "rain"]):
            with st.status("Checking weather...", expanded=False):
                weather_info = get_weather(user_city)
        
        if any(word in user_input.lower() for word in ["news", "who", "latest", "update"]):
            if tavily_api_key:
                with st.status("Searching the web...", expanded=False):
                    tavily = TavilyClient(api_key=tavily_api_key)
                    search_info = str(tavily.search(query=user_input, search_depth="basic"))

        # GROUNDING PROMPT
        sys_msg = (
            f"Persona: {persona}. Location: {user_city}. Current Time: {curr_time}.\n"
            f"Weather Data: {weather_info}\n"
            f"Search Results: {search_info}\n"
            "INSTRUCTIONS: Only mention time/weather if the user asks. "
            "If the user says 'hey', just reply normally based on your persona."
        )

        # STREAMING
        full_res = st.write_stream(llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=full_res))
