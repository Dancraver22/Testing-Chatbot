import streamlit as st
import os
import requests
import uuid
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

# 1. SECRETS LOADING (Cloud & Local)
load_dotenv()
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Global AI Agent", page_icon="🌎", layout="wide")

# 2. THE HARD-CODED CLOCK (Python-powered)
@st.cache_data(ttl=300)
def get_verified_context(city_name):
    try:
        # Using a unique ID to avoid 403 Forbidden errors
        geolocator = Nominatim(user_agent=f"gemini_clone_{uuid.uuid4().hex[:8]}")
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc: return city_name, "N/A", "N/A"
        
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lng=loc.longitude, lat=loc.latitude)
        l_time = datetime.now(pytz.timezone(tz_str)).strftime('%I:%M %p, %A, %B %d, %Y')
        
        weather = requests.get(f"https://wttr.in/{city_name}?format=3").text
        return loc.address, l_time, weather
    except:
        return city_name, datetime.now().strftime('%I:%M %p'), "Weather unavailable"

# 3. SIDEBAR & PERSONAS
with st.sidebar:
    st.title("🤖 Bot Settings")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Chill", "Emo"])
    user_city = st.text_input("Home City:", "Kuala Lumpur")
    if st.button("🗑️ Reset All"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a precise, elite AI assistant.",
        "Sassy": "You are witty and think the user is lucky to talk to you. 💅",
        "Chill": "You're a relaxed friend. ✨",
        "Emo": "You are moody and deep. 🖤"
    }

# 4. CHAT STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# 5. UI DISPLAY
st.title(f"🤖 {persona} AI")

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 6. THE DYNAMIC BRAIN
if user_input := st.chat_input("Ask me about the time, weather, or news..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        # STEP A: Get the Python Clock (Cannot Hallucinate)
        addr, l_time, l_weather = get_verified_context(user_city)
        
        # STEP B: Real-Time Web Search
        search_results = ""
        keywords = ["news", "latest", "today", "who is", "what is", "price"]
        if any(k in user_input.lower() for k in keywords) and tavily_api_key:
            with st.status("Searching the web...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                search_results = tavily.search(query=user_input, search_depth="basic")

        # STEP C: THE FORCEFUL PROMPT
        # We tell the AI it HAS access so it stops saying it doesn't.
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            "CRITICAL: You HAVE real-time access to the world through the following data:\n"
            f"- CURRENT LOCATION: {addr}\n"
            f"- EXACT LOCAL TIME: {l_time}\n"
            f"- CURRENT WEATHER: {l_weather}\n"
            f"- LIVE SEARCH RESULTS: {search_results}\n\n"
            "INSTRUCTION: Use the EXACT LOCAL TIME provided above if the user asks for the time. "
            "Never say 'I don't have access to real-time data.' You are grounded in these facts."
        )
        
        # STEP D: Streaming Response
        placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        st.session_state.chat_history.append(AIMessage(content=full_response))
