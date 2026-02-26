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

# 1. SETUP
load_dotenv()

# Safe Secret Loading
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Global AI Agent", page_icon="🌍", layout="wide")

# 2. ACCURATE TOOLS (Cached for speed)
@st.cache_data(ttl=300)
def get_verified_context(city_name):
    try:
        geolocator = Nominatim(user_agent="mini_gemini_pro_" + str(uuid.uuid4())[:4])
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc: return "Unknown", "N/A", "N/A"
        
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lng=loc.longitude, lat=loc.latitude)
        l_time = datetime.now(pytz.timezone(tz_str)).strftime('%I:%M %p, %A') if tz_str else "N/A"
        
        weather = requests.get(f"https://wttr.in/{city_name}?format=3").text
        return loc.address, l_time, weather
    except:
        return city_name, "N/A", "N/A"

# 3. SIDEBAR
with st.sidebar:
    st.title("🤖 Agent Settings")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Chill", "Emo"])
    user_city = st.text_input("Home Location:", "Kuala Lumpur")
    st.markdown("---")
    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

# 4. ENGINE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not groq_api_key:
    st.error("Please add GROQ_API_KEY to your .env file!")
    st.stop()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# 5. CHAT DISPLAY
st.title(f"🤖 {persona} AI")

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# 6. INPUT & SEARCH LOGIC
if user_input := st.chat_input("Ask about time, weather, or news..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        # A. Always get the Local Clock/Weather
        addr, l_time, l_weather = get_verified_context(user_city)
        
        # B. Real Web Search (Tavily)
        search_results = ""
        keywords = ["news", "latest", "who is", "what is", "happened", "price of"]
        if any(k in user_input.lower() for k in keywords) and tavily_api_key:
            with st.status("Searching the web...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                search_results = tavily.search(query=user_input, search_depth="basic")
                st.write("Found latest info on the web!")

        # C. Grounding & Persona
        persona_prompts = {
            "Professional": "You are a precise, data-grounded AI.",
            "Sassy": "You are witty, but your facts are always 100% correct. 💅",
            "Chill": "You're a relaxed friend. ✨",
            "Emo": "You are moody, but you never lie about the data. 🖤"
        }
        
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"STRICT TRUTH: Location: {addr} | Time: {l_time} | Weather: {l_weather}\n"
            f"SEARCH DATA: {search_results}\n"
            "INSTRUCTION: Use STRICT TRUTH for time/weather. Use SEARCH DATA for news/facts. "
            "Never guess. If you search, summarize the findings."
        )
        
        # D. Streaming
        placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        st.session_state.chat_history.append(AIMessage(content=full_response))
