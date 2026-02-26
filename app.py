import streamlit as st
import os, requests, uuid, pytz
from datetime import datetime
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

# 1. SETUP & SECRETS (Safe for Cloud & Local)
load_dotenv()
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Global Mini-Gemini", page_icon="🌎", layout="wide")

# 2. THE FACT-CHECKER (No Hallucinations)
def get_realtime_data(city_name):
    try:
        # Get Location
        geolocator = Nominatim(user_agent=f"gemini_v4_{uuid.uuid4().hex[:4]}")
        loc = geolocator.geocode(city_name, timeout=10)
        if not loc: return city_name, "N/A", "N/A"
        
        # Get Exact Clock
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lng=loc.longitude, lat=loc.latitude)
        l_time = datetime.now(pytz.timezone(tz_str)).strftime('%I:%M %p, %A, %b %d')
        
        # Get Weather
        weather = requests.get(f"https://wttr.in/{city_name}?format=3", timeout=5).text
        return loc.address, l_time, weather
    except:
        return city_name, datetime.now().strftime('%I:%M %p'), "Weather unavailable"

# 3. SIDEBAR & PERSONA DEFINITIONS
with st.sidebar:
    st.title("🤖 Assistant Settings")
    persona = st.selectbox("Persona Style:", ["Professional", "Sassy", "Chill", "Emo"])
    user_city = st.text_input("Home Base (City):", "Kuala Lumpur")
    st.markdown("---")
    if st.button("🗑️ CLEAR CHAT"):
        st.session_state.chat_history = []
        st.rerun()

    # The AI's "Soul"
    persona_prompts = {
        "Professional": "You are a precise, data-grounded AI like Gemini.",
        "Sassy": "You are witty, sarcastic, and frankly smarter than everyone. 🙄💅",
        "Chill": "You're a relaxed friend. Everything is easy-going. ✨",
        "Emo": "You are moody, dark, and poetic. 🖤"
    }

# 4. CHAT HISTORY ENGINE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# 5. UI DISPLAY
st.title(f"🤖 {persona} Assistant")

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 6. INPUT & DYNAMIC GROUNDING
if user_input := st.chat_input("Ask about time, weather, or news..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        # A. Always Fetch Fresh Facts
        addr, l_time, l_weather = get_realtime_data(user_city)
        
        # B. Conditional Google Search (Only for News/Facts)
        search_info = ""
        news_keywords = ["news", "latest", "today", "happened", "who is", "price"]
        if any(k in user_input.lower() for k in news_keywords) and tavily_api_key:
            with st.status("Searching the web...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                search_info = str(tavily.search(query=user_input, search_depth="basic"))

        # C. Re-Inforce the Facts (The Accuracy Fix)
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"MANDATORY FACTS: Location: {addr} | Exact Time: {l_time} | Weather: {l_weather}\n"
            f"SEARCH RESULTS: {search_info}\n"
            "INSTRUCTION: Use MANDATORY FACTS for time/location. Use SEARCH for news. "
            "Never lie about the time. If you don't know, use your internal 2026 knowledge."
        )

        # D. Gemini-Style Streaming
        placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        st.session_state.chat_history.append(AIMessage(content=full_response))
