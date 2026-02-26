import streamlit as st
import os, requests, uuid, pytz
from datetime import datetime
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

load_dotenv()

# --- SECRETS ---
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Testing AI", page_icon="🎯")

# --- ACCURATE DATA ENGINE ---
def get_verified_context(city_name):
    try:
        geolocator = Nominatim(user_agent=f"accuracy_check_{uuid.uuid4().hex[:4]}")
        loc = geolocator.geocode(city_name, timeout=10)
        
        # Fallback to Malaysia Time if geocode fails
        tz = pytz.timezone("Asia/Kuala_Lumpur")
        if loc:
            tf = TimezoneFinder()
            tz_str = tf.timezone_at(lng=loc.longitude, lat=loc.latitude)
            if tz_str: tz = pytz.timezone(tz_str)
        
        l_time = datetime.now(tz).strftime('%I:%M %p, %A')
        weather = requests.get(f"https://wttr.in/{city_name}?format=3").text
        return loc.address if loc else city_name, l_time, weather
    except:
        # Emergency fallback for KL
        kl_time = datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime('%I:%M %p')
        return city_name, kl_time, "Weather unavailable"

# --- SIDEBAR ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("Settings")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    user_city = st.text_input("Home City:", "Kuala Lumpur")
    if st.button("Clear Memory"):
        st.session_state.chat_history = []
        st.rerun()

# --- CHAT LOGIC ---
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
st.title(f"🤖 {persona} AI")

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

if user_input := st.chat_input("Ask me anything..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        addr, l_time, l_weather = get_verified_context(user_city)
        
        # Persona Logic
        persona_prompts = {
            "Professional": "You are a helpful assistant. Only provide time/weather if asked.",
            "Sassy": "You are witty and sarcastic. Don't volunteer info unless asked. 💅",
            "Emo": "You are moody. You think time is a meaningless cycle. 🖤"
        }

        # THE FIX: We tell the AI to be SILENT about the data unless it's relevant
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"DATA BANK: Time: {l_time} | Location: {addr} | Weather: {l_weather}\n"
            "RULE: Do NOT mention the time, location, or weather in your greeting. "
            "ONLY use the DATA BANK if the user specifically asks for it. "
            "If they ask 'what time is it', use the Exact Time above. Do not guess."
        )

        placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        st.session_state.chat_history.append(AIMessage(content=full_response))

