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

# 1. SETUP
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Global AI Agent", page_icon="🌎", layout="wide")

# 2. THE ENGINE: ACCURATE CLOCK & WEATHER
@st.cache_data(ttl=300) # Fast cache for 5 minutes
def get_global_facts(city_name):
    try:
        # Geocode to find coordinates
        geolocator = Nominatim(user_agent="global_agent_" + str(uuid.uuid4())[:4])
        location = geolocator.geocode(city_name, timeout=10)
        
        if not location:
            return "Unknown Location", "N/A", "N/A"

        # Calculate exact local time
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)
        local_time = "Unknown"
        if tz_str:
            tz = pytz.timezone(tz_str)
            local_time = datetime.now(tz).strftime('%I:%M %p, %A')

        # Get live weather
        weather_res = requests.get(f"https://wttr.in/{city_name}?format=3", timeout=5)
        weather = weather_res.text if weather_res.status_code == 200 else "Weather unavailable."

        return location.address, local_time, weather
    except:
        return city_name, "N/A", "N/A"

# 3. UI SIDEBAR
with st.sidebar:
    st.title("Settings")
    persona_type = st.selectbox("Persona:", ["Professional", "Sassy", "Chill", "Emo"])
    user_city = st.text_input("Target City:", "Kuala Lumpur")
    
    if st.button("Clear Memory"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a precise, data-driven AI assistant.",
        "Sassy": "You are witty and think time is a suggestion. 🙄💅",
        "Chill": "You're a relaxed friend. ✨",
        "Emo": "You are moody. You think time is a meaningless cycle of darkness. 🖤"
    }

# 4. CHAT LOGIC
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Render Chat
st.title(f"🤖 {persona_type} AI ({user_city})")
for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 5. THE DYNAMIC RESPONSE
if user_input := st.chat_input("Ask about time, weather, or conversions..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        # We fetch the clock data EVERY time a message is sent
        full_addr, local_time, weather = get_global_facts(user_city)
        
        # Grounding the AI with facts it cannot ignore
        system_instructions = (
            f"{persona_prompts[persona_type]}\n"
            f"MANDATORY DATA: Location: {full_addr} | Local Time: {local_time} | Weather: {weather}\n"
            "INSTRUCTION: If the user asks for time or weather, you MUST provide the data above. "
            "Do not claim you don't know the time. Be accurate with unit conversions."
        )
        
        # Streaming response
        placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream([SystemMessage(content=system_instructions)] + st.session_state.chat_history):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
            
        placeholder.markdown(full_response)
        st.session_state.chat_history.append(AIMessage(content=full_response))
