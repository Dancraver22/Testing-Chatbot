import streamlit as st
import time
import base64
import os
from datetime import datetime
import pytz
from PIL import Image
from io import BytesIO

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tavily import TavilyClient
from transformers import pipeline
import torch

# --- CONFIG & SECRETS ---
st.set_page_config(page_title="Global Vision AI 2026", layout="wide")
groq_api_key = st.secrets["GROQ_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]

# --- 1. OPTIMIZED MODELS (MARCH 2026) ---
# Using Llama 4 Scout for high-speed Vision + Multilingual support
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.6)

# Local Sentiment Analysis (Running on CPU for stability)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased")

sentiment_pipe = load_sentiment_model()

# --- 2. UTILITY FUNCTIONS ---
def get_device_time():
    tz = pytz.timezone('Asia/Kuala_Lumpur') # Matches your local context
    now = datetime.now(tz)
    return now.strftime("%I:%M %p"), tz.zone

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

# --- 3. DYNAMIC PERSONAS ---
persona_prompts = {
    "Professional": "Tech Consultant. Mirrored dialect. Polite but efficient. Focus on technical accuracy.",
    "Sassy": "Witty friend. High energy. Uses 'Abuden', 'Weh', and matches user's slang perfectly.",
    "Emo": "Burnt-out KL Dev. Low energy. Mixes English/Malay/Slang. Everything is 'sien' or 'koyak'."
}

# --- 4. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 5. UI SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    persona = st.selectbox("Choose Persona", list(persona_prompts.keys()))
    uploaded_image = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- 6. CHAT INTERFACE ---
for message in st.session_state.chat_history:
    # Handle rendering of both text and image history
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        if isinstance(message.content, list):
            for item in message.content:
                if item["type"] == "text": st.markdown(item["text"])
        else:
            st.markdown(message.content)

if user_input := st.chat_input("Ask me about anything or the image..."):
    # A. Initial Processing
    curr_time, tz_name = get_device_time()
    sentiment = sentiment_pipe(user_input)[0]
    user_mood = sentiment['label']
    
    # B. Smarter Search Trigger
    search_keywords = ["who", "what", "where", "how", "price", "news", "specs", "vs", "weather"]
    needs_search = any(k in user_input.lower() for k in search_keywords)
    
    search_data = "No search performed."
    if needs_search and tavily_api_key:
        with st.spinner("🔍 Checking the internet..."):
            tavily = TavilyClient(api_key=tavily_api_key)
            results = tavily.search(query=user_input, search_depth="advanced")
            search_data = "\n".join([f"- {r['content']}" for r in results['results'][:3]])

    # C. Build Multimodal Content
    content = [{"type": "text", "text": user_input}]
    if uploaded_image:
        base64_img = encode_image(uploaded_image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        st.chat_message("user").image(uploaded_image, width=300)

    # D. Final System Prompt (Anti-Time-Spam + Chameleon)
    sys_msg = (
        f"SYSTEM ROLE: {persona_prompts[persona]}\n"
        "GLOBAL_HUMAN_PROTOCOL: Detect and MIRROR user dialect exactly (Manglish, AAV, etc). No formal translation.\n"
        f"<REFERENCE_DATA_ONLY>\n"
        f"Time: {curr_time} | Mood: {user_mood} | Context: {search_data}\n"
        "</REFERENCE_DATA_ONLY>\n"
        "INSTRUCTIONS:\n"
        "1. DO NOT mention the time/sentiment unless the user asks.\n"
        "2. Use the provided Context to answer factual questions.\n"
        "3. Match the user's slang ratio (Rojak style)."
    )

    # E. Execution
    with st.chat_message("assistant"):
        msg = HumanMessage(content=content)
        full_response = st.write_stream(
            llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history + [msg])
        )
    
    # Save History
    st.session_state.chat_history.append(msg)
    st.session_state.chat_history.append(AIMessage(content=full_response))
