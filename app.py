import streamlit as st
import base64
import pytz
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tavily import TavilyClient
from transformers import pipeline

# --- CONFIG & SECRETS ---
st.set_page_config(page_title="AI Prototype", layout="wide", page_icon="🌐")
groq_api_key = st.secrets["GROQ_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]

# --- 1. OPTIMIZED MODELS ---
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.6)

@st.cache_resource
def load_sentiment_model():
    # Local NLP model for speed and privacy
    return pipeline("sentiment-analysis", model="distilbert-base-uncased")

sentiment_pipe = load_sentiment_model()

# --- 2. SPECIALIST UTILITIES ---
def get_device_time():
    tz = pytz.timezone('Asia/Kuala_Lumpur') 
    now = datetime.now(tz)
    return now.strftime("%I:%M %p"), tz.zone

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

def process_data_file(file):
    """The Specialist Data Pipeline"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Clean data: drop empty rows and columns
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Create a snapshot for the AI context
        summary = {
            "columns": list(df.columns),
            "rows": len(df),
            "sample_data": df.head(3).to_dict(orient='records'),
            "stats": df.describe().to_dict() # Numerical insights for the AI
        }
        return df, summary
    except Exception as e:
        return None, f"Error processing file: {e}"

# --- 3. PERSONAS ---
persona_prompts = {
    "Professional": "Tech Consultant. Mirrored dialect. Polite but efficient. Focus on technical accuracy.",
    "Sassy": "Witty friend. High energy. Sassy, and use local slang",
    "Emo": "Burnt-out living guy. Low energy. Mixes English/Malay/Slang. Depressed"
}

# --- 4. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 5. UI SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    persona = st.selectbox("Choose Persona", list(persona_prompts.keys()))
    uploaded_image = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])
    uploaded_data = st.file_uploader("📊 Upload Data (CSV/Excel)", type=["csv", "xlsx"])
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- 6. CHAT INTERFACE ---
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        if isinstance(message.content, list):
            for item in message.content:
                if item["type"] == "text": st.markdown(item["text"])
        else:
            st.markdown(message.content)

if user_input := st.chat_input("Ask me about the data, the image, or anything..."):
    # A. Metadata Processing
    curr_time, tz_name = get_device_time()
    sentiment = sentiment_pipe(user_input)[0]
    user_mood = sentiment['label']
    
    # B. Data Analysis Trigger
    data_context = "No file uploaded."
    if uploaded_data:
        df, data_summary = process_data_file(uploaded_data)
        if df is not None:
            data_context = f"CURRENT_DATASET_SUMMARY: {data_summary}"
            st.info(f"📁 Data Detected: {len(df)} rows loaded.")
    
    # C. Smarter Search Trigger (RAG)
    search_keywords = ["who", "what", "where", "how", "price", "news", "specs", "vs", "weather"]
    needs_search = any(k in user_input.lower() for k in search_keywords)
    
    search_data = "No search performed."
    if needs_search and tavily_api_key:
        with st.spinner("🔍 Consulting the internet..."):
            tavily = TavilyClient(api_key=tavily_api_key)
            results = tavily.search(query=user_input, search_depth="advanced")
            search_data = "\n".join([f"- {r['content']}" for r in results['results'][:3]])

    # D. Content Construction
    content = [{"type": "text", "text": user_input}]
    if uploaded_image:
        base64_img = encode_image(uploaded_image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        st.chat_message("user").image(uploaded_image, width=300)

    # E. The "Specialist" System Message
    sys_msg = (
        f"SYSTEM ROLE: {persona_prompts[persona]}\n"
        "GLOBAL_HUMAN_PROTOCOL: Detect and MIRROR user dialect exactly (Manglish, AAV, etc). No formal translation.\n"
        f"<REFERENCE_DATA>\n"
        f"Time: {curr_time} | Mood: {user_mood}\n"
        f"Web Context: {search_data}\n"
        f"Data Context: {data_context}\n"
        f"</REFERENCE_DATA>\n"
        "INSTRUCTIONS:\n"
        "1. If Data Context is provided, answer questions based on the dataset summary.\n"
        "2. Do not mention system metadata (time/mood) unless asked.\n"
        "3. Match the user's slang ratio perfectly."
    )

    # F. AI Execution
    with st.chat_message("assistant"):
        msg = HumanMessage(content=content)
        # Combine System Message + History + Current Message
        full_response = st.write_stream(
            llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history + [msg])
        )
    
    # G. Save History
    st.session_state.chat_history.append(msg)
    st.session_state.chat_history.append(AIMessage(content=full_response))
