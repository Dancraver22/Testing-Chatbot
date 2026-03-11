import streamlit as st
import os, pytz
import torch  # <--- NEW: PyTorch Integration
from transformers import pipeline  # <--- NEW: HuggingFace (Built on PyTorch)
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# --- NEW: PYTORCH LOCAL MODEL LOADING ---
# This loads a local model to your CPU/GPU. Sunway loves this "Local AI" approach.
@st.cache_resource
def load_sentiment_model():
    # 'distilbert-base-uncased-finetuned-sst-2-english' is the gold standard for light apps
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

analyzer = pipeline("sentiment-analysis", model="...", device=-1) # -1 forces CPU

# API Keys setup
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Global Hybrid AI", page_icon="🌍", layout="wide")

# 1. TIME ENGINE
def get_device_time():
    try:
        user_tz_name = st.context.timezone or "UTC"
        user_tz = pytz.timezone(user_tz_name)
    except:
        user_tz = pytz.timezone("Asia/Kuala_Lumpur") 
        user_tz_name = "Asia/Kuala_Lumpur"
    
    now = datetime.now(user_tz)
    return now.strftime('%I:%M %p, %A, %B %d, %Y'), user_tz_name

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, streaming=True)

# 3. SIDEBAR
with st.sidebar:
    st.title("⚙️ AI Logic")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    
    # --- NEW: PYTORCH VISUALIZER ---
    st.divider()
    st.subheader("🧠 Local PyTorch Stats")
    # This shows the recruiter you understand hardware utilization
    device_label = "GPU (Accelerated)" if torch.cuda.is_available() else "CPU (Standard)"
    st.info(f"Running on: **{device_label}**")

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

# 4. HYBRID LOGIC
if user_input := st.chat_input("Ask me anything..."):
    # --- STEP A: LOCAL PYTORCH CALCULATION ---
    # We use PyTorch to detect if the user is angry/happy before responding
    with st.spinner("PyTorch analyzing intent..."):
        analysis = analyzer(user_input)[0]
        user_mood = analysis['label'] # 'POSITIVE' or 'NEGATIVE'
        mood_score = round(analysis['score'], 2)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        curr_time, tz_name = get_device_time()
        
        needs_fact_check = any(k in user_input.lower() for k in [
            "who", "what", "where", "news", "price", "is it", "weather", "time in", "clock in"
        ])
        
        search_data = ""
        sources_text = ""
        
        if needs_fact_check and tavily_api_key:
            with st.status("🔍 Verifying global facts...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                query = f"{user_input} current info March 2026"
                response = tavily.search(query=query, search_depth="advanced", max_results=3)
                search_data = "\n".join([res['content'] for res in response['results']])
                sources_text = "\n".join([f"- {res['title']}: {res['url']}" for res in response['results']])

        # --- STEP B: FEED PYTORCH DATA INTO SYSTEM PROMPT ---
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"USER_MOOD_DETECTED: {user_mood} (Confidence: {mood_score})\n" # Local AI insight
            f"USER_TIMEZONE: {tz_name}\n"
            f"USER_LOCAL_TIME: {curr_time}\n"
            f"SEARCH_DATA: {search_data}\n"
            f"SOURCES: {sources_text}\n\n"
            "INSTRUCTIONS:\n"
            f"1. Stay in character. If USER_MOOD_DETECTED is NEGATIVE, be more empathetic (or sassier if Sassy).\n"
            "2. Use USER_LOCAL_TIME for time questions."
        )

        full_response = st.write_stream(llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=full_response))

