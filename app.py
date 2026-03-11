import streamlit as st
import os, pytz
import torch
from transformers import pipeline
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

# Load local environment variables if they exist
load_dotenv()

# --- 1. CONFIG & API SETUP ---
st.set_page_config(page_title="Global Hybrid AI", page_icon="🌍", layout="wide")

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

# --- 2. LOCAL PYTORCH OPTIMIZATION ---
@st.cache_resource
def load_sentiment_model():
    """
    Loads a lightweight DistilBERT model for sentiment analysis.
    Forcing device=-1 (CPU) ensures it stays stable on Streamlit Cloud.
    """
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

analyzer = load_sentiment_model()

# --- 3. UTILITIES: TIME & HISTORY ---
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

# Initialize LLM (Temperature 0 is crucial for factual RAG)
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, streaming=True, temperature=0)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ AI Logic Center")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    
    st.divider()
    st.subheader("🧠 Local PyTorch Engine")
    device_info = "GPU (Accelerated)" if torch.cuda.is_available() else "CPU (Standard)"
    st.info(f"Inference Mode: **{device_info}**")
    st.caption("Using DistilBERT for local intent analysis.")

    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a factual, elite assistant. Be polite and precise.",
        "Sassy": "You are witty and sarcastic. Be funny but helpful.",
        "Emo": "You are moody and deep. Everything is gray and meaningless."
    }

# --- 5. CHAT INTERFACE ---
st.title(f"🤖 {persona} Grounded Assistant")

# Display History
for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# Input Logic
if user_input := st.chat_input("Ask me anything..."):
    # Step A: Pre-processing with Local PyTorch
    with st.spinner("Analyzing sentiment..."):
        analysis = analyzer(user_input)[0]
        user_mood = analysis['label']
        mood_score = round(analysis['score'], 2)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Step B: Response Generation
    with st.chat_message("assistant"):
        curr_time, tz_name = get_device_time()
        
        # Determine if we need to RAG (Search)
        needs_fact_check = any(k in user_input.lower() for k in [
            "who", "what", "where", "news", "price", "is it", "weather", "time in", "studios", "policy", "location"
        ])
        
        search_data = "NO_EXTERNAL_SEARCH_RESULTS_FOUND"
        sources_text = ""
        
        if needs_fact_check and tavily_api_key:
            with st.status("🔍 Searching live database...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                query = f"{user_input} latest info 2026"
                response = tavily.search(query=query, search_depth="advanced", max_results=4)
                
                # Format context for the LLM
                search_data = "\n\n".join([
                    f"--- DOCUMENT {i+1} ---\nSource: {res['title']}\nSnippet: {res['content']}" 
                    for i, res in enumerate(response['results'])
                ])
                sources_text = "\n".join([f"- [{i+1}] {res['title']}: {res['url']}" for i, res in enumerate(response['results'])])

        # --- THE GROUNDING PROMPT ---
        sys_msg = (
            f"SYSTEM ROLE: {persona_prompts[persona]}\n"
            f"USER_METADATA: Sentiment={user_mood}, Timezone={tz_name}, LocalTime={curr_time}\n\n"
            f"PROVIDED_CONTEXT:\n{search_data}\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer ONLY using the PROVIDED_CONTEXT. Do not use outside knowledge.\n"
            "2. If the context does not contain the answer, say: 'I don't have enough specific data to answer that accurately.'\n"
            "3. Use inline citations like [1] or [2] next to facts extracted from the context.\n"
            "4. Maintain your persona but prioritize FACTUAL ACCURACY over creativity.\n"
            "5. If asked about time, refer to USER_METADATA LocalTime."
        )

        # Generate and Stream
        full_response = st.write_stream(
            llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history)
        )
        
        # Display Sources UI
        if sources_text:
            st.markdown(f"**Sources Found:**\n{sources_text}")
            full_response += f"\n\nSources Found:\n{sources_text}"

        st.session_state.chat_history.append(AIMessage(content=full_response))
