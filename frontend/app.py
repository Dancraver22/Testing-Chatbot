import streamlit as st
import base64
import pandas as pd
import requests  # Added to talk to the Backend
from datetime import datetime
from streamlit_javascript import st_javascript

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="AI Prototype", layout="wide", page_icon="🌐")

# --- 2. IP & TIMEZONE DETECTION ---
user_tz = st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")

# --- 3. DATA & IMAGE UTILS ---
def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')

def process_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        # Summary for the UI/Prompt
        return df, {"cols": list(df.columns), "rows": len(df), "sample": df.head(2).to_dict()}
    except Exception as e:
        return None, str(e)

# Personas (Kept exactly as yours)
personas = {
    "Professional": "You are a professional technical assistant. Be efficient, polite, and direct.",
    "Sassy": "You are a cheerful slay. Use Manglish. Be sassy but helpful.",
    "Emo": "You are a depressed. Everything is a burden. Low energy, no hope, but you'll answer if you must."
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    selected_persona = st.selectbox("Persona", list(personas.keys()))
    uploaded_img = st.file_uploader("📸 Image", type=["jpg", "png"])
    uploaded_csv = st.file_uploader("📊 Data", type=["csv", "xlsx"])
    st.info(f"📍 Detected Time Zone: {user_tz if user_tz else 'Locating...'}")
    
    # NEW: Index Data Button for ChromaDB
    if uploaded_csv:
        if st.button("Index to Long-Term Memory"):
            with st.spinner("Indexing..."):
                files = {"file": (uploaded_csv.name, uploaded_csv.getvalue())}
                res = requests.post("http://backend:8000/upload", files=files)
                if res.status_code == 200:
                    st.success("Data indexed successfully!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- 5. RENDER HISTORY ---
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 6. EXECUTION ---
if user_input := st.chat_input("Ask me anything..."):
    
    # Process local snippet for context (Your logic preserved)
    data_ctx = "No file uploaded."
    if uploaded_csv:
        df, summary = process_data(uploaded_csv)
        data_ctx = f"Summary: {summary}. Data Snippet: {df.head(5).to_dict()}"

    # Build Message with Image Support
    b64_img = None
    if uploaded_img:
        b64_img = encode_image(uploaded_img)

    # UI: Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
        if uploaded_img: st.image(uploaded_img, width=300)

    # --- 7. TALK TO BACKEND ---
    # Instead of running the LLM here, we send it to main.py
    payload = {
        "message": user_input,
        "persona": selected_persona,
        "history": st.session_state.chat_history[-5:], # Send last 5 for context
        "user_tz": str(user_tz) if user_tz else "UTC",
        "image_data": b64_img, # Optional: if your main.py supports it
        "local_data_context": data_ctx
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # This points to the Docker service name 'backend'
                response = requests.post("http://backend:8000/chat", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    ans = result["response"]
                    st.markdown(ans)
                    
                    # Save to History
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}. Is the backend running?")
