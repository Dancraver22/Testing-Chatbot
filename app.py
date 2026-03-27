import streamlit as st
import base64
import pandas as pd
from PIL import Image
from io import BytesIO

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from transformers import pipeline

# --- IMPORT YOUR TOOLS ---
from tools import all_tools  # This pulls in your World Clock, Tavily Search, etc.

# --- CONFIG & SECRETS ---
st.set_page_config(page_title="AI Agent Prototype", layout="wide", page_icon="🤖")
groq_api_key = st.secrets["GROQ_API_KEY"]

# --- 1. MODELS & TOOLS BINDING ---
# We use Llama-4-Scout because it is specifically optimized for Tool Use
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.6)

# This is the "Magic" line: It connects the Brain (LLM) to the Hands (Tools)
llm_with_tools = llm.bind_tools(all_tools)

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased")

sentiment_pipe = load_sentiment_model()

# --- 2. SPECIALIST UTILITIES ---
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

def process_data_file(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        summary = {
            "columns": list(df.columns),
            "rows": len(df),
            "sample_data": df.head(3).to_dict(orient='records'),
            "stats": df.describe().to_dict()
        }
        return df, summary
    except Exception as e:
        return None, f"Error: {e}"

# --- 3. PERSONAS ---
persona_prompts = {
    "Professional": "Tech Consultant. Mirrored dialect. Polite but efficient. Focus on technical accuracy.",
    "Sassy": "Witty friend. High energy. Sassy, use local slang (Manglish).",
    "Emo": "Burnt-out KL Dev. Low energy. Mixes English/Malay/Slang. Tired of life."
}

# --- 4. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 5. UI SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Agent Settings")
    persona = st.selectbox("Choose Persona", list(persona_prompts.keys()))
    uploaded_image = st.file_uploader("📸 Image Input", type=["jpg", "png"])
    uploaded_data = st.file_uploader("📊 Data Input (CSV/Excel)", type=["csv", "xlsx"])
    
    if st.button("Clear Memory"):
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

if user_input := st.chat_input("Message the agent..."):
    # A. Sentiment Detection
    sentiment = sentiment_pipe(user_input)[0]
    user_mood = sentiment['label']
    
    # B. Data Context (If uploaded)
    data_context = "No file uploaded."
    if uploaded_data:
        df, data_summary = process_data_file(uploaded_data)
        if df is not None:
            data_context = f"CURRENT_DATASET_SUMMARY: {data_summary}"

    # C. Image Handling
    content = [{"type": "text", "text": user_input}]
    if uploaded_image:
        base64_img = encode_image(uploaded_image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        st.chat_message("user").image(uploaded_image, width=300)

    # D. The Unified System Prompt
    sys_msg = SystemMessage(content=(
        f"SYSTEM ROLE: {persona_prompts[persona]}\n"
        "GLOBAL_HUMAN_PROTOCOL: Mirror user dialect exactly (Manglish/Rojak). No formal translation.\n"
        f"USER_MOOD: {user_mood}\n"
        f"DATA_CONTEXT: {data_context}\n"
        "TOOL_USAGE: You have access to a World Clock, Tavily Search, and File Saving tools. "
        "Use them ONLY when necessary to provide accurate, grounded answers."
    ))

    # E. Execution (The Agent Loop)
    with st.chat_message("assistant"):
        user_msg = HumanMessage(content=content)
        
        # 1. First Pass: Let the AI decide if it needs tools
        response = llm_with_tools.invoke([sys_msg] + st.session_state.chat_history + [user_msg])
        
        # 2. Tool Execution Logic
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Find the tool in your tools.py list
                selected_tool = next(t for t in all_tools if t.name == tool_call["name"])
                with st.spinner(f"🛠️ Agent using {selected_tool.name}..."):
                    tool_output = selected_tool.func(**tool_call["args"])
                    # Provide the tool output back to the AI for a final answer
                    final_response = llm.invoke([sys_msg, user_msg, response, AIMessage(content=str(tool_output))])
                    st.write(final_response.content)
                    ai_content = final_response.content
        else:
            # No tools needed, just stream the regular chat
            full_res = st.write_stream(llm.stream([sys_msg] + st.session_state.chat_history + [user_msg]))
            ai_content = full_res

    # F. Save History
    st.session_state.chat_history.append(user_msg)
    st.session_state.chat_history.append(AIMessage(content=ai_content))
