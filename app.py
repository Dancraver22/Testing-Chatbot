import streamlit as st
import base64
import pandas as pd
from PIL import Image
from io import BytesIO

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- IMPORT YOUR TOOLS ---
# This pulls the 'World Clock', 'Tavily Search', and 'Save' logic we built
from tools import all_tools 

# --- CONFIG ---
st.set_page_config(page_title="Global Vision AI", layout="wide", page_icon="🌐")

# --- 1. MODEL BINDING ---
# Llama-4-Scout is the best for NOT hallucinating when tools are present
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5)
llm_with_tools = llm.bind_tools(all_tools)

# --- 2. SPECIALIST UTILITIES (Pandas & Image Logic Kept!) ---
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

def process_data_file(file):
    """The Data Science Pipeline (Pandas/Numpy)"""
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
    "Professional": "Tech Consultant. Mirrored dialect. Polite but efficient.",
    "Sassy": "Witty friend. High energy. Use 'Abuden', 'Weh', and heavy Manglish slang.",
    "Emo": "Burnt-out KL Dev. Everything is 'sien' or 'koyak'. Low energy. Depressed."
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. UI SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    persona = st.selectbox("Choose Persona", list(persona_prompts.keys()))
    uploaded_image = st.file_uploader("📸 Image Input", type=["jpg", "png"])
    uploaded_data = st.file_uploader("📊 Data Input (CSV/Excel)", type=["csv", "xlsx"])
    if st.button("Clear Memory"):
        st.session_state.chat_history = []
        st.rerun()

# --- 5. CHAT INTERFACE ---
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        # Handle mixed content (text + images) in history
        if isinstance(message.content, list):
            for item in message.content:
                if item["type"] == "text": st.markdown(item["text"])
        else:
            st.markdown(message.content)

if user_input := st.chat_input("Ask about the data, the image, or the world..."):
    # A. Data Processing
    data_context = "No file uploaded."
    if uploaded_data:
        df, data_summary = process_data_file(uploaded_data)
        if df is not None:
            data_context = f"CURRENT_DATASET_SUMMARY: {data_summary}"
            st.info(f"📊 Dataset Loaded: {len(df)} rows found.")

    # B. Content Construction (Image + Text)
    content = [{"type": "text", "text": user_input}]
    if uploaded_image:
        base_4_img = encode_image(uploaded_image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base_4_img}"}})
        st.chat_message("user").image(uploaded_image, width=300)

    # C. System Message (Grounded Logic)
    sys_msg = SystemMessage(content=(
        f"SYSTEM ROLE: {persona_prompts[persona]}\n"
        f"DATA_CONTEXT: {data_context}\n"
        "INSTRUCTIONS: Mirror user dialect (Manglish). Respond based on Data_Context if provided. "
        "Use 'get_world_clock' for ANY time query. Use 'fact_check_search' for any news/facts. "
        "DO NOT GUESS. If you don't know, use a tool."
    ))

    # D. Agent Execution Loop
    with st.chat_message("assistant"):
        user_msg = HumanMessage(content=content)
        
        # 1. Decision Step: Brain decides which tool to use
        response = llm_with_tools.invoke([sys_msg] + st.session_state.chat_history + [user_msg])
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                selected_tool = next(t for t in all_tools if t.name == tool_call["name"])
                with st.spinner(f"🛠️ Agent using {selected_tool.name}..."):
                    observation = selected_tool.func(**tool_call["args"])
                    
                    # 2. Final Step: Stream the answer based on real data
                    final_stream = llm.stream([sys_msg, user_msg, response, AIMessage(content=str(observation))])
                    ai_content = st.write_stream(final_stream)
        else:
            # Direct response if no tools needed
            ai_content = st.write_stream(llm.stream([sys_msg] + st.session_state.chat_history + [user_msg]))

    # E. Save to History
    st.session_state.chat_history.append(user_msg)
    st.session_state.chat_history.append(AIMessage(content=ai_content))
