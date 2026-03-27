import streamlit as st
import base64
import pandas as pd
from PIL import Image
from io import BytesIO

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- IMPORT YOUR TOOLS ---
# Ensure tools.py is in the same folder on GitHub
from tools import all_tools 

# --- CONFIG ---
st.set_page_config(page_title="Global Vision AI", layout="wide", page_icon="🌐")

# --- 1. MODEL BINDING ---
# Llama-4-Scout: Optimized for reasoning and tool-calling
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5)
llm_with_tools = llm.bind_tools(all_tools)

# --- 2. UTILITIES ---
def encode_image(uploaded_file):
    """Encodes image to base64 for the Vision model"""
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

def process_data_file(file):
    """The Data Science Pipeline (Pandas Logic)"""
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

# --- 4. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 5. UI SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    persona = st.selectbox("Choose Persona", list(persona_prompts.keys()))
    uploaded_image = st.file_uploader("📸 Image Input", type=["jpg", "png"])
    uploaded_data = st.file_uploader("📊 Data Input (CSV/Excel)", type=["csv", "xlsx"])
    
    if st.button("Clear Memory"):
        st.session_state.chat_history = []
        st.rerun()

# --- 6. RENDER CHAT HISTORY ---
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        if isinstance(message.content, list):
            for item in message.content:
                if item["type"] == "text": st.markdown(item["text"])
        else:
            st.markdown(message.content)

# --- 7. MAIN EXECUTION LOOP ---
if user_input := st.chat_input("Ask about the data, the image, or the world..."):
    
    # A. Data Context Processing
    data_context = "No file uploaded."
    if uploaded_data:
        df, data_summary = process_data_file(uploaded_data)
        if df is not None:
            data_context = f"CURRENT_DATASET_SUMMARY: {data_summary}"
            st.info(f"📊 Dataset Loaded: {len(df)} rows found.")

    # B. Construct User Message (Immediately show in UI)
    user_content = [{"type": "text", "text": user_input}]
    if uploaded_image:
        base_64_img = encode_image(uploaded_image)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base_64_img}"}})
    
    user_msg = HumanMessage(content=user_content)
    
    with st.chat_message("user"):
        st.markdown(user_input)
        if uploaded_image:
            st.image(uploaded_image, width=300)

    # Save to history immediately to prevent UI lag
    st.session_state.chat_history.append(user_msg)

    # C. System Prompt Construction
    sys_msg = SystemMessage(content=(
        f"SYSTEM ROLE: {persona_prompts[persona]}\n"
        f"DATA_CONTEXT: {data_context}\n"
        "INSTRUCTIONS: Mirror user dialect (Manglish). Respond based on Data_Context if provided. "
        "Use 'get_world_clock' for ANY time query. Use 'fact_check_search' for news/facts. "
        "NEVER guess or calculate time/facts manually. ALWAYS use a tool."
    ))

    # D. Agent Reasoning & Tool Execution
    with st.chat_message("assistant"):
        # 1. First Call: AI decides which tool to use
        response = llm_with_tools.invoke([sys_msg] + st.session_state.chat_history)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Find tool by name
                selected_tool = next(t for t in all_tools if t.name == tool_call["name"])
                
                # Show tool activity to user
                with st.status(f"🛠️ Agent using {selected_tool.name}...", expanded=False) as status:
                    observation = selected_tool.func(**tool_call["args"])
                    status.update(label=f"✅ {selected_tool.name} completed!", state="complete")
                
                # 2. Final Pass: Use the tool result to write final answer
                final_stream = llm.stream([sys_msg] + st.session_state.chat_history + [response, AIMessage(content=str(observation))])
                ai_content = st.write_stream(final_stream)
        else:
            # Simple chat response if no tools needed
            ai_content = st.write_stream(llm.stream([sys_msg] + st.session_state.chat_history))

    # E. Save Response & Refresh UI
    st.session_state.chat_history.append(AIMessage(content=ai_content))
    st.rerun()
