import streamlit as st
import base64
import pandas as pd
from datetime import datetime
from streamlit_javascript import st_javascript
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- IMPORT YOUR TOOLS ---
from tools import all_tools 

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="AI Prototype", layout="wide", page_icon="🌐")

# --- 2. IP & TIMEZONE DETECTION ---
# Detects user's actual location via browser JS
user_tz = st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")

# --- 3. MODEL BINDING ---
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
llm_with_tools = llm.bind_tools(all_tools)

# --- 4. DATA & IMAGE UTILS (Kept all your features!) ---
def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')

def process_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        return df, {"cols": list(df.columns), "rows": len(df)}
    except Exception as e:
        return None, str(e)

# Personas
personas = {
    "Professional": "Professional. Efficient and polite.",
    "Sassy": "Witty friend. sassy, caring and Manglish slang.",
    "Emo": "Depressed. No hope in life. Low energy."
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    selected_persona = st.selectbox("Persona", list(personas.keys()))
    uploaded_img = st.file_uploader("📸 Image", type=["jpg", "png"])
    uploaded_csv = st.file_uploader("📊 Data", type=["csv", "xlsx"])
    st.info(f"📍 Detected TZ: {user_tz if user_tz else 'Locating...'}")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- 6. RENDER HISTORY ---
chat_container = st.container()
with chat_container:
    for m in st.session_state.chat_history:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            if isinstance(m.content, list):
                for item in m.content:
                    if item["type"] == "text": st.markdown(item["text"])
            else:
                st.markdown(m.content)

# --- 7. EXECUTION ---
if user_input := st.chat_input("Ask me anything..."):
    
    # Process Files
    data_ctx = "No file."
    if uploaded_csv:
        _, summary = process_data(uploaded_csv)
        data_ctx = str(summary)

    # Build Message
    u_content = [{"type": "text", "text": user_input}]
    if uploaded_img:
        b64 = encode_image(uploaded_img)
        u_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    u_msg = HumanMessage(content=u_content)
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            if uploaded_img: st.image(uploaded_img, width=300)

    st.session_state.chat_history.append(u_msg)

    # System Prompt with IP Awareness
    sys_prompt = SystemMessage(content=(
        f"ROLE: {personas[selected_persona]}\n"
        f"USER_TIMEZONE: {user_tz}\n"
        f"DATA_CONTEXT: {data_ctx}\n"
        f"CURRENT_DATE: {datetime.now().strftime('%Y-%m-%d')}\n"
        "RULES: Use 'get_world_clock' for time. Use 'fact_check_search' for any live facts. "
        "DO NOT GUESS. Mirror user dialect. Be grounded."
    ))

    # AI Response
    with st.chat_message("assistant"):
        box = st.empty()
        # Step 1: Tool Call Decision
        call = llm_with_tools.invoke([sys_prompt] + st.session_state.chat_history)
        
        if call.tool_calls:
            for t_call in call.tool_calls:
                t_map = {t.name: t for t in all_tools}
                target_tool = t_map[t_call["name"]]
                
                with st.status(f"🛠️ Agent checking {target_tool.name}...", expanded=False) as s:
                    obs = target_tool.invoke(t_call["args"])
                    s.update(label="✅ Data Verified", state="complete")
                
                # Step 2: Final Stream
                stream = llm.stream([sys_prompt] + st.session_state.chat_history + [call, AIMessage(content=str(obs))])
                final_txt = box.write_stream(stream)
        else:
            final_txt = box.write_stream(llm.stream([sys_prompt] + st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=final_txt))
    st.rerun()
