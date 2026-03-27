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
user_tz = st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")

# --- 3. MODEL BINDING ---
# Using a lower temperature (0.1) is good for grounding, but we need the prompt to be less "pushy"
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
llm_with_tools = llm.bind_tools(all_tools)

# --- 4. DATA & IMAGE UTILS ---
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
    "Professional": "You are a professional technical assistant. Be efficient, polite, and direct.",
    "Sassy": "You are a cheerful slay. Use Manglish. Be sassy but helpful.",
    "Emo": "You are a depressed. Everything is a burden. Low energy, no hope, but you'll answer if you must."
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
    data_ctx = "No file uploaded."
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

   # --- THE STRICT SYSTEM PROMPT ---
    sys_prompt = SystemMessage(content=(
        f"CORE PERSONA: {personas[selected_persona]}\n\n"
        "INSTRUCTIONS:\n"
        "1. You DO NOT know the current time or date. Do not guess.\n"
        "2. If the user asks for the time, you MUST use the 'get_world_clock' tool.\n"
        "3. If the user asks for facts, use 'fact_check_search'.\n"
        "4. Use the 'USER_TIMEZONE' below only as a hint for which city to check if they don't specify one.\n\n"
        f"REFERENCE - User's Home Timezone: {user_tz}\n"
        "Dont assume the time without checking the tool first"
        "Stay in character. Be grounded."
    ))

    # AI Response
    with st.chat_message("assistant"):
        box = st.empty()
        # Step 1: Tool Call Decision
        call = llm_with_tools.invoke([sys_prompt] + st.session_state.chat_history)
        
        if call.tool_calls:
            # We add a list to store the tool results to maintain the conversation flow
            tool_msgs = [call]
            for t_call in call.tool_calls:
                t_map = {t.name: t for t in all_tools}
                target_tool = t_map[t_call["name"]]
                
                with st.status(f"🛠️ Agent checking {target_tool.name}...", expanded=False) as s:
                    obs = target_tool.invoke(t_call["args"])
                    s.update(label="✅ Verified", state="complete")
                    # Proper LangChain ToolMessage handling
                    from langchain_core.messages import ToolMessage
                    tool_msgs.append(ToolMessage(content=str(obs), tool_call_id=t_call["id"]))
                
                # Step 2: Final Stream using the tool output
                stream = llm.stream([sys_prompt] + st.session_state.chat_history + tool_msgs)
                final_txt = box.write_stream(stream)
        else:
            final_txt = box.write_stream(llm.stream([sys_prompt] + st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=final_txt))
    st.rerun()
