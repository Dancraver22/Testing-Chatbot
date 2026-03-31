import streamlit as st
import base64
import pandas as pd
from datetime import datetime
from streamlit_javascript import st_javascript
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# --- IMPORT YOUR TOOLS ---
from tools import all_tools 

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="AI Prototype", layout="wide", page_icon="🌐")

# --- 2. IP & TIMEZONE DETECTION ---
user_tz = st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")

# --- 3. MODEL BINDING ---
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
llm_with_tools = llm.bind_tools(all_tools)

# --- 4. DATA & IMAGE UTILS ---
def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')

def process_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        # Optimization: We return a small string summary for the prompt
        return df, {"cols": list(df.columns), "rows": len(df), "sample": df.head(2).to_dict()}
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
    st.info(f"📍 Detected Time Zone: {user_tz if user_tz else 'Locating...'}")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- 6. RENDER HISTORY ---
chat_container = st.container()
with chat_container:
    for m in st.session_state.chat_history:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            # Optimization: Only render strings to keep the UI clean
            if isinstance(m.content, str):
                st.markdown(m.content)
            elif isinstance(m.content, list):
                for item in m.content:
                    if item["type"] == "text": st.markdown(item["text"])

# --- 7. EXECUTION ---
if user_input := st.chat_input("Ask me anything..."):
    
    # Process Files (Improvisation: Capture the data context)
    data_ctx = "No file uploaded."
    if uploaded_csv:
        _, summary = process_data(uploaded_csv)
        data_ctx = f"Data Summary: {summary}"

    # Build Content for the current turn
    u_content = [{"type": "text", "text": user_input}]
    if uploaded_img:
        b64 = encode_image(uploaded_img)
        u_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    # We use this for the actual LLM call
    u_msg_with_media = HumanMessage(content=u_content)
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            if uploaded_img: st.image(uploaded_img, width=300)

    # --- THE STRICT SYSTEM PROMPT (Optimized with your data_ctx) ---
    sys_prompt = SystemMessage(content=(
        f"CORE PERSONA: {personas[selected_persona]}\n\n"
        f"DATA_CONTEXT: {data_ctx}\n\n" # Optimization: AI can now 'see' the CSV
        "INSTRUCTIONS:\n"
        "1. You DO NOT know the current time or date. Do not guess.\n"
        "2. If the user asks for the time, you MUST use the 'get_world_clock' tool.\n"
        "3. If the user asks for facts, use 'fact_check_search'.\n"
        "4. Use the 'USER_TIMEZONE' below only as a hint for which city to check if they don't specify one.\n\n"
        f"REFERENCE - User's Home Timezone: {user_tz}\n"
        "Dont assume the time without checking the tool first"
        "Stay in character. Be grounded."
        "FACT-CHECKING RULES:\n"
        "1. When using 'fact_check_search', your response MUST be based ONLY on the search results provided.\n"
        "2. DO NOT use your own internal knowledge to 'correct' or add details to the search results.\n"
        "3. If the search result is missing a specific detail (like a specific town), say you don't know rather than guessing.\n"
        "4. If you see a specific name or date in the tool output, use it EXACTLY as written.\n\n"
        "Always check fact on wiki or any other search engine like Google before telling"
        "Stay in character, but prioritize accuracy over fluff."
    ))

    # AI Response
    with st.chat_message("assistant"):
        box = st.empty()
        # Optimization: history + current message (with image)
        full_context = [sys_prompt] + st.session_state.chat_history + [u_msg_with_media]
        call = llm_with_tools.invoke(full_context)
        
        if call.tool_calls:
            tool_msgs = [call]
            for t_call in call.tool_calls:
                t_map = {t.name: t for t in all_tools}
                target_tool = t_map[t_call["name"]]
                
                with st.status(f"🛠️ Agent checking {target_tool.name}...", expanded=False) as s:
                    obs = target_tool.invoke(t_call["args"])
                    s.update(label="✅ Verified", state="complete")
                    tool_msgs.append(ToolMessage(content=str(obs), tool_call_id=t_call["id"]))
                
                stream = llm.stream(full_context + tool_msgs)
                final_txt = box.write_stream(stream)
        else:
            final_txt = box.write_stream(llm.stream(full_context))

    # Save to history (Improvisation: Save text-only to prevent repeating images)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=final_txt))
    st.rerun()
