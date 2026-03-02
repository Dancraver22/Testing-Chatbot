import streamlit as st
import os, requests, pytz
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Fact-Checked AI", page_icon="🎯", layout="wide")

# 1. THE TRUTH TOOLS
def get_verified_time():
    # Direct calculation to stop the 05:33 AM hallucination
    myt = pytz.timezone("Asia/Kuala_Lumpur")
    return datetime.now(myt).strftime('%I:%M %p, %A, %B %d, %Y')

# 2. CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, streaming=True)

# 3. SIDEBAR
with st.sidebar:
    st.title("⚙️ AI Controls")
    persona = st.selectbox("Persona:", ["Professional", "Sassy", "Emo"])
    if st.button("🗑️ Reset Brain"):
        st.session_state.chat_history = []
        st.rerun()

    persona_prompts = {
        "Professional": "You are a factual, elite assistant. Accuracy is mandatory.",
        "Sassy": "You are witty and sarcastic, but you NEVER lie about facts.",
        "Emo": "You are moody, but you see the cold, hard facts of reality."
    }

st.title(f"🤖 {persona} (Fact-Checked)")

for msg in st.session_state.chat_history:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 4. GROUNDED LOGIC
if user_input := st.chat_input("Ask a factual question..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        curr_time = get_verified_time()
        search_context = []
        
        # We always check facts if the question seems specific
        if tavily_api_key:
            with st.status("Verifying facts and sources...", expanded=False):
                tavily = TavilyClient(api_key=tavily_api_key)
                # Advanced search to distinguish Korean vs Malay food
                response = tavily.search(query=user_input, search_depth="advanced", max_results=3)
                search_context = response['results']

        # Formatting sources for the AI to use
        sources_text = "\n".join([f"- {res['title']}: {res['url']}" for res in search_context])
        content_text = "\n".join([res['content'] for res in search_context])

        # THE "NO-HALLUCINATION" PROMPT
        sys_msg = (
            f"{persona_prompts[persona]}\n"
            f"CURRENT_TIME: {curr_time}\n"
            f"WEB_SEARCH_DATA: {content_text}\n"
            f"SOURCES_TO_CITE: {sources_text}\n\n"
            "STRICT RULES:\n"
            "1. Use the provided WEB_SEARCH_DATA as the primary truth. If it says Naknak is Korean, you MUST say it is Korean.\n"
            "2. Citations: You MUST list the source URLs used at the end of your response under a 'Sources:' header.\n"
            "3. If the user asks for time, use ONLY the CURRENT_TIME provided above.\n"
            "4. If you don't find the answer in the search data, admit you don't know instead of guessing.\n"
            "5. Never say 'I don't have access'. You are looking at the data right now."
        )

        # 5. STREAMING WITH SOURCES
        full_response = st.write_stream(llm.stream([SystemMessage(content=sys_msg)] + st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=full_response))
