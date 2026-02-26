import streamlit as st
import datetime
import pytz
import os
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool

# Ensure your tools are correctly imported
from tools import search_tool, wiki_tool, save_tool

# --- 1. DYNAMIC GLOBAL TOOLS ---

@tool
def universal_converter(query: str) -> str:
    """Provides high-precision global time and unit conversions via search."""
    try:
        search_query = f"IANA timezone ID for {query}"
        search_result = search_tool.run(search_query)
        target_zone = next((tz for tz in pytz.all_timezones if tz.lower() in search_result.lower()), None)
        
        if target_zone:
            # Fixing the 11-hour drift by anchoring to UTC
            utc_now = datetime.datetime.now(pytz.utc)
            local_time = utc_now.astimezone(pytz.timezone(target_zone))
            return f"{local_time.strftime('%I:%M %p, %A')} in {target_zone}"
        return "I can't find that place. Is it in another galaxy?"
    except Exception:
        return "Clock's broken. Try again in a sec."

# --- 2. PERSONALITY & UI ---
st.set_page_config(page_title="The Zesty Agent", page_icon="💅")

with st.sidebar:
    st.title("💅 Vibe Check")
    personality_type = st.selectbox("Choose Persona:", ["Sassy", "Zesty", "Professional"])
    
    prompts = {
        "Sassy": "You are witty, sarcastic, and judgmental. Use 🙄. Never be a boring AI.",
        "Zesty": "You are flamboyant and use TONS of emojis! ✨🌈 Everything is tea! ☕️",
        "Professional": "You are a helpful, polite business assistant."
    }
    
    if st.button("Hard Reset"):
        st.session_state.clear()
        st.rerun()

st.title(f"🎭 {personality_type} Assistant")

# --- 3. AGENT CONFIG (Web-Optimized) ---
# Using 8B model to stay under Rate Limits on the web
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8) 
tools = [search_tool, wiki_tool, save_tool, universal_converter]

prompt = ChatPromptTemplate.from_messages([
    ("system", f"{prompts[personality_type]} Rules: 1. Be brief. 2. Use tools for facts. 3. Stay in character."),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. MEMORY (Limit-Proof) ---
msgs = StreamlitChatMessageHistory(key="chat_messages")
if len(msgs.messages) > 4:
    msgs.messages = msgs.messages[-4:] # Keeps the link from crashing

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: msgs,
    input_messages_key="query",
    history_messages_key="chat_history",
)

# --- 5. INTERFACE ---
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if user_query := st.chat_input("Tell me something spicy..."):
    st.chat_message("human").write(user_query)
    with st.chat_message("assistant"):
        try:
            response = agent_with_history.invoke(
                {"query": user_query},
                config={"configurable": {"session_id": "web_session"}}
            )
            st.write(response["output"])
        except Exception:
            st.error("The link is hitting its limits! Wait 30 seconds.")