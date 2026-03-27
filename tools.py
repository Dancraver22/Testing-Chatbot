import os
import pytz
import pandas as pd
from datetime import datetime
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from tavily import TavilyClient
import streamlit as st

# --- 1. GLOBAL TIME TOOL (Prevents Hallucination) ---
def get_global_time(location: str = "Asia/Kuala_Lumpur"):
    """
    Returns the current time for a specific location.
    Default is Malaysia, but handles 'America/New_York', 'Europe/London', etc.
    Input should be an IANA timezone string.
    """
    try:
        tz = pytz.timezone(location)
        now = datetime.now(tz)
        return f"The current time in {location} is {now.strftime('%I:%M %p')} ({tz.zone})."
    except Exception:
        # Fallback if the AI provides a city name instead of a timezone string
        return "Error: Please provide a valid IANA timezone string (e.g., 'Asia/Kuala_Lumpur')."

time_tool = Tool(
    name="get_world_clock",
    func=get_global_time,
    description="Use this to find the current time in ANY specific city or country worldwide."
)

# --- 2. TAVILY SEARCH TOOL (Grounded & Accurate) ---
def tavily_search_tool(query: str):
    """
    Searches the live internet for verified, fact-grounded information.
    Use this for news, current prices, technical specs, or real-time data.
    """
    try:
        client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        # 'advanced' search provides cleaner, AI-optimized context
        results = client.search(query=query, search_depth="advanced", max_results=3)
        
        context = "\n".join([
            f"Source: {r['url']}\nContent: {r['content']}" 
            for r in results['results']
        ])
        return context if context else "No relevant search results found."
    except Exception as e:
        return f"Search Error: {e}"

search_tool = Tool(
    name="fact_check_search",
    func=tavily_search_tool,
    description="Essential for real-time facts, news, and verified data to prevent hallucinations."
)

# --- 3. WIKIPEDIA TOOL (Entity & History Expert) ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- 4. DATA PERSISTENCE TOOL (The Secretary) ---
def save_research_to_file(data: str):
    """
    Saves structured text or research findings into a local .txt file.
    Use this ONLY when the user explicitly asks to 'save', 'export', or 'archive'.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"--- AI GENERATED RESEARCH LOG ---\nTimestamp: {datetime.now()}\n\n{data}")
        return f"✅ Success: Data archived to {filename}"
    except Exception as e:
        return f"❌ System Error saving file: {e}"

save_tool = Tool(
    name="save_to_file",
    func=save_research_to_file,
    description="Saves information to a permanent local text file for future reference."
)

# --- 5. EXPORT LIST ---
# This list is imported by app.py to initialize the Agent
all_tools = [time_tool, search_tool, wiki_tool, save_tool]
