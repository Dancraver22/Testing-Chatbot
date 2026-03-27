import os
import pytz
import pandas as pd
from datetime import datetime
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from tavily import TavilyClient
import streamlit as st

# --- 1. LIVE GROUNDING TOOL ---
@tool
def fact_check_search(query: str):
    """
    USE THIS ONLY when the user asks for FACTS, NEWS, or CURRENT EVENTS.
    Do not use this for general conversation.
    """
    try:
        client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        results = client.search(query=query, search_depth="advanced", max_results=3, include_answer=True)
        
        direct_answer = results.get('answer', "")
        context = "\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results['results']])
        
        return f"Direct Answer: {direct_answer}\n\nSupporting Details:\n{context}"
    except Exception as e:
        return f"Search Error: {str(e)}"

# --- 2. DYNAMIC TIME CONVERTER ---
@tool
def get_world_clock(location: str):
    """
    USE THIS ONLY when the user EXPLICITLY asks 'what time is it' or 'current time'.
    Input 'location' must be a city name. 
    DO NOT use this just because a city is mentioned in passing.
    """
    try:
        best_match = None
        search_term = location.strip().replace(" ", "_").lower()
        
        for tz in pytz.all_timezones:
            if search_term in tz.lower():
                best_match = tz
                break
        
        if not best_match:
            return f"Timezone for {location} not found."

        target_tz = pytz.timezone(best_match)
        now = datetime.now(target_tz)
        return f"The current time in {location} is {now.strftime('%I:%M %p')}."
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. WIKIPEDIA ---
try:
    # Adding a check to see if wikipedia is actually installed
    import wikipedia
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
except ImportError:
    wiki_tool = None

# --- 4. ARCHIVE TOOL ---
@tool
def save_research_to_file(data: str):
    """Saves text findings into a local .txt file. Use only if user asks to save/export."""
    try:
        filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)
        return f"✅ Archived to {filename}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

all_tools = [get_world_clock, fact_check_search, save_research_to_file]
if wiki_tool:
    all_tools.append(wiki_tool)
