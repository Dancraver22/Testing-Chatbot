import os
import pytz
import pandas as pd
from datetime import datetime
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from tavily import TavilyClient
import streamlit as st
from pydantic import BaseModel, Field

# --- 1. SCHEMAS (This prevents the TypeError by forcing strict inputs) ---
class QueryInput(BaseModel):
    query: str = Field(description="The search query for live facts or current events.")

class LocationInput(BaseModel):
    location: str = Field(description="The city or country name, e.g., 'Kuala Lumpur' or 'London'.")

class SaveInput(BaseModel):
    data: str = Field(description="The text content to be saved to a file.")

# --- 2. LIVE GROUNDING TOOL ---
@tool("fact_check_search", args_schema=QueryInput)
def fact_check_search(query: str):
    """
    Searches the live internet for verified facts, news, and CURRENT info. 
    Use this for any real-world facts to prevent hallucinations.
    """
    try:
        client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        results = client.search(query=query, search_depth="advanced", max_results=3, include_answer=True)
        
        direct_answer = results.get('answer', "")
        context = "\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results['results']])
        
        return f"Direct Answer: {direct_answer}\n\nSupporting Details:\n{context}"
    except Exception as e:
        return f"Search Error: {str(e)}"

# --- 3. DYNAMIC TIME CONVERTER ---
@tool("get_world_clock", args_schema=LocationInput)
def get_world_clock(location: str):
    """
    Returns the current time for ANY specific city or timezone.
    Input 'location' should be a city name (e.g., 'Mumbai', 'New York').
    """
    try:
        best_match = None
        # Standardize input for pytz (e.g., 'kuala lumpur' -> 'kuala_lumpur')
        search_term = location.strip().replace(" ", "_").lower()

        for tz in pytz.all_timezones:
            if search_term in tz.lower():
                best_match = tz
                break
        
        if not best_match:
            return f"Couldn't find precise timezone for {location}. Try a major capital city."

        target_tz = pytz.timezone(best_match)
        now = datetime.now(target_tz)
        return f"The current time in {location} ({best_match}) is {now.strftime('%I:%M %p')}."
    except Exception as e:
        return f"Error: {str(e)}"

# --- 4. WIKIPEDIA ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- 5. ARCHIVE TOOL ---
@tool("save_research_to_file", args_schema=SaveInput)
def save_research_to_file(data: str):
    """Saves text findings into a local .txt file. Use only if user asks to save/export."""
    try:
        filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)
        return f"✅ Archived to {filename}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

all_tools = [get_world_clock, fact_check_search, wiki_tool, save_research_to_file]
