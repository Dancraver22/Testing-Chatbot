🤖 Multi-Persona AI Orchestration Agent
A Robust Python Application for Fact-Grounded Conversational AI

📖 Project Overview
This project demonstrates a production-ready AI Agent built with Python and LangChain. It is designed to bridge the gap between Large Language Models (LLMs) and real-world data sources, providing a framework that is both accurate and adaptable to different professional use cases.

🚀 Technical Core Capabilities
Dynamic Persona Engine: Features a modular prompt system allowing the agent to switch between various communication styles (Professional, Creative, and Technical) without losing context.

RAG-Driven Accuracy: Utilizes Retrieval-Augmented Generation (RAG) via the Tavily Search API to anchor responses in live, verified data.

Intelligent Tool Calling: The agent autonomously determines when to utilize external tools (Google Search, Wikipedia, Custom Converters) based on user intent.

Optimized Memory Management: Implements a sliding-window message history to maintain stability, manage Context Window limits, and control API token costs.

🛠️ Tech Stack & Integration
Framework: LangChain

Model Provider: Groq (Llama 3.1 Architecture)

Frontend: Streamlit with Custom CSS injection for enhanced UX

Data Protocols: REST APIs (Tavily, Wikipedia) and IANA timezone management

⚙️ Deployment
This application is fully containerized for deployment on Streamlit Community Cloud, utilizing secure secrets management for API credentials.
