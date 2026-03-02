🤖 Multi-Persona AI Orchestration Agent
A Python Framework for Fact-Grounded Conversational AI
📖 Project Overview
This project is an Active Development Prototype built with Python and LangChain. It is designed to bridge the gap between Large Language Models (LLMs) and real-world data sources, providing a framework that is both accurate and adaptable.

The primary goal of this project is to solve the "Hallucination Problem" in AI by implementing a Grounded RAG (Retrieval-Augmented Generation) pipeline.

🚀 Technical Core Capabilities
RAG-Driven Accuracy: Utilizes Retrieval-Augmented Generation via the Tavily Search API to anchor responses in live, verified data.

Dynamic Persona Engine: Features a modular prompt system allowing the agent to switch between Professional, Creative, and Technical communication styles.

Intelligent Tool Calling: The agent determines when to utilize external tools (Google Search, Wikipedia) based on user intent.

Optimized Memory Management: Implements a sliding-window message history to manage Context Window limits and control API token costs.

🛠️ Tech Stack & Integration
Framework: LangChain

Model Provider: Groq (Llama 3.1 Architecture)

Frontend: Streamlit with Custom CSS injection for enhanced UX

Data Protocols: REST APIs (Tavily, Wikipedia) and IANA timezone management

⚠️ Current Status & Future Roadmap (WIP)
As this project is in the Initial Prototype Phase, I am currently focused on the following iterations:

Logic Refinement: Improving the "Intent Classifier" to reduce redundant search triggers for casual conversation.

Latency Optimization: Recently achieved an 80% reduction in boot time by refactoring environment dependencies.

Data Persistence: Researching the integration of Vector Databases (e.g., Pinecone) to replace temporary session-based memory.
