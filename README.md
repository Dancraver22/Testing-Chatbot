---
title: RAG Prototype Backend
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

🤖 Global Vision AI: Multimodal RAG & Data Operations Agent
Live Demo: [View Prototype](https://rag-agent-prototype-ecvzz8hjn2aauzprcae5sc.streamlit.app/)

📖 Project Overview
Global Vision AI is a High-Performance AI Orchestration Framework built to bridge the gap between Large Language Models (LLMs) and structured business data. Unlike standard chatbots, this system implements a Grounded RAG (Retrieval-Augmented Generation) pipeline that combines live web intelligence with local data analytics.

Designed for the 2026 tech landscape, this agent focuses on Technical Agility: shifting between complex data processing, visual recognition, and high-context conversational personas.

🚀 Core Technical Capabilities
🧠 Multimodal Intelligence & Vision
Computer Vision Integration: Implements asynchronous image encoding (Base64) to allow the Llama 4 Scout architecture to analyze visual data in real-time.

NLP Sentiment Engine: Utilizes a local distilbert-base-uncased transformer pipeline to detect user mood and dynamically adjust response parameters without increasing API latency.

📊 Data Operations (Pandas & Numpy)
Automated Data Ingestion: Integrated a specialized pipeline for CSV/Excel processing.

Contextual Summarization: Uses Pandas to generate metadata snapshots (statistical descriptions, head/tail samples) to provide the LLM with a structured "source of truth," eliminating hallucinations in data-heavy queries.

🔍 Grounded RAG & Web Orchestration
Internet-Augmented Retrieval: Leverages the Tavily Search API for real-time fact-checking and news retrieval.

Intent-Based Search Triggering: Implements a keyword-density classifier to determine when an external search is architecturally necessary, optimizing token usage.

🎭 Dynamic Persona Engine
Context-Aware Personalization: A modular system that switches between Professional (Tech Consultant), Sassy (Social/Slang), and Internal Dev (Technical/Burnout) styles, mirroring local Malaysian dialects (Manglish/Rojak) for enhanced user engagement.

🛠️ Specialist Tech Stack
Orchestration: LangChain (Message History & System Prompt Engineering)

Inference Power: Groq (Meta-Llama-4-Scout-17B Architecture)

Data Science: Pandas, NumPy, OpenPyXL

Machine Learning: Hugging Face Transformers (PyTorch)

Infrastructure: Streamlit (UI), REST APIs, IANA Timezone Management, Base64 Image Processing

📈 Engineering Roadmap
[COMPLETED] Integrated Pandas/Numpy for structured data analysis.

[IN PROGRESS] Transitioning from temporary st.session_state to Vector Database Persistence (ChromaDB/Pinecone) for long-term user memory.

[UPCOMING] Implementation of FastAPI to decouple the backend logic from the UI, turning the agent into a scalable microservice.
