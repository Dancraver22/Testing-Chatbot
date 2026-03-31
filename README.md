---
title: RAG Prototype Backend
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🤖 Global Vision AI: Multimodal RAG & Data Operations Agent

**Live Backend:** [Hugging Face Space](https://huggingface.co/spaces/Dancraver/RAG_Prototype)  
**Live Demo:** [View Streamlit Prototype](https://rag-agent-prototype-ecvzz8hjn2aauzprcae5sc.streamlit.app/)

## 📖 Project Overview
Global Vision AI is a high-performance AI orchestration framework designed for the 2026 tech landscape. Built by a **Technical Artist**, this system bridges the gap between Large Language Models (LLMs) and structured technical data. It implements a **Grounded RAG (Retrieval-Augmented Generation)** pipeline that combines live web intelligence with local data analytics and visual recognition.

## 🚀 Core Technical Capabilities

### 🧠 Multimodal Intelligence & Vision
* **Computer Vision Integration:** Implements asynchronous image encoding (Base64) allowing the **Llama 4 Scout** architecture to analyze visual data in real-time.
* **NLP Sentiment Engine:** Utilizes local transformers to detect user intent and dynamically adjust response parameters, mirroring local Malaysian dialects (Manglish/Rojak) for enhanced engagement.

### 📊 Data Operations (Pandas & Vector Memory)
* **Automated Data Ingestion:** Specialized pipeline for CSV/Excel processing using Pandas to generate "source of truth" metadata snapshots.
* **Vector Persistence:** Integrated **ChromaDB** for long-term memory, ensuring the agent "remembers" technical context across sessions.

## 📈 Engineering Roadmap
- [x] **Phase 1:** Integrated Pandas/Numpy for structured data analysis.
- [x] **Phase 2:** Migrated to **Dockerized FastAPI** backend for production stability.
- [x] **Phase 3:** Implemented **Vector Database Persistence (ChromaDB)** for long-term memory.
- [ ] **Phase 4:** **[IN PROGRESS]** Refactoring Image-to-Text Pipeline to improve visual technical description accuracy.
- [ ] **Phase 5:** **[UPCOMING]** Deploying specialized sub-agents for dedicated Shader/VFX code auditing.

---

## ⚠️ Prototype Disclaimer & Liability
**This project is a technical prototype and is currently under active development.**

* **Accuracy:** As an AI-driven system, the agent may occasionally generate "hallucinations" or inaccurate data summaries. Always verify critical technical data manually.
* **Liability:** The developer is not responsible for any decisions made based on the AI's output or any data loss occurring during the use of this prototype.
* **Data Privacy:** This prototype uses third-party APIs (Groq, Tavily). Avoid uploading sensitive or proprietary corporate data.
* **Continuous Improvement:** We are constantly refining the RAG retrieval logic and vision processing to minimize errors.

---
