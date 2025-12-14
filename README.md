# Chatbot---Implementation-using-Langgraph

# 🤖 AI Chatbot using LangGraph & Perplexity AI

A **production-ready AI chatbot** built using **FastAPI**, **LangGraph**, and **Perplexity AI**, featuring a **single-agent decision-based architecture** and a **Streamlit chat interface**.  
The system supports **real-time research, persona creation, content generation, and context-aware conversations**.

---

## 🔍 Key Highlights

- Implemented a **single intelligent agent** using **LangGraph**
- Integrated **Perplexity AI (sonar-pro)** for real-time factual research
- Built **intent detection, persona generation, and task execution** pipeline
- Developed a **FastAPI backend** with session-based memory handling
- Created a **Streamlit chat UI** for interactive conversations
- Secured API keys using **environment variables**
- Followed **Git & GitHub best practices**

---

## 🧠 Architecture

User
↓
Streamlit UI
↓
FastAPI API
↓
LangGraph (Single Agent)
├─ Decision Node
├─ Research Node (Perplexity)
├─ Persona Node
├─ Task Node
└─ Chat Node
↓
Response


---

## 🛠️ Tech Stack

- **Backend**: FastAPI  
- **Agent Framework**: LangGraph  
- **AI / Search**: Perplexity AI (sonar-pro)  
- **Frontend**: Streamlit  
- **Language**: Python  

---

## 📂 Project Structure



Agent/
├── main.py
├── streamlit_app.py
├── .gitignore
├── .env
└── README.md


---

## 🔐 Environment Setup

Create a `.env` file:

```env
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxx

▶️ Run the Project
Start Backend
uvicorn main:app --reload

Start Frontend
streamlit run streamlit_app.py

💬 Example Prompts

Create persona for Elon Musk

Tell me about Hyperledger Fabric

Write a professional follow-up email

Generate LinkedIn post about AI agents

📡 API Endpoint

POST /chat

{
  "message": "Your message"
}