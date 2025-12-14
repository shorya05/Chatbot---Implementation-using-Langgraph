
import datetime
import tempfile
import json
import time
from functools import wraps
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union
from dotenv import load_dotenv
from typing import TypedDict, Any

# Create FastAPI app instance
app = FastAPI()

import datetime
import tempfile
import json
import time
from functools import wraps
import os, re, requests
from fastapi import FastAPI, File, Form, UploadFile,Request
import io, base64, os, re
from collections import Counter
# --- LangChain Imports ---

# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

# from langsmith import Client as LangSmithClient
# from langchain.callbacks import LangChainTracer
# --- Imports ---
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import sys
from typing import Optional 
# import httpx
from typing import List, Dict, Any
# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from fastapi.responses import HTMLResponse
#Image Generation
# from PIL import Image
# from google.genai import types
import uuid
# from io import BytesIO
import os
import requests
import re
import json

from dotenv import load_dotenv
load_dotenv()

print("PERPLEXITY_API_KEY:", os.getenv("PERPLEXITY_API_KEY"))

def call_perplexity(prompt: str, max_tokens: int = 1200) -> str:
    """
    Central helper to call Perplexity AI (sonar-pro).
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("❌ PERPLEXITY_API_KEY not set in environment")

    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)

    # Raise HTTP error if any
    response.raise_for_status()

    data = response.json()

    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    # Remove code blocks if Perplexity adds them
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    return text

class ResearchRequest(BaseModel):
    topic: str
    competitors: List[str]
    target_keywords: List[str] = []


# Configure Gemini

# --- SESSION MEMORY ---
session_memory: dict[str, dict[str, Any]] = {}

# --- REQUEST MODEL ---
class ChatRequest(BaseModel):
    message: str


def extract_context_from_message(message: str) -> dict:
    prompt = f"""
Analyze the user message and return ONLY valid JSON.

Keys:
- topic: main company/person/product (or empty string)
- intent: one of ["who","create_persona","task","general"]

Rules:
- "task" = write, draft, generate, email, post, presentation
- "create_persona" = explicitly says create persona/profile
- "who" = asking identity
- else general

Message:
"{message}"

Return JSON only.
"""

    try:
        text = call_perplexity(prompt, 400)
        return json.loads(text)
    except:
        return {"topic": "", "intent": "general"}

# STEP 2 — PERPLEXITY RESEARCH
def perplexity_search(topic: str, user_message: str = "") -> str:
    if not topic:
        return ""

    prompt = f"""
Provide factual research about "{topic}".

Include:
- Background
- Skills / Expertise (if person)
- Products / Services (if company)
- Achievements
- Current focus

Be concise but informative.
"""
    return call_perplexity(prompt, 1200)

# def perplexity_search(topic: str, user_message: str = "") -> str:
#     """
#     Fetches research dynamically from Perplexity AI.
#     Handles both web search and user-provided context.
#     """
#     if not topic or len(topic.strip()) < 2:
#         return ""
    
#     # Check if user provided detailed context in message (more than 100 chars of description)
#     # This indicates they've pasted profile/company info
#     if len(user_message) > 150:
#         # User has provided detailed context, use it directly
#         print("Using user-provided context instead of API search")
#         return user_message
    
#     # Check if user message contains LinkedIn URL
#     linkedin_url_match = re.search(r'https?://(?:www\.)?linkedin\.com/in/[\w-]+/?', user_message or topic)
    
#     url = "https://api.perplexity.ai/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
#         "Content-Type": "application/json"
#     }
    
#     # Customize prompt based on whether it's a person or company
#     if linkedin_url_match:
#         linkedin_url = linkedin_url_match.group()
#         # Note: Perplexity may not be able to access LinkedIn directly due to auth
#         content_prompt = f"""Search the web for professional information about {topic}. 
# Note: LinkedIn profile URL provided is {linkedin_url} but may not be directly accessible.
# Look for:
# - Professional background and experience
# - Skills and expertise areas
# - Education and qualifications
# - Current role or career stage
# - Notable achievements or projects

# Provide a comprehensive professional overview."""
#     elif any(keyword in user_message.lower() for keyword in ["student", "person", "profile", "individual"]):
#         content_prompt = f"Search for professional information about {topic}. Include their background, skills, experience, education, achievements, and professional positioning. Create a comprehensive professional overview."
#     else:
#         content_prompt = f"Provide a detailed factual overview about {topic}. Include: what it is, main products/services, key features, and recent developments. Be specific and informative."
    
#     payload = {
#         "model": "sonar-pro",
#         "messages": [
#             {
#                 "role": "user", 
#                 "content": content_prompt
#             }
#         ],
#         "max_tokens": 1500,
#         "temperature": 0.2
#     }
#     try:
#         response = requests.post(url, headers=headers, json=payload, timeout=25)
#         response.raise_for_status()
#         data = response.json()
#         result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
#         # Clean up any code blocks
#         result = re.sub(r"```.*?```", "", result, flags=re.DOTALL).strip()
#         return result if result else f"Limited information found for '{topic}'. Please provide more details about {topic} for better persona creation."
#     except Exception as e:
#         print(f"Perplexity error: {e}")
#         return f"Could not fetch external information for '{topic}'. Please provide profile details manually for accurate persona creation."


# # STEP 3 — PERSONA BUILDER
# def build_persona_from_research(research_text: str, topic: str, user_message: str) -> dict:
#     """
#     Builds a persona context dynamically from user message and research.
#     Uses Gemini to create structured persona from the research data.
#     """
#     model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
#     prompt = f"""Based on the following information, create a detailed professional persona.

# Entity/Person: {topic}
# User Context: {user_message}
# Research Data:
# {research_text}

# Create a structured professional persona with these sections:
# 1. Professional Identity (2-3 sentences describing who they are)
# 2. Core Skills & Expertise (bullet points)
# 3. Background & Experience (brief overview)
# 4. Current Focus & Goals (what they're working on/toward)
# 5. Key Strengths (what makes them stand out)

# Keep it professional, factual, and well-structured. Format with clear sections.
# """
    
#     try:
#         response = model.generate_content(prompt)
#         persona_text = response.text.strip()
        
#         return {
#             "name": topic or "Unknown Entity",
#             "summary": research_text,
#             "persona_profile": persona_text,
#             "core_identity": f"You are an AI assistant representing {topic or 'this professional'}. Use the detailed persona information to provide accurate, insightful responses about their professional background, skills, and capabilities. Maintain a professional and knowledgeable tone."
#         }
#     except Exception as e:
#         print(f"Persona building error: {e}")
#         return {
#             "name": topic or "Unknown Entity",
#             "summary": research_text,
#             "persona_profile": research_text,
#             "core_identity": f"You are an AI assistant representing {topic or 'the entity mentioned'}. Use the research data to provide accurate, helpful information."
#         }

def build_persona_from_research(research_text: str, topic: str) -> dict:
    prompt = f"""
Using the research below, create a professional persona.

Entity: {topic}

Research:
{research_text}

Return persona in this JSON format:
{{
  "name": "{topic}",
  "professional_identity": "",
  "skills": [],
  "experience": "",
  "goals": "",
  "strengths": []
}}

Return JSON only.
"""

    try:
        persona_json = call_perplexity(prompt, 800)
        return json.loads(persona_json)
    except:
        return {"name": topic, "summary": research_text}


# # STEP 4 — GEMINI RESPONSE GENERATOR
# def gemini_generate_reply(message: str, persona: dict, history: List[Dict]) -> str:
#     """
#     Uses Gemini to respond conversationally using persona + short history.
#     """
#     model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
#     # Format history
#     history_text = ""
#     if history:
#         recent_history = history[-6:]  # Last 6 exchanges
#         history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['ai']}" for h in recent_history])
    
#     name = persona.get('name', '')
#     summary = persona.get('summary', '')
#     persona_profile = persona.get('persona_profile', '')
#     core_identity = persona.get('core_identity', '')
    
#     # Build prompt based on whether we have persona or not
#     if name and (summary or persona_profile):
#         prompt = f"""You are an AI assistant representing {name}.

# Professional Persona:
# {persona_profile if persona_profile else summary}

# {core_identity}

# Previous Conversation:
# {history_text}

# Current User Message:
# {message}

# Instructions:
# - Respond naturally and helpfully based on the persona information provided
# - If asked "who are you", provide the professional identity from the persona
# - Use the research/persona data to provide accurate, relevant information
# - Keep responses conversational and professional
# - Speak AS or ABOUT this professional/entity (based on context)

# Your response:"""
#     else:
#         prompt = f"""You are a helpful AI assistant.

# Previous Conversation:
# {history_text}

# User Message:
# {message}

# Respond naturally and helpfully. If asked who you are, say you're an AI assistant ready to help with various tasks.

# Your response:"""

#     try:
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Gemini generation error: {e}")
#         return "Sorry, I encountered an error generating a response. Please try again."

def generate_chat_reply(message: str, persona: dict, history: list) -> str:
    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['ai']}" for h in history[-6:]]
    )

    prompt = f"""
You are an AI assistant.

Persona:
{json.dumps(persona, indent=2)}

Conversation so far:
{history_text}

User message:
{message}

Respond naturally and professionally.
"""

    return call_perplexity(prompt, 800)


def generate_task_content(message: str, history: list, persona: dict) -> str:
    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['ai']}" for h in history[-4:]]
    )

    prompt = f"""
Create the requested content.

Persona Context:
{json.dumps(persona, indent=2)}

Previous conversation:
{history_text}

User request:
{message}

Rules:
- If email → include Subject, greeting, body, closing
- If info missing → use placeholders
- Make it ready to send/use
"""

    return call_perplexity(prompt, 1200)

# STEP 5 — LangGraph State
class AgentState(TypedDict):
    message: str
    topic: str
    intent: str
    research_text: str
    persona: dict
    history: list
    generated_content: str


# LangGraph Nodes
# def decision_node(state: AgentState) -> AgentState:
#     """Extract context and decide next step"""
#     context = extract_context_from_message(state["message"])
#     state["topic"] = context["topic"]
#     state["intent"] = context["intent"]
#     return state

def decision_node(state):
    ctx = extract_context_from_message(state["message"])
    state["topic"] = ctx["topic"]
    state["intent"] = ctx["intent"]
    return state

def who_are_you_node(state: AgentState) -> AgentState:
    """Handle 'who are you' queries"""
    state["generated_content"] = "I'm an AI assistant here to help you with information, tasks, and conversations. I can research topics, create personas, draft content, and more. How can I assist you today?"
    return state


# def research_node(state: AgentState) -> AgentState:
#     """Fetch research from Perplexity"""
#     research = perplexity_search(state["topic"], state.get("message", ""))
#     state["research_text"] = research
#     return state


# def persona_node(state: AgentState) -> AgentState:
#     """Build persona from research"""
#     persona = build_persona_from_research(
#         state["research_text"],
#         state["topic"],
#         state["message"]
#     )
#     state["persona"] = persona
#     return state


# def chat_node(state: AgentState) -> AgentState:
#     """Generate conversational response"""
#     reply = gemini_generate_reply(
#         state["message"], 
#         state.get("persona", {}), 
#         state.get("history", [])
#     )
#     state["generated_content"] = reply
#     return state

def research_node(state):
    state["research_text"] = perplexity_search(state["topic"], state["message"])
    return state

def persona_node(state):
    state["persona"] = build_persona_from_research(
        state["research_text"], state["topic"]
    )
    return state

def chat_node(state):
    state["generated_content"] = generate_chat_reply(
        state["message"],
        state.get("persona", {}),
        state.get("history", [])
    )
    return state

# def task_node(state: AgentState) -> AgentState:
#     """Handle task requests - generate actual content"""
#     model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
#     message = state["message"]
#     history = state.get("history", [])
#     persona = state.get("persona", {})
    
#     # Format history
#     history_text = ""
#     if history:
#         recent_history = history[-4:]
#         history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['ai']}" for h in recent_history])
    
#     # Build context-aware prompt
#     company_context = ""
#     if persona.get("company_name") and persona.get("company_summary"):
#         company_context = f"\nYou are representing: {persona['company_name']}\nCompany Info: {persona['company_summary'][:500]}\n"
    
#     prompt = f"""You are a helpful AI assistant that creates professional content.
# {company_context}
# Previous Conversation:
# {history_text}

# User Request:
# {message}

# Instructions:
# - Analyze what the user is asking for (email, promo, presentation, etc.)
# - Generate the actual content they requested
# - Be professional and complete
# - If it's an email, include Subject, proper greeting, body, and closing
# - If context is missing (like recipient name, purpose), use placeholders like [Recipient Name], [Meeting Purpose]
# - Make it ready to use

# Generate the requested content now:"""

#     try:
#         response = model.generate_content(prompt)
#         state["generated_content"] = response.text.strip()
#     except Exception as e:
#         print(f"Task generation error: {e}")
#         state["generated_content"] = "I can help with tasks like drafting emails, creating marketing content, or writing presentations. Could you provide more details about what you need?"
    
#     return state

def task_node(state):
    state["generated_content"] = generate_task_content(
        state["message"],
        state.get("history", []),
        state.get("persona", {})
    )
    return state

# Conditional routing function
def route_after_decision(state: AgentState) -> str:
    """Route to appropriate node based on intent"""
    intent = state.get("intent", "general")
    
    if intent == "who":
        return "who_are_you"
    elif intent == "create_persona":
        return "research"
    elif intent == "task":
        return "task"
    else:
        # For general queries with topic, do research
        if state.get("topic"):
            return "research"
        else:
            return "chat"


# Build LangGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decision", decision_node)
workflow.add_node("who_are_you", who_are_you_node)
workflow.add_node("research", research_node)
workflow.add_node("persona", persona_node)
workflow.add_node("chat", chat_node)
workflow.add_node("task", task_node)

# Set entry point
workflow.set_entry_point("decision")

# Add conditional edges from decision
workflow.add_conditional_edges(
    "decision",
    route_after_decision,
    {
        "who_are_you": "who_are_you",
        "research": "research",
        "task": "task",
        "chat": "chat"
    }
)

# Add sequential edges
workflow.add_edge("research", "persona")
workflow.add_edge("persona", "chat")

# All paths lead to END
workflow.add_edge("who_are_you", "__end__")
workflow.add_edge("chat", "__end__")
workflow.add_edge("task", "__end__")

# Compile the graph
agent_app = workflow.compile()


# API Endpoint
@app.post("/chat", tags=["Agents"])
async def chat(req: ChatRequest, session_id: str | None = Header(None)):
    """
    Multi-company intelligent agent with proper flow.
    """
    if not session_id:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
    
    # Initialize session if new
    if session_id not in session_memory:
        session_memory[session_id] = {
            "topic": "",
            "intent": "",
            "persona": {},
            "research_text": "",
            "history": []
        }
    
    session_data = session_memory[session_id]
    
    # Prepare state for LangGraph
    initial_state = {
        "message": req.message,
        "topic": "",
        "intent": "",
        "research_text": session_data.get("research_text", ""),
        "persona": session_data.get("persona", {}),
        "history": session_data.get("history", []),
        "generated_content": ""
    }
    
    # Run the agent workflow
    try:
        final_state = agent_app.invoke(initial_state)
        
        # Extract results
        ai_reply = final_state.get("generated_content", "I couldn't generate a response.")
        
        # Update session memory
        session_data["topic"] = final_state.get("topic", "")
        session_data["intent"] = final_state.get("intent", "")
        session_data["research_text"] = final_state.get("research_text", "")
        session_data["persona"] = final_state.get("persona", {})
        session_data["history"].append({"user": req.message, "ai": ai_reply})
        
        # Keep only last 20 exchanges
        if len(session_data["history"]) > 20:
            session_data["history"] = session_data["history"][-20:]
        
        return {
            "session_id": session_id,
            "history": session_data.get("history", [])
        }
        
    except Exception as e:
        print(f"Agent workflow error: {e}")
        return {
            "session_id": session_id,
            "response": "Sorry, I encountered an error processing your request.",
            "error": str(e)
        }
