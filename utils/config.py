"""
utils/config.py
----------------
Centralized configuration, clients, and models.

This file is heavily commented so AI coding assistants (e.g., Copilot/Codex)
understand the architecture and suggest correct completions.
"""
import os
from dotenv import load_dotenv
from langsmith import Client
from pinecone import Pinecone
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Required ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resume_rag")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Create a .env from .env.example")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set. Create a .env from .env.example")

# --- LangSmith (optional but recommended) ---
# Used for tracing, experiments, and evaluation scorecards
client = Client(api_key=LANGSMITH_API_KEY) if LANGSMITH_API_KEY else None

# --- Pinecone ---
# Assumes the index already exists. See README for creation instructions.
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- OpenAI LLM ---
# We keep temperature low for deterministic QA; change as needed.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)