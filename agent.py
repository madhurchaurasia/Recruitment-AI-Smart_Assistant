"""
agent.py
--------
OpenAI tool-calling agent that can orchestrate parsing, ingestion, and QA.
"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from utils.config import llm
from tools import parse_resume_tool, ingest_text_tool, generate_tool

TOOLS = [parse_resume_tool, ingest_text_tool, generate_tool]

SYSTEM = (
    "You are an AI recruiter operations agent. "
    "Choose tools to parse resumes, ingest text into the vector store, "
    "and answer queries via retrieval-augmented generation. "
    "Prefer: parse -> ingest for new resumes; generate_answer for queries. "
    "Explain briefly what you did when appropriate."
)

def build_agent() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_openai_tools_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

def run_agent(user_input: str) -> Dict[str, Any]:
    """
    Execute the agent for a single input.
    Examples:
      - 'Parse and ingest /path/to/resume.pdf with docling'
      - 'Find React developers with 5 years'
    """
    executor = build_agent()
    return executor.invoke({"input": user_input})