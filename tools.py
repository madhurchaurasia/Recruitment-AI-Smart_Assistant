"""
tools.py
--------
LangChain Tools wrapping our main capabilities so an Agent can orchestrate them.
"""
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from parsers import get_parser
from ingest import ingest_text as _ingest_text
from generate import generate_answer as _generate_answer
from langchain.schema import Document

# ---- Schemas ----
class ParseInput(BaseModel):
    file_path: str = Field(..., description="Absolute path to the resume file (.pdf/.docx)")
    backend: Literal["baseline","docling"] = Field("baseline", description="Parser backend")

class IngestInput(BaseModel):
    text: str
    chunking: Literal["recursive","token"] = "recursive"
    embedding_model: Literal["text-embedding-3-small","text-embedding-3-large"] = "text-embedding-3-small"
    namespace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class GenerateInput(BaseModel):
    query: str
    embedding_model: Literal["text-embedding-3-small","text-embedding-3-large"] = "text-embedding-3-small"
    rerank: Literal["none","llm","bge","cohere"] = "none"
    prompt_variant: Literal["baseline","strict"] = "baseline"
    k: int = 5
    namespace: Optional[str] = None

# ---- Tools ----
@tool("parse_resume", args_schema=ParseInput, return_direct=False)
def parse_resume_tool(file_path: str, backend: str = "baseline") -> str:
    """Parse a resume from disk and return plain text/markdown."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    parser = get_parser(backend)
    return parser.parse(file_bytes, file_ext=("." + file_path.split(".")[-1]))

@tool("ingest_text", args_schema=IngestInput, return_direct=False)
def ingest_text_tool(text: str, chunking: str = "recursive",
                     embedding_model: str = "text-embedding-3-small",
                     namespace: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
    """Chunk and upsert resume text into Pinecone, returning a status string."""
    n = _ingest_text(text, chunking=chunking, embedding_model=embedding_model, metadata=metadata, namespace=namespace)
    return f"Ingested {n} chunks into namespace {namespace or '(default)'}."

@tool("generate_answer", args_schema=GenerateInput, return_direct=False)
def generate_tool(query: str,
                  embedding_model: str = "text-embedding-3-small",
                  rerank: str = "none",
                  prompt_variant: str = "baseline",
                  k: int = 5,
                  namespace: Optional[str] = None) -> str:
    """Run retrieval-augmented QA for the given query and return the LLM answer."""
    answer, _ = _generate_answer(query, embedding_model=embedding_model, rerank=rerank,
                                 prompt_variant=prompt_variant, k=k, namespace=namespace)
    return answer
