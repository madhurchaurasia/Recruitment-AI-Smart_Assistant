"""
rag_strategies.py
-----------------
Pluggable strategies for chunking and prompting.
"""
from typing import List, Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.schema import Document

# ---- Chunking ----
def chunk_text(text: str, method: Literal["recursive","token"]="recursive", **kwargs) -> List[Document]:
    """
    Split raw text into chunks for RAG.
    - recursive: semantic character splitter (default)
    - token: token-based splitter
    """
    if method == "token":
        splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50, **kwargs)
        chunks = splitter.split_text(text)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, **kwargs)
        chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]

# ---- Prompt variants ----
PROMPTS = {
    "baseline": (
        "You are a recruitment assistant. Use ONLY the context to answer.\n"
        "If the answer is not explicitly stated in the context, reply with: \"I don't know.\""
        "\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}"
    ),
    "strict": (
        "Act as a precise retrieval QA assistant.\n"
        "Use only the provided context. If the answer isn't in the context, say: 'I don't know.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n"
        "Answer concisely with bullet points where possible."
    ),
}
