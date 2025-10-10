"""
ingest.py
---------
Split -> embed -> upsert chunks into Pinecone.
Supports variant isolation via `namespace` and optional per-chunk metadata.
"""
import os
from typing import Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from rag_strategies import chunk_text

def ingest_text(text: str, *, 
                chunking: str = "recursive",
                embedding_model: str = "text-embedding-3-small",
                metadata: Optional[Dict[str, Any]] = None,
                namespace: Optional[str] = None) -> int:
    docs = chunk_text(text, method=chunking)
    if metadata:
        for d in docs:
            d.metadata = {**(d.metadata or {}), **metadata}

    emb = OpenAIEmbeddings(model=embedding_model)
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=emb,
        index_name=os.getenv("PINECONE_INDEX_NAME", "resume-rag"),
        namespace=namespace,
    )
    return len(docs)
