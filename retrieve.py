"""
retrieve.py
-----------
Vector retrieval with optional rerankers and namespace isolation.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from utils.config import index, llm
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from rerankers import get_reranker

def _vector_retrieve(query: str, k: int, embedding, namespace: Optional[str]):
    base = PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=namespace,
    ).as_retriever(search_kwargs={"k": k})
    return base.get_relevant_documents(query)

def retrieve_docs(query: str, k: int = 5, *, embedding, rerank: str = "none", namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    # Oversample before reranking to improve headroom
    docs = _vector_retrieve(query, k=max(k, 10), embedding=embedding, namespace=namespace)
    docs = [{"content": d.page_content, "metadata": d.metadata or {}} for d in docs]

    if rerank in ("bge", "cohere"):
        rr = get_reranker(rerank)
        return rr.rerank(query, docs, top_k=k)

    if rerank == "llm":
        compressor = LLMChainExtractor.from_llm(llm)
        base = PineconeVectorStore(index=index, embedding=embedding, namespace=namespace).as_retriever(search_kwargs={"k": k})
        retr = ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)
        docs_lc = retr.get_relevant_documents(query)
        return [{"content": d.page_content, "metadata": d.metadata or {}} for d in docs_lc]

    return docs[:k]