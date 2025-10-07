"""
rerankers.py
------------
Optional stronger rerankers:
- BGE CrossEncoder (local, open-source)
- Cohere Rerank (hosted API)

Expose a common .rerank(query, docs, top_k) -> docs interface.
Each doc is a dict: {"content": str, "metadata": {...}}
"""
from __future__ import annotations
from typing import List, Dict, Any, Literal

# ----- BGE Reranker -----
from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str | None = None):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        pairs = [(query, d["content"]) for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["_rerank_score"] = float(s)
        return sorted(docs, key=lambda x: x["_rerank_score"], reverse=True)[:top_k]

# ----- Cohere Rerank -----
import os, cohere

class CohereReranker:
    def __init__(self, model: str = "rerank-english-v3.0", api_key: str | None = None):
        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY not set for Cohere reranker.")
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        items = [d["content"] for d in docs]
        res = self.client.rerank(model=self.model, query=query, documents=items, top_n=top_k)
        ranked = []
        for r in res.results:
            d = docs[r.index].copy()
            d["_rerank_score"] = float(r.relevance_score)
            ranked.append(d)
        return ranked

def get_reranker(name: Literal["none","llm","bge","cohere"], **kwargs):
    if name == "bge":
        return BGEReranker(**kwargs)
    if name == "cohere":
        return CohereReranker(**kwargs)
    return None