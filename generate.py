"""
generate.py
-----------
Final question-answering step: retrieve -> prompt -> LLM.
Exposes knobs for embedding model, reranker, prompt variant, top-k, and namespace.
"""
from typing import Tuple, List, Dict, Any, Optional
from utils.config import llm
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from retrieve import retrieve_docs
from rag_strategies import PROMPTS

def generate_answer(query: str, *,
                    embedding_model: str = "text-embedding-3-small",
                    rerank: str = "none",
                    prompt_variant: str = "baseline",
                    k: int = 5,
                    namespace: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    emb = OpenAIEmbeddings(model=embedding_model)
    docs = retrieve_docs(query, k=k, embedding=emb, rerank=rerank, namespace=namespace)
    context = "\n\n".join(d["content"] for d in docs)
    tmpl = PROMPTS[prompt_variant]
    prompt = ChatPromptTemplate.from_template(tmpl)
    msgs = prompt.format_messages(context=context, query=query)
    resp = llm.invoke(msgs)
    return resp.content, docs