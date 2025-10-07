"""
datasets/manual_eval.py
-----------------------
Manual evaluation runner for a single resume:
- Ingests resume into a specified namespace (using selected parser/chunking/embedding)
- Builds/updates a LangSmith dataset from a YAML of gold answers
- Runs an evaluation experiment (faithfulness, answer relevancy, QA-with-reference, conciseness)
- All results show up in LangSmith > Experiments

Usage example:
python datasets/manual_eval.py   --resume ./data/resumes/jane_doe.pdf   --gold   ./datasets/sample_gold.yaml   --namespace docling_recursive_small   --parser_backend docling   --chunking recursive   --embedding_model text-embedding-3-small   --rerank bge   --prompt_variant strict   --k 5   --exp_label "docling-recursive-small-bge-strict"
"""
import os, argparse
from pathlib import Path
from typing import Dict, Any

import yaml
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

from parsers import get_parser
from ingest import ingest_text
from generate import generate_answer

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dataset(client: Client, name: str, qa: list[dict]) -> str:
    ds = client.get_dataset(name=name)
    if ds is None:
        ds = client.create_dataset(name=name, description="Manual resume QA dataset")
    for ex in qa:
        client.create_example(inputs={"question": ex["question"]},
                              outputs={"answer": ex.get("answer", "")},
                              dataset_id=ds.id)
    return ds.id

def ingest_resume(file_path: Path, parser_backend: str, chunking: str, embedding_model: str, namespace: str):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    parser = get_parser(parser_backend)
    text = parser.parse(file_bytes, file_ext=file_path.suffix)
    n = ingest_text(
        text,
        chunking=chunking,
        embedding_model=embedding_model,
        metadata={"parser": parser_backend, "chunking": chunking, "embedding": embedding_model},
        namespace=namespace,
    )
    return n

@traceable(run_type="chain", name="rag-adapter")
def rag_chain_adapter(inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    q = inputs["question"]
    ans, docs = generate_answer(
        q,
        embedding_model=config.get("embedding_model", "text-embedding-3-small"),
        rerank=config.get("rerank", "none"),
        prompt_variant=config.get("prompt_variant", "baseline"),
        k=int(config.get("k", 5)),
        namespace=config.get("namespace"),
    )
    ctx = "\n\n".join(d["content"] if isinstance(d, dict) else d.page_content for d in docs)
    return {"answer": ans, "context": ctx}

def run_eval(dataset_name: str, experiment_name: str, config: Dict[str, Any]):
    client = Client()
    dataset = client.get_dataset(dataset_name)
    results = evaluate(
        rag_chain_adapter,
        data=dataset,
        evaluators=[
            "qa_with_reference",
            "faithfulness",
            "answer_relevancy",
            "conciseness",
        ],
        experiment_prefix=experiment_name,
        metadata=config,
    )
    return results

def main():
    ap = argparse.ArgumentParser(description="Manual LangSmith eval for a single resume.")
    ap.add_argument("--resume", required=True, help="Path to resume (PDF/DOCX)")
    ap.add_argument("--gold", required=True, help="Path to YAML of QA gold answers")
    ap.add_argument("--namespace", required=True, help="Pinecone namespace to ingest/query (e.g., docling_recursive)")
    ap.add_argument("--parser_backend", default="baseline", choices=["baseline","docling"])
    ap.add_argument("--chunking", default="recursive", choices=["recursive","token"])
    ap.add_argument("--embedding_model", default="text-embedding-3-small",
                    choices=["text-embedding-3-small","text-embedding-3-large"])
    ap.add_argument("--rerank", default="none", choices=["none","llm","bge","cohere"])
    ap.add_argument("--prompt_variant", default="baseline", choices=["baseline","strict"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--exp_label", default=None, help="Optional experiment label suffix")
    args = ap.parse_args()

    for key in ["LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"]:
        if not os.getenv(key):
            print(f"⚠️  Warning: {key} is not set. Tracing/organization in LangSmith may be limited.")

    resume_path = Path(args.resume).expanduser().resolve()
    gold_path = Path(args.gold).expanduser().resolve()
    assert resume_path.exists(), f"Resume not found: {resume_path}"
    assert gold_path.exists(), f"Gold file not found: {gold_path}"

    print(f"📥 Ingesting {resume_path.name} with {args.parser_backend}/{args.chunking}/{args.embedding_model} → ns={args.namespace}")
    n_chunks = ingest_resume(resume_path, args.parser_backend, args.chunking, args.embedding_model, args.namespace)
    print(f"✅ Ingested {n_chunks} chunks")

    gold = load_yaml(gold_path)
    qa_list = gold.get("qa", [])
    ds_name = f"Resume-QA::{resume_path.stem}"
    client = Client()
    ds_id = ensure_dataset(client, ds_name, qa_list)
    print(f"📚 Dataset ready: {ds_name} ({ds_id}) with {len(qa_list)} examples")

    label = args.exp_label or f"{args.parser_backend}-{args.chunking}-{args.embedding_model}-{args.rerank}-{args.prompt_variant}"
    exp_name = f"manual::{resume_path.stem}::{label}"
    cfg = {
        "namespace": args.namespace,
        "parser_backend": args.parser_backend,
        "chunking": args.chunking,
        "embedding_model": args.embedding_model,
        "rerank": args.rerank,
        "prompt_variant": args.prompt_variant,
        "k": args.k,
    }
    print(f"🧪 Running experiment: {exp_name}")
    run_eval(ds_name, exp_name, cfg)
    print("🎯 Done. Open LangSmith › Experiments to compare scores.")

if __name__ == "__main__":
    main()