# Resume RAG + Agent + LangSmith Experiments

A production-grade, **A/B testable** RAG system for resumes with:
- **Parsers**: baseline (pdfplumber/docx2txt + OCR) **vs** **Docling** (layout-aware)
- **Chunking**: recursive vs token
- **Embeddings**: OpenAI `text-embedding-3-small` / `-large`
- **Vector DB**: Pinecone (namespaces per variant)
- **Rerankers**: none / LLM compression / **BGE** / **Cohere Rerank**
- **LLM**: GPT‑4o‑mini
- **LangSmith**: tracing, datasets, **evaluation scorecards**
- **Streamlit UI**: ingest, query, agent, eval runner, and **sweep runner**
- **Namespaces Manager**: list/delete/switch + cached dropdown (`./.ns_history.json`)

> Built to be **readable by AI coding copilots** — modules are small, typed, and richly commented so tools like VS Code + GitHub Copilot/Codex can quickly infer intent and help you extend.

---

## 0) Prerequisites (system packages)

- **Tesseract OCR** (for baseline OCR fallback)
  - macOS (brew): `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`
- **poppler** (for `pdf2image`)
  - macOS: `brew install poppler`
  - Ubuntu: `sudo apt-get install poppler-utils`

Create the **Pinecone index** named in `.env` (default: `resume_rag`):
- Use Pinecone Serverless index with a suitable dimension (OpenAI embeddings will set automatically on first upsert via the new client),
- Or pre-create in the console.

---

## 1) Setup

```bash
# Clone or unzip this folder
cd rag_project

# Python env (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Configure env
cp .env.example .env
# fill in: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, LANGSMITH_API_KEY (optional), etc.
```

> **LangSmith**: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT=Resume-RAG-Experiments` in `.env` for clean organization.

---

## 2) Run the UI

```bash
streamlit run main.py
```

Open http://localhost:8501

### What you can do
- **Upload resume** (PDF/DOCX), pick parser (`baseline` or `docling`), chunking, embedding, etc.
- **Ingest** into a **namespace** (auto-suggested; editable). Namespace history is stored in `./.ns_history.json` and shown in a dropdown.
- **Ask questions** — the app retrieves from the selected namespace with your reranker and prompt settings.
- **Agent** — natural-language command interface that can call tools to parse, ingest, and answer.
- **Evaluation** — upload a resume and **gold YAML**, then run a full **LangSmith** evaluation (faithfulness, answer relevancy, etc.).
- **Sweep Runner** — grid-evaluate multiple variants; each run is labeled and isolated by namespace.

> Every successful ingestion / eval shows a success message with a **LangSmith quick link**.

---

## 3) Manual Evaluation (CLI)

Prepare a gold YAML (see `datasets/sample_gold.yaml`) **per resume**.
Then run:

```bash
python datasets/manual_eval.py   --resume ./path/to/resume.pdf   --gold   ./datasets/sample_gold.yaml   --namespace docling_recursive_small   --parser_backend docling   --chunking recursive   --embedding_model text-embedding-3-small   --rerank bge   --prompt_variant strict   --k 5   --exp_label "docling-recursive-small-bge-strict"
```

Open **LangSmith › Experiments** to compare scorecards.

---

## 4) Project Structure

```
rag_project/
├── .env.example                  # Copy to .env and fill keys
├── README.md                     # You are here
├── requirements.txt
├── main.py                       # Streamlit app (UI, Agent, Eval, Sweeps)
│
├── parsers.py                    # Baseline vs Docling (pluggable)
├── rag_strategies.py             # Chunkers + prompt templates
├── ingest.py                     # Split → embed → upsert (Pinecone)
├── retrieve.py                   # Vector retrieve (+ optional rerankers) with namespace
├── generate.py                   # RAG QA: retrieve → prompt → LLM
├── rerankers.py                  # BGE & Cohere rerankers
│
├── tools.py                      # LangChain Tools for the Agent
├── agent.py                      # OpenAI tool-calling Agent
│
├── namespace_store.py            # Saves namespace history (JSON)
├── namespace_admin.py            # Purge namespace from Pinecone
│
└── datasets/
    ├── manual_eval.py            # Ingest + dataset + run evaluators (LangSmith)
    └── sample_gold.yaml          # Example gold QA for one resume
```

---

## 5) How it works (workflow overview)

1. **Parse**: You choose parser backend. Baseline = pdfplumber/docx2txt + OCR fallback; Docling = layout-aware Markdown export.
2. **Chunk**: Choose recursive or token splitter; attach metadata (parser/chunk/embedding).
3. **Embed + Upsert**: OpenAI embeddings to a Pinecone namespace (one per variant).
4. **Retrieve**: Pull k matches from selected namespace; optional reranker: none/LLM/BGE/Cohere.
5. **Prompt + Generate**: Choose prompt variant; GPT‑4o‑mini answers using only retrieved context.
6. **Trace & Evaluate**: All runs are traced in **LangSmith**. The evaluation script builds a dataset and runs automatic evaluators to produce **scorecards**.

---

## 6) Notes for VS Code + Copilot/Codex

- Files use **type annotations and docstrings** to provide clear intent.
- Each module is loosely coupled; swap implementations without touching others.
- Start in `main.py` for UI wiring, and follow imports to see the flow.
- Key extension points:
  - Add **new prompts** in `rag_strategies.PROMPTS`
  - Add **new rerankers** in `rerankers.py`
  - Add **metadata enrichers** in `ingest.py` (e.g., extract candidate name/email)
  - Add **namespaces analytics** (Pinecone stats) in a new `ns_stats.py` and a Streamlit expander

---

## 7) Troubleshooting

- **Tesseract not found**: install system package; on Windows, ensure PATH includes Tesseract.
- **pdf2image errors**: install `poppler` and ensure `pdftoppm` is on PATH.
- **Pinecone index not found**: create the index (same name as `PINECONE_INDEX_NAME`) in the Pinecone console.
- **Cohere rerank**: requires `COHERE_API_KEY` set.
- **Docling**: some environments need additional system deps; see Docling docs if you hit errors.

---

## 8) License

This template is provided as-is for internal prototyping and evaluation.