"""
main.py
-------
Streamlit UI with:
- Experiment controls (parser/chunk/embedding/rerank/prompt/k)
- Namespace picker + Namespaces Manager (with history JSON)
- Direct ingest & query
- Agent command runner
- Evaluation runner (calls datasets/manual_eval.py)
- Sweep runner (grid search over configurations)
"""
import streamlit as st
import tempfile, os, subprocess
from datetime import datetime

from parsers import get_parser
from ingest import ingest_text
from generate import generate_answer
from agent import build_agent
from namespace_store import add_namespace, list_namespaces, delete_namespace
from namespace_admin import purge_namespace

def langsmith_link():
    proj = os.getenv("LANGCHAIN_PROJECT", "default")
    url = "https://smith.langchain.com/"
    return f"[Open LangSmith (project: {proj})]({url})"

st.set_page_config(page_title="AI Resume Intelligence", layout="wide")
st.title("🤖 AI Resume Intelligence System")

# ---------- Sidebar: Experiment Controls ----------
st.sidebar.header("🔧 Experiment Controls")
parser_backend = st.sidebar.selectbox("Parser", ["baseline", "docling"])
chunking = st.sidebar.selectbox("Chunking", ["recursive", "token"])
embedding_model = st.sidebar.selectbox("Embedding", ["text-embedding-3-small", "text-embedding-3-large"])
rerank = st.sidebar.selectbox("Reranker", ["none", "llm", "bge", "cohere"])
prompt_variant = st.sidebar.selectbox("Prompt", ["baseline", "strict"])
k = st.sidebar.slider("Top-K", 3, 10, 5)

# ---------- Namespace Picker (with history) ----------
st.sidebar.subheader("📦 Namespace")
ns_history = list_namespaces()
suggested = f"{parser_backend}_{chunking}_{embedding_model}".replace(".", "").replace("-", "")
namespace_choice = st.sidebar.selectbox("Select a saved namespace (or type below)", ["(custom)"] + ns_history, index=0)
custom_ns = st.sidebar.text_input("Custom namespace", value=suggested if namespace_choice=="(custom)" else namespace_choice)
active_ns = custom_ns if namespace_choice == "(custom)" else namespace_choice
st.sidebar.caption("Tip: Keep a stable scheme like parser_chunk_embed")

# ---------- Namespaces Manager ----------
with st.sidebar.expander("🗂️ Namespaces Manager"):
    st.write("Switch or delete stored namespaces.")
    ns_list = list_namespaces()
    if ns_list:
        sel_del = st.selectbox("Select namespace to delete", ns_list, key="del_ns")
        if st.button("Delete selected namespace from Pinecone"):
            purge_namespace(sel_del)
            delete_namespace(sel_del)
            st.success(f"Deleted namespace '{sel_del}' from Pinecone and history.")
    else:
        st.info("No namespaces saved yet. Ingest something first.")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Direct RAG (UI)", "Agent & Evaluation", "Sweep Runner"])

# -------- Tab 1: Direct RAG -------- #
with tab1:
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        parser = get_parser(parser_backend)
        text = parser.parse(uploaded_file.read(), file_ext="." + uploaded_file.name.split(".")[-1])
        st.success("✅ Parsed resume")
        st.text_area("Parsed Preview", text[:1500], height=200)

        if st.button("Ingest into Vector Store", type="primary"):
            count = ingest_text(
                text,
                chunking=chunking,
                embedding_model=embedding_model,
                metadata={"parser": parser_backend, "chunking": chunking, "embedding": embedding_model},
                namespace=active_ns,
            )
            add_namespace(active_ns)  # remember
            st.success(f"✅ Ingested {count} chunks into namespace '{active_ns}'. {langsmith_link()}")
            st.toast(f"Ingested into '{active_ns}'. Open LangSmith to view traces.", icon="✅")

    st.markdown("---")
    q = st.text_input("Ask a question about your ingested resumes")
    if st.button("Search", key="search_btn"):
        if not active_ns:
            st.error("Please set a namespace (sidebar).")
        else:
            with st.spinner("Generating..."):
                ans, docs = generate_answer(
                    q,
                    embedding_model=embedding_model,
                    rerank=rerank,
                    prompt_variant=prompt_variant,
                    k=k,
                    namespace=active_ns,
                )
            st.subheader("Answer")
            st.write(ans)
            with st.expander("Show retrieved chunks (debug)"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**#{i}** — {d.get('metadata',{})}")
                    st.write(d["content"][:800] + ("..." if len(d["content"]) > 800 else ""))

# -------- Tab 2: Agent & Evaluation -------- #
with tab2:
    st.subheader("🤖 Agent Command")
    agent_cmd = st.text_area("Type a natural command (e.g., 'Parse and ingest /path/resume.pdf with docling')", height=100)
    if st.button("Run Agent"):
        agent = build_agent()
        out = agent.invoke({"input": agent_cmd})
        st.subheader("Agent Output")
        st.write(out.get("output") or out)

    st.markdown("---")
    st.subheader("🧪 Run LangSmith Evaluation (single resume)")
    eval_resume_file = st.file_uploader("Select resume for evaluation", type=["pdf", "docx"], key="eval_resume")
    gold_yaml_file = st.file_uploader("Upload gold YAML", type=["yaml", "yml"], key="gold_yaml")

    if st.button("Run LangSmith Evaluation (current settings)"):
        if not eval_resume_file or not gold_yaml_file:
            st.error("Please upload BOTH a resume and a gold YAML.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                resume_path = os.path.join(tmpdir, eval_resume_file.name)
                with open(resume_path, "wb") as f:
                    f.write(eval_resume_file.read())
                gold_path = os.path.join(tmpdir, gold_yaml_file.name)
                with open(gold_path, "wb") as f:
                    f.write(gold_yaml_file.read())

                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                exp_label = f"ui-{parser_backend}-{chunking}-{embedding_model}-{rerank}-{prompt_variant}-{ts}"
                cmd = [
                    "python", "datasets/manual_eval.py",
                    "--resume", resume_path,
                    "--gold", gold_path,
                    "--namespace", active_ns,
                    "--parser_backend", parser_backend,
                    "--chunking", chunking,
                    "--embedding_model", embedding_model,
                    "--rerank", rerank,
                    "--prompt_variant", prompt_variant,
                    "--k", str(k),
                    "--exp_label", exp_label,
                ]
                st.code(" ".join(cmd), language="bash")
                with st.spinner("Running evaluation…"):
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                st.subheader("Eval Output (stdout)")
                st.text(proc.stdout or "(no stdout)")
                if proc.stderr:
                    st.subheader("Warnings/Errors (stderr)")
                    st.text(proc.stderr)
                if proc.returncode == 0:
                    st.success(f"✅ Submitted. {langsmith_link()}")
                    st.toast("LangSmith experiment submitted.", icon="✅")
                    add_namespace(active_ns)
                else:
                    st.error(f"Eval script exited with code {proc.returncode}")

# -------- Tab 3: Sweep Runner -------- #
with tab3:
    st.write("Run multiple evaluation variants in one go (grid sweep).")
    st.info("Tip: Ingest once per parser/chunking/embedding namespace before running sweeps, or let the eval script ingest each run.")

    parsers_sel   = st.multiselect("Parsers", ["baseline","docling"], default=[parser_backend])
    chunk_sel     = st.multiselect("Chunking", ["recursive","token"], default=[chunking])
    embed_sel     = st.multiselect("Embeddings", ["text-embedding-3-small","text-embedding-3-large"], default=[embedding_model])
    rerank_sel    = st.multiselect("Rerankers", ["none","llm","bge","cohere"], default=[rerank])
    prompt_sel    = st.multiselect("Prompts", ["baseline","strict"], default=[prompt_variant])
    topk          = st.slider("Top-K", 3, 10, k, key="sweep_k")

    sweep_ns_prefix = st.text_input("Namespace prefix (sweep)", value=active_ns or "sweep")
    eval_resume_file_s = st.file_uploader("Resume for sweep (PDF/DOCX)", type=["pdf","docx"], key="sweep_resume")
    gold_yaml_file_s   = st.file_uploader("Gold YAML for sweep", type=["yaml","yml"], key="sweep_gold")

    if st.button("Run Sweep"):
        if not eval_resume_file_s or not gold_yaml_file_s:
            st.error("Please upload BOTH a resume and a gold YAML for the sweep.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                resume_path = os.path.join(tmpdir, eval_resume_file_s.name)
                with open(resume_path, "wb") as f:
                    f.write(eval_resume_file_s.read())
                gold_path = os.path.join(tmpdir, gold_yaml_file_s.name)
                with open(gold_path, "wb") as f:
                    f.write(gold_yaml_file_s.read())

                total = 0
                for p in parsers_sel:
                    for c in chunk_sel:
                        for e in embed_sel:
                            for r in rerank_sel:
                                for pv in prompt_sel:
                                    total += 1
                st.write(f"Planned runs: **{total}**")

                ran = 0
                for p in parsers_sel:
                    for c in chunk_sel:
                        for e in embed_sel:
                            for r in rerank_sel:
                                for pv in prompt_sel:
                                    ran += 1
                                    ns = f"{sweep_ns_prefix}_{p}_{c}_{e}".replace(".","").replace("-","")
                                    exp_label = f"sweep-{p}-{c}-{e}-{r}-{pv}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                                    cmd = [
                                        "python","datasets/manual_eval.py",
                                        "--resume", resume_path,
                                        "--gold", gold_path,
                                        "--namespace", ns,
                                        "--parser_backend", p,
                                        "--chunking", c,
                                        "--embedding_model", e,
                                        "--rerank", r,
                                        "--prompt_variant", pv,
                                        "--k", str(topk),
                                        "--exp_label", exp_label
                                    ]
                                    st.write(f"({ran}/{total}) Running: `{exp_label}` → ns=`{ns}`")
                                    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                                    if proc.returncode == 0:
                                        st.success(f"✅ {exp_label} submitted. {langsmith_link()}")
                                        add_namespace(ns)
                                    else:
                                        st.error(f"❌ {exp_label} failed (code {proc.returncode})")
                                        if proc.stderr:
                                            with st.expander(f"stderr: {exp_label}"):
                                                st.text(proc.stderr)