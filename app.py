"""
app.py
------
Streamlit UI for GovernanceRAG: Legal-to-Code Bridge (Light Theme).

Run with:
    streamlit run app.py
"""

from pathlib import Path
import streamlit as st

# ── 1. PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="GovernanceRAG: Legal-to-Code Bridge",
    page_icon="◈",
    layout="wide",
)

# ── 2. LIGHT MODE CUSTOM CSS ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* Main light theme background and text */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #ffffff !important;
    color: #1a1d20 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.main .block-container { max-width: 860px; padding: 3rem 2rem; margin: 0 auto; }

/* Header */
.rai-header { border-bottom: 1px solid #e2e8f0; padding-bottom: 2rem; margin-bottom: 2.5rem; }
.rai-title { font-family: 'DM Serif Display', serif; font-size: 2.4rem; font-weight: 400; color: #0f172a; letter-spacing: -0.02em; line-height: 1.1; margin: 0; }
.rai-title em { font-style: italic; color: #854d0e; }
.rai-subtitle { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #64748b; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.6rem; }

/* Stats Bar */
.stats-bar { display: flex; gap: 2rem; margin-bottom: 2rem; padding: 1rem 1.2rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; }
.stat-item { display: flex; flex-direction: column; gap: 2px; }
.stat-value { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #854d0e; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }

/* Inputs & Textarea */
.stTextArea textarea { background: #f8fafc !important; border: 1px solid #cbd5e1 !important; border-radius: 6px !important; color: #0f172a !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; padding: 1rem !important; resize: none !important; }
.stTextArea textarea:focus { border-color: #854d0e !important; box-shadow: none !important; }
.stTextArea textarea::placeholder { color: #94a3b8 !important; }

/* Buttons */
.stButton button { background: #854d0e !important; color: #ffffff !important; border: none !important; border-radius: 4px !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; padding: 0.6rem 1.8rem !important; }
.stButton button:hover { background: #713f12 !important; color: #ffffff !important; }

/* Answer Output Box */
.answer-container { background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #854d0e; border-radius: 6px; padding: 1.5rem 1.8rem; margin: 1.5rem 0; font-size: 0.95rem; line-height: 1.75; color: #1e293b; }

/* Sources Cards */
.sources-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #64748b; letter-spacing: 0.12em; text-transform: uppercase; margin: 1.5rem 0 0.8rem; }
.source-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; align-items: flex-start; gap: 0.8rem; }
.source-number { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #854d0e; min-width: 1.5rem; line-height: 1.3; }
.source-info { flex: 1; }
.source-title { font-size: 0.85rem; color: #0f172a; font-weight: 500; line-height: 1.3; }
.source-meta { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #64748b; margin-top: 3px; }
.score-pill { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #475569; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 3px; padding: 2px 6px; white-space: nowrap; }
.examples-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #64748b; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.8rem; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 2rem 0; }
.stSpinner > div { border-top-color: #854d0e !important; }
</style>
""", unsafe_allow_html=True)

# ── 3. LOAD MODULES & MODELS ─────────────────────────────────
from src.ingest import run_ingestion
from src.retriever import load_retriever, retrieve, rerank
from src.generator import generate_streaming

@st.cache_resource(show_spinner=False)
def get_retriever_cached():
    from src.config import CHROMA_DIR, COLLECTION_NAME
    import chromadb

    chroma_path = Path(CHROMA_DIR)
    needs_ingestion = False

    if not chroma_path.exists():
        needs_ingestion = True
    else:
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(COLLECTION_NAME)
            if collection.count() == 0:
                needs_ingestion = True
        except Exception:
            needs_ingestion = True

    return needs_ingestion

@st.cache_resource(show_spinner=False)
def load_ml_models():
    # Only called once, returns the actual models
    embedder, reranker, collection, bm25_data = load_retriever()
    return embedder, reranker, collection, bm25_data

# Check if we need ingestion (fast)
needs_ingestion = get_retriever_cached()

# We only show the loading UI if the models aren't loaded in session state yet
if "models_loaded" not in st.session_state:
    loading_container = st.empty()
    with loading_container.container():
        if needs_ingestion:
            from src.config import PAPERS_DIR
            from src.ingest import extract_documents, chunk_documents, embed_and_store

            with st.status("No ingestion was found, running ingestion Pipeline...", expanded=True) as status:
                st.write("&nbsp;&nbsp;&nbsp;&nbsp;⏳ Extracting text from PDFs...")
                docs = extract_documents(PAPERS_DIR)
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ Extracted {len(docs)} documents.")
                
                st.write("&nbsp;&nbsp;&nbsp;&nbsp;⏳ Chunking documents...")
                chunks = chunk_documents(docs)
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ Created {len(chunks)} chunks.")
                
                st.write("&nbsp;&nbsp;&nbsp;&nbsp;⏳ Embedding into ChromaDB...")
                embed_and_store(chunks)
                st.write("&nbsp;&nbsp;&nbsp;&nbsp;✅ Embeddings saved.")
                status.update(label="Ingestion complete", state="complete", expanded=False)

        with st.status("Starting ML Engine...", expanded=True) as ml_status:
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;⏳ Loading Embedding Model...")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;⏳ Loading Cross-Encoder Reranker...")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;⏳ Connecting to ChromaDB...")
            embedder, reranker, collection, bm25_data = load_ml_models()
            ml_status.update(label="ML Engine Ready!", state="complete", expanded=False)
            
    # Clear the UI completely once everything is loaded
    loading_container.empty()
    st.session_state.models_loaded = True
else:
    # Models are already in cache, just retrieve them instantly without UI
    embedder, reranker, collection, bm25_data = load_ml_models()


# ── 4. HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="rai-header">
    <h1 class="rai-title">AI Compliance<br><em>Copilot</em></h1>
    <p class="rai-subtitle">Legal-to-Engineering RAG Bridge | EU AI Act & NIST</p>
</div>
""", unsafe_allow_html=True)

# ── 5. STATS BAR ─────────────────────────────────────────────
total_chunks = collection.count()
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-value">{total_chunks:,}</span>
        <span class="stat-label">Chunks embedded</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">2</span>
        <span class="stat-label">Stage retrieval</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 6. EXAMPLE QUESTIONS ─────────────────────────────────────
st.markdown('<p class="examples-header">Example questions</p>', unsafe_allow_html=True)

examples = [
    "What are the mandatory requirements for high-risk AI systems?",
    "How do we mitigate bias to comply with Article 10 of the EU AI Act?",
    "What explainability metrics should we use for high-risk systems?",
]

if "selected_example" not in st.session_state:
    st.session_state.selected_example = ""

cols = st.columns(len(examples))
for i, (col, example) in enumerate(zip(cols, examples)):
    with col:
        label = example[:35] + "..." if len(example) > 35 else example
        if st.button(label, key=f"ex_{i}"):
            st.session_state.selected_example = example

# ── 7. QUESTION INPUT ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

question = st.text_area(
    label="Your question",
    value=st.session_state.selected_example,
    placeholder="Ask anything about AI compliance, fairness algorithms, EU AI Act...",
    height=100,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    search = st.button("Ask", use_container_width=True)

# ── 8. QUERY HANDLING ────────────────────────────────────────
if search and question.strip():
    st.session_state.selected_example = ""

    with st.spinner("Retrieving legal & technical sources..."):
        retrieved = retrieve(question, embedder, collection, bm25_data=bm25_data)
        reranked = rerank(question, retrieved, reranker)

    if not reranked or reranked[0]["rerank_score"] < 0.5:
        st.markdown("""
        <div class="answer-container">
        This question appears to be outside the scope of the compliance corpus.
        Try asking about the EU AI Act, fairness metrics, or bias mitigation.
        </div>
        """, unsafe_allow_html=True)
    else:
        answer_placeholder = st.empty()
        
        # Get the generator iterator
        generator = generate_streaming(question, reranked)
        
        # Use st.spinner specifically for the wait time before the first word
        with st.spinner("Generating answer..."):
            try:
                first_token = next(generator)
            except StopIteration:
                first_token = ""
                
        full_answer = first_token
        answer_placeholder.markdown(f'<div class="answer-container">\n\n{full_answer}▌\n\n</div>', unsafe_allow_html=True)

        for token in generator:
            full_answer += token
            answer_placeholder.markdown(f'<div class="answer-container">\n\n{full_answer}▌\n\n</div>', unsafe_allow_html=True)

        answer_placeholder.markdown(f'<div class="answer-container">\n\n{full_answer}\n\n</div>', unsafe_allow_html=True)

        st.markdown('<p class="sources-header">Sources retrieved</p>', unsafe_allow_html=True)

        for i, chunk in enumerate(reranked, 1):
            st.markdown(f"""
            <div class="source-card">
                <span class="source-number">{i}</span>
                <div class="source-info">
                    <div class="source-title">{chunk['title']}</div>
                    <div class="source-meta">
                        Similarity: {chunk['similarity']} | Chunk {chunk.get('chunk_idx', 'N/A')}
                    </div>
                </div>
                <span class="score-pill">score {chunk['rerank_score']}</span>
            </div>
            """, unsafe_allow_html=True)

elif search and not question.strip():
    st.warning("Please enter a question.")

# ── 9. FOOTER ────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #64748b; text-align: center; letter-spacing: 0.08em;">
LEGAL-TO-CODE RAG PIPELINE | STREAMLIT | BUILT FOR PORTFOLIO
</p>
""", unsafe_allow_html=True)