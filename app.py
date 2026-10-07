import gradio as gr
from pathlib import Path
import glob

from src.ingest import run_ingestion
from src.retriever import load_retriever, retrieve, rerank
from src.generator import generate_streaming
from src.verifier import CitationVerifier
from src.config import ENABLE_CITATION_VERIFICATION


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


embedder, reranker, collection, bm25_data = None, None, None, None
verifier = None
total_chunks = 0
total_papers = 0


def init_models():
    global embedder, reranker, collection, bm25_data, verifier, total_chunks, total_papers
    needs_ingestion = get_retriever_cached()

    if needs_ingestion:
        print("No ingestion was found, running ingestion Pipeline...")
        from src.config import PAPERS_DIR
        from src.ingest import extract_documents, chunk_documents, embed_and_store

        print("Extracting text from PDFs...")
        docs = extract_documents(PAPERS_DIR)
        print("Chunking documents...")
        chunks = chunk_documents(docs)
        print("Embedding into ChromaDB...")
        embed_and_store(chunks)
        print("Ingestion complete.")

    if embedder is None:
        print("Loading ML Engine (Embedder, Reranker, BM25)...")
        embedder, reranker, collection, bm25_data = load_retriever()
        total_chunks = collection.count()
        from src.config import PAPERS_DIR
        total_papers = len(glob.glob(str(Path(PAPERS_DIR) / "*.pdf")))
        print("Retriever Ready!")

    if verifier is None and ENABLE_CITATION_VERIFICATION:
        print("Loading NLI Citation Verifier Engine...")
        verifier = CitationVerifier()
        print("NLI Verifier Ready!")


def process_query(question):
    if not question.strip():
        yield "<div class='answer-container'>Please enter a question.</div>"
        return

    yield "<div class='answer-container'><em>Retrieving legal & technical sources...</em></div>"

    retrieved = retrieve(question, embedder, collection, bm25_data=bm25_data)
    reranked = rerank(question, retrieved, reranker)

    if not reranked or reranked[0]["rerank_score"] < -1.5:
        yield """<div class="answer-container">
This question appears to be outside the scope of the compliance corpus.
Try asking about the EU AI Act, fairness metrics, or bias mitigation.
</div>"""
        return

    # Start streaming generator
    generator = generate_streaming(question, reranked)

    full_answer = ""
    for token in generator:
        full_answer += token
        yield f'<div class="answer-container">\n\n{full_answer}|\n\n</div>'

    # Citation Verification Pass (NLI Guardrails)
    audit = None
    if verifier:
        yield f'<div class="answer-container">\n\n{full_answer}\n\n</div>\n<div class="audit-banner audit-badge-warning"><span class="audit-badge-icon">⏳</span><div class="audit-badge-text"><em>Auditing citations with NLI guardrails against literature...</em></div></div>'
        audit = verifier.verify_answer(full_answer, reranked)

    # Build final output
    final_output = ""
    if audit:
        badge_icon = "🛡️" if audit["badge_type"] == "success" else ("⚠️" if audit["badge_type"] == "warning" else "⛔")
        score_label = f"Grounding: {int(audit['grounding_rate'] * 100)}%" if audit["total_citations"] > 0 else "Grounding: 0%"
        final_output += f"""<div class="audit-banner audit-badge-{audit['badge_type']}">
<span class="audit-badge-icon">{badge_icon}</span>
<div class="audit-badge-text">
<strong>NLI Citation Audit:</strong> {audit['badge']}
<span class="audit-score">{score_label}</span>
</div>
</div>\n"""

    final_output += f'<div class="answer-container">\n\n{full_answer}\n\n</div>'
    final_output += '\n<p class="sources-header">Sources retrieved & evaluated</p>\n\n'

    for i, chunk in enumerate(reranked, 1):
        domain = chunk.get("domain", "general").lower()
        domain_label = "LEGAL" if domain == "legal" else "TECHNICAL"
        domain_class = f"domain-{domain}"

        status_html = ""
        quote_html = ""
        if audit:
            src_info = audit["source_summary"].get(i, {})
            status = src_info.get("status", "unreferenced")
            conf = src_info.get("confidence", 0.0)
            best_q = src_info.get("best_quote", "")

            if status == "verified":
                status_html = f'<span class="status-pill status-verified">✓ Verified ({int(conf * 100)}%)</span>'
            elif status == "contradicted":
                status_html = f'<span class="status-pill status-contradicted">⛔ Contradicted</span>'
            elif status == "partial" or status == "unsupported":
                status_html = f'<span class="status-pill status-unsupported">⚠️ Weak Grounding</span>'
            else:
                status_html = '<span class="status-pill status-unreferenced">Not cited</span>'

            if best_q:
                quote_html = f"""<details class="quote-accordion">
<summary class="quote-toggle">Show supporting evidence</summary>
<div class="quote-box">"{best_q}"</div>
</details>"""

        final_output += f"""<div class="source-card">
<span class="source-number">{i}</span>
<div class="source-info">
<div class="source-title">{chunk['title']}</div>
<div class="source-meta">
Similarity: {chunk['similarity']} | Chunk {chunk.get('chunk_idx', 'N/A')}
</div>
{quote_html}
</div>
<div class="source-actions">
<span class="score-pill">score {chunk['rerank_score']}</span>
<span class="domain-pill {domain_class}">{domain_label}</span>
{status_html}
</div>
</div>
"""
    yield final_output


css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* Dynamic theme background and text */
.gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Header */
.rai-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 2rem; margin-bottom: 2.5rem; }
.rai-title { font-family: 'DM Serif Display', serif; font-size: 2.4rem; font-weight: 400; color: var(--body-text-color); letter-spacing: -0.02em; line-height: 1.1; margin: 0; }
.rai-title em { font-style: italic; color: #b45309; }
.rai-subtitle { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--body-text-color-subdued); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.6rem; }

/* Stats Bar */
.stats-bar { display: flex; gap: 2rem; margin-bottom: 2rem; padding: 1rem 1.2rem; background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 6px; }
.stat-item { display: flex; flex-direction: column; gap: 2px; }
.stat-value { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #b45309; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--body-text-color-subdued); text-transform: uppercase; letter-spacing: 0.1em; }

/* Answer Output Box */
.answer-container { background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-left: 4px solid #b45309; border-radius: 6px; padding: 1.5rem 1.8rem; margin: 1.5rem 0; font-size: 0.95rem; line-height: 1.75; color: var(--body-text-color); }

/* Audit Banner */
.audit-banner {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.8rem 1.2rem;
    border-radius: 6px;
    margin: 1rem 0;
    font-size: 0.88rem;
    border: 1px solid var(--border-color-primary);
}
.audit-badge-success {
    background: rgba(22, 163, 74, 0.08);
    border-color: rgba(22, 163, 74, 0.3);
    color: #16a34a;
}
.audit-badge-warning {
    background: rgba(217, 119, 6, 0.08);
    border-color: rgba(217, 119, 6, 0.3);
    color: #d97706;
}
.audit-badge-danger {
    background: rgba(220, 38, 38, 0.08);
    border-color: rgba(220, 38, 38, 0.3);
    color: #dc2626;
}
.audit-badge-text { flex: 1; color: var(--body-text-color); }
.audit-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--background-fill-primary);
    margin-left: 0.5rem;
    border: 1px solid var(--border-color-primary);
}

/* Domain and Status Pills */
.domain-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.05em;
    padding: 2px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    font-weight: 500;
}
.domain-legal {
    background: rgba(37, 99, 235, 0.12);
    color: #2563eb;
    border: 1px solid rgba(37, 99, 235, 0.3);
}
.domain-technical {
    background: rgba(124, 58, 237, 0.12);
    color: #7c3aed;
    border: 1px solid rgba(124, 58, 237, 0.3);
}

.status-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 500;
}
.status-verified {
    background: rgba(22, 163, 74, 0.12);
    color: #16a34a;
    border: 1px solid rgba(22, 163, 74, 0.3);
}
.status-unsupported {
    background: rgba(217, 119, 6, 0.12);
    color: #d97706;
    border: 1px solid rgba(217, 119, 6, 0.3);
}
.status-contradicted {
    background: rgba(220, 38, 38, 0.12);
    color: #dc2626;
    border: 1px solid rgba(220, 38, 38, 0.3);
}
.status-unreferenced {
    background: var(--background-fill-primary);
    color: var(--body-text-color-subdued);
    border: 1px solid var(--border-color-primary);
}

/* Expandable Evidence Quote */
.quote-accordion {
    margin-top: 0.6rem;
    border-top: 1px dashed var(--border-color-primary);
    padding-top: 0.4rem;
}
.quote-toggle {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #b45309;
    cursor: pointer;
    user-select: none;
}
.quote-box {
    margin-top: 0.4rem;
    padding: 0.5rem 0.8rem;
    background: var(--background-fill-primary);
    border-left: 2px solid #b45309;
    font-size: 0.82rem;
    font-style: italic;
    color: var(--body-text-color-subdued);
    border-radius: 0 4px 4px 0;
}

/* Sources Cards */
.sources-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--body-text-color-subdued); letter-spacing: 0.12em; text-transform: uppercase; margin: 1.5rem 0 0.8rem; }
.source-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    gap: 0.8rem !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
.source-number { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #b45309; min-width: 1.5rem; line-height: 1.3; }
.source-info { flex: 1 !important; min-width: 0 !important; }
.source-title { font-size: 0.85rem; color: var(--body-text-color); font-weight: 500; line-height: 1.3; word-break: break-word; }
.source-meta { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--body-text-color-subdued); margin-top: 3px; }
.source-actions {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
    gap: 4px !important;
    flex-shrink: 0 !important;
}
.score-pill { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--body-text-color-subdued); background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 3px; padding: 2px 6px; white-space: nowrap; }
.examples-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--body-text-color-subdued); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.8rem; }

/* Examples */
.example-btn {
    background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important; color: var(--body-text-color) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important; 
    text-transform: none !important; text-align: left !important; white-space: normal !important;
    padding: 1rem !important; border-radius: 6px !important;
    display: flex; height: 100%; align-items: flex-start;
}
.example-btn:hover { background: var(--background-fill-primary) !important; color: var(--body-text-color) !important; border-color: #b45309 !important; }

/* Custom Inputs & Buttons */
textarea { background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important; border-radius: 6px !important; color: var(--body-text-color) !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; padding: 1rem !important; resize: none !important; }
textarea:focus { border-color: #b45309 !important; box-shadow: none !important; }

#ask-button { background: #b45309 !important; color: #ffffff !important; border: none !important; border-radius: 4px !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; padding: 0.6rem 1.8rem !important; margin-top: 0.5rem; width: 100%; }
#ask-button:hover { background: #92400e !important; color: #ffffff !important; }

/* Hide default borders */
.block { border: none !important; box-shadow: none !important; background: transparent !important; }
"""

init_models()

with gr.Blocks() as demo:
    gr.HTML(f"""
    <div class="rai-header">
        <h1 class="rai-title">GovernanceRAG<br><em>AI Compliance</em></h1>
        <p class="rai-subtitle">Legal-to-Engineering RAG Bridge | EU AI Act & NIST</p>
    </div>
    
    <div class="stats-bar">
        <div class="stat-item">
            <span class="stat-value">{total_papers}</span>
            <span class="stat-label">Papers Indexed</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">{total_chunks:,}</span>
            <span class="stat-label">Chunks embedded</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">2-Stage</span>
            <span class="stat-label">Dual-Domain Retrieval</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">DeBERTa-NLI</span>
            <span class="stat-label">Citation Guardrail</span>
        </div>
    </div>
    """)
    
    gr.HTML('<p class="examples-header">Example questions</p>')
    
    with gr.Row():
        ex1 = gr.Button("What are the mandatory requirements for high-risk AI systems?", elem_classes="example-btn")
        ex2 = gr.Button("How do we mitigate bias to comply with Article 10 of the EU AI Act?", elem_classes="example-btn")
        ex3 = gr.Button("What explainability metrics should we use for high-risk systems?", elem_classes="example-btn")
        ex4 = gr.Button("How should providers manage data governance for training datasets under the EU AI Act?", elem_classes="example-btn")
        ex5 = gr.Button("What are the obligations for providers and deployers of General-Purpose AI (GPAI) models?", elem_classes="example-btn")
        
    with gr.Column(scale=1):
        question_input = gr.Textbox(lines=4, placeholder="Ask anything about AI compliance, fairness algorithms, EU AI Act...", label="", show_label=False)
        ask_btn = gr.Button("ASK", elem_id="ask-button")
        
    answer_output = gr.Markdown()
    
    def set_q1(): return "What are the mandatory requirements for high-risk AI systems?"
    def set_q2(): return "How do we mitigate bias to comply with Article 10 of the EU AI Act?"
    def set_q3(): return "What explainability metrics should we use for high-risk systems?"
    def set_q4(): return "How should providers manage data governance for training datasets under the EU AI Act?"
    def set_q5(): return "What are the obligations for providers and deployers of General-Purpose AI (GPAI) models?"
    
    ex1.click(set_q1, inputs=None, outputs=question_input)
    ex2.click(set_q2, inputs=None, outputs=question_input)
    ex3.click(set_q3, inputs=None, outputs=question_input)
    ex4.click(set_q4, inputs=None, outputs=question_input)
    ex5.click(set_q5, inputs=None, outputs=question_input)
    
    ask_btn.click(process_query, inputs=[question_input], outputs=[answer_output])

    gr.HTML("""
    <hr>
    <p style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--body-text-color-subdued); text-align: center; letter-spacing: 0.08em;">
    LEGAL-TO-CODE RAG PIPELINE | GRADIO | NLI CITATION GUARDRAILS ACTIVE
    </p>
    """)

if __name__ == "__main__":
    demo.launch(share=False, css=css, theme=gr.themes.Default(primary_hue="orange", secondary_hue="slate"))
