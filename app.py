"""Gradio UI for GovernanceRAG. Run with `governance-rag serve` or `python app.py`."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from governance_rag.config import CHROMA_DIR, COLLECTION_NAME, ENABLE_CITATION_VERIFICATION
from governance_rag.corpus.manifest import load_manifest
from governance_rag.generator import generate_streaming, source_label
from governance_rag.retriever import load_retriever, rerank, retrieve
from governance_rag.verifier import CitationVerifier

# ── engine state ─────────────────────────────────────────────
engine: dict = {}
manifest = load_manifest()


def index_is_empty() -> bool:
    if not Path(CHROMA_DIR).exists():
        return True
    try:
        import chromadb

        return (
            chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME).count()
            == 0
        )
    except Exception:  # noqa: BLE001 — any failure means "not usable"
        return True


def init_models() -> None:
    if index_is_empty():
        print(
            "No index found — running ingestion (this embeds every source; a few minutes on CPU)…"
        )
        from governance_rag.ingest import run_ingestion

        run_ingestion()

    print("Loading embedder, reranker, BM25…")
    embedder, reranker, collection, bm25 = load_retriever()
    engine.update(embedder=embedder, reranker=reranker, collection=collection, bm25=bm25)
    engine["total_chunks"] = collection.count()

    if ENABLE_CITATION_VERIFICATION:
        print("Loading NLI citation verifier…")
        engine["verifier"] = CitationVerifier()
    print("Ready.")


# ── query ────────────────────────────────────────────────────
def _esc(s: str) -> str:
    return html.escape(str(s), quote=False)


def process_query(question: str):
    if not question.strip():
        yield "<div class='answer-container'>Please enter a question.</div>"
        return

    yield "<div class='answer-container'><em>Retrieving law and research sources…</em></div>"
    hits = retrieve(question, engine["embedder"], engine["collection"], bm25_data=engine["bm25"])
    ranked = rerank(question, hits, engine["reranker"])

    if not ranked or ranked[0]["rerank_score"] < -1.5:
        yield (
            '<div class="answer-container">This looks outside the corpus. '
            "Try the EU AI Act, fairness metrics or bias mitigation.</div>"
        )
        return

    answer = ""
    for token in generate_streaming(question, ranked):
        answer += token
        yield f'<div class="answer-container">\n\n{answer}|\n\n</div>'

    audit = None
    if "verifier" in engine:
        yield (
            f'<div class="answer-container">\n\n{answer}\n\n</div>\n'
            '<div class="audit-banner audit-badge-warning"><span class="audit-badge-icon">⏳</span>'
            '<div class="audit-badge-text"><em>Checking every citation against its source…</em></div></div>'
        )
        audit = engine["verifier"].verify_answer(answer, ranked)

    yield render_result(answer, ranked, audit)


def render_result(answer: str, ranked: list[dict], audit: dict | None) -> str:
    out = ""
    if audit:
        icon = {"success": "🛡️", "warning": "⚠️", "danger": "⛔"}[audit["badge_type"]]
        score = f"Grounding: {int(audit['grounding_rate'] * 100)}%"
        out += (
            f'<div class="audit-banner audit-badge-{audit["badge_type"]}">'
            f'<span class="audit-badge-icon">{icon}</span>'
            f'<div class="audit-badge-text"><strong>NLI Citation Audit:</strong> {audit["badge"]}'
            f'<span class="audit-score">{score}</span></div></div>\n'
        )

    out += f'<div class="answer-container">\n\n{answer}\n\n</div>'
    out += f'\n<p class="sources-header">Sources · corpus as of {manifest.as_of}</p>\n\n'

    for i, c in enumerate(ranked, 1):
        domain = c.get("domain", "research")
        status_html = quote_html = ""
        if audit:
            info = audit["source_summary"].get(i, {})
            status, conf, quote = (
                info.get("status", "unreferenced"),
                info.get("confidence", 0.0),
                info.get("best_quote", ""),
            )
            status_html = {
                "verified": f'<span class="status-pill status-verified">✓ Verified ({int(conf * 100)}%)</span>',
                "contradicted": '<span class="status-pill status-contradicted">⛔ Contradicted</span>',
                "partial": '<span class="status-pill status-unsupported">⚠️ Weak grounding</span>',
                "unsupported": '<span class="status-pill status-unsupported">⚠️ Weak grounding</span>',
            }.get(status, '<span class="status-pill status-unreferenced">Not cited</span>')
            if quote:
                quote_html = (
                    '<details class="quote-accordion"><summary class="quote-toggle">Show supporting evidence</summary>'
                    f'<div class="quote-box">"{_esc(quote)}"</div></details>'
                )

        label = _esc(source_label(c))
        url = c.get("source_url", "")
        title_html = f'<a href="{url}" target="_blank" rel="noopener">{label}</a>' if url else label
        if domain == "law":
            meta_line = f"{_esc(c['meta'].get('article_title') or c['meta'].get('annex_title') or '')} · applies from {c['meta'].get('applies_from', '')}"
        else:
            meta_line = f"arXiv {c['meta'].get('paper_id', '')} · chunk {c.get('chunk_idx', '')}"
        sim = c.get("similarity", "")
        out += f"""<div class="source-card">
<span class="source-number">{i}</span>
<div class="source-info">
<div class="source-title">{title_html}</div>
<div class="source-meta">{meta_line} · similarity {sim}</div>
{quote_html}
</div>
<div class="source-actions">
<span class="score-pill">score {c["rerank_score"]}</span>
<span class="domain-pill domain-{domain}">{domain.upper()}</span>
{status_html}
</div>
</div>
"""
    return out


# ── styling ──────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
.gradio-container { font-family: 'DM Sans', sans-serif !important; max-width: 1400px !important; margin: 0 auto !important; }
.rai-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 2rem; margin-bottom: 2.5rem; }
.rai-title { font-family: 'DM Serif Display', serif; font-size: 2.4rem; font-weight: 400; color: var(--body-text-color); letter-spacing: -0.02em; line-height: 1.1; margin: 0; }
.rai-title em { font-style: italic; color: #b45309; }
.rai-subtitle { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--body-text-color-subdued); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.6rem; }
.stats-bar { display: flex; gap: 2rem; margin-bottom: 2rem; padding: 1rem 1.2rem; background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 6px; flex-wrap: wrap; }
.stat-item { display: flex; flex-direction: column; gap: 2px; }
.stat-value { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #b45309; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--body-text-color-subdued); text-transform: uppercase; letter-spacing: 0.1em; }
.answer-container { background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-left: 4px solid #b45309; border-radius: 6px; padding: 1.5rem 1.8rem; margin: 1.5rem 0; font-size: 0.95rem; line-height: 1.75; color: var(--body-text-color); }
.audit-banner { display: flex; align-items: center; gap: 0.8rem; padding: 0.8rem 1.2rem; border-radius: 6px; margin: 1rem 0; font-size: 0.88rem; border: 1px solid var(--border-color-primary); }
.audit-badge-success { background: rgba(22,163,74,.08); border-color: rgba(22,163,74,.3); color: #16a34a; }
.audit-badge-warning { background: rgba(217,119,6,.08); border-color: rgba(217,119,6,.3); color: #d97706; }
.audit-badge-danger { background: rgba(220,38,38,.08); border-color: rgba(220,38,38,.3); color: #dc2626; }
.audit-badge-text { flex: 1; color: var(--body-text-color); }
.audit-score { font-family: 'DM Mono', monospace; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; background: var(--background-fill-primary); margin-left: 0.5rem; border: 1px solid var(--border-color-primary); }
.domain-pill, .status-pill, .score-pill { font-family: 'DM Mono', monospace; font-size: 0.62rem; padding: 2px 6px; border-radius: 3px; font-weight: 500; white-space: nowrap; }
.domain-pill { letter-spacing: 0.05em; text-transform: uppercase; }
.domain-law { background: rgba(37,99,235,.12); color: #2563eb; border: 1px solid rgba(37,99,235,.3); }
.domain-research { background: rgba(124,58,237,.12); color: #7c3aed; border: 1px solid rgba(124,58,237,.3); }
.status-verified { background: rgba(22,163,74,.12); color: #16a34a; border: 1px solid rgba(22,163,74,.3); }
.status-unsupported { background: rgba(217,119,6,.12); color: #d97706; border: 1px solid rgba(217,119,6,.3); }
.status-contradicted { background: rgba(220,38,38,.12); color: #dc2626; border: 1px solid rgba(220,38,38,.3); }
.status-unreferenced { background: var(--background-fill-primary); color: var(--body-text-color-subdued); border: 1px solid var(--border-color-primary); }
.quote-accordion { margin-top: 0.6rem; border-top: 1px dashed var(--border-color-primary); padding-top: 0.4rem; }
.quote-toggle { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #b45309; cursor: pointer; user-select: none; }
.quote-box { margin-top: 0.4rem; padding: 0.5rem 0.8rem; background: var(--background-fill-primary); border-left: 2px solid #b45309; font-size: 0.82rem; font-style: italic; color: var(--body-text-color-subdued); border-radius: 0 4px 4px 0; }
.sources-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--body-text-color-subdued); letter-spacing: 0.12em; text-transform: uppercase; margin: 1.5rem 0 0.8rem; }
.source-card { background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex !important; flex-direction: row !important; align-items: flex-start !important; gap: 0.8rem !important; width: 100% !important; box-sizing: border-box !important; }
.source-number { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #b45309; min-width: 1.5rem; line-height: 1.3; }
.source-info { flex: 1 !important; min-width: 0 !important; }
.source-title { font-size: 0.85rem; color: var(--body-text-color); font-weight: 500; line-height: 1.3; word-break: break-word; }
.source-title a { color: inherit; text-decoration: underline dotted; }
.source-meta { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--body-text-color-subdued); margin-top: 3px; }
.source-actions { display: flex !important; flex-direction: column !important; align-items: flex-end !important; gap: 4px !important; flex-shrink: 0 !important; }
.score-pill { color: var(--body-text-color-subdued); background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); }
.examples-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--body-text-color-subdued); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.8rem; }
.example-btn { background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important; color: var(--body-text-color) !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important; text-transform: none !important; text-align: left !important; white-space: normal !important; padding: 1rem !important; border-radius: 6px !important; display: flex; height: 100%; align-items: flex-start; }
.example-btn:hover { background: var(--background-fill-primary) !important; border-color: #b45309 !important; }
textarea { background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important; border-radius: 6px !important; color: var(--body-text-color) !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; padding: 1rem !important; resize: none !important; }
textarea:focus { border-color: #b45309 !important; box-shadow: none !important; }
#ask-button { background: #b45309 !important; color: #fff !important; border: none !important; border-radius: 4px !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; padding: 0.6rem 1.8rem !important; margin-top: 0.5rem; width: 100%; }
#ask-button:hover { background: #92400e !important; }
.block { border: none !important; box-shadow: none !important; background: transparent !important; }
"""

EXAMPLES = [
    "What does Article 10(2)(f) of the EU AI Act require, and how do we test for it?",
    "Which Annex III category covers an AI system that screens CVs, and what obligations follow?",
    "What must the Annex IV technical documentation contain about training data?",
    "How do we mitigate bias to comply with Article 10 of the EU AI Act?",
    "What explainability measures satisfy Article 13 transparency requirements?",
]


def build_ui() -> gr.Blocks:
    law_sources = [s for s in manifest.enabled() if s.domain == "law"]
    research_sources = [s for s in manifest.enabled() if s.domain == "research"]
    n_papers = sum(len(s.meta.get("papers", [])) for s in research_sources)
    law_names = ", ".join(s.short for s in law_sources) or "—"

    with gr.Blocks() as demo:
        gr.HTML(
            f"""
    <div class="rai-header">
        <h1 class="rai-title">GovernanceRAG<br><em>EU AI Act → engineering</em></h1>
        <p class="rai-subtitle">Regulation text + ML research · hybrid retrieval · NLI-verified citations</p>
    </div>
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-value">{_esc(law_names)}</span><span class="stat-label">Law in corpus</span></div>
        <div class="stat-item"><span class="stat-value">{n_papers}</span><span class="stat-label">Research papers</span></div>
        <div class="stat-item"><span class="stat-value">{engine.get("total_chunks", 0):,}</span><span class="stat-label">Chunks indexed</span></div>
        <div class="stat-item"><span class="stat-value">{manifest.as_of}</span><span class="stat-label">Corpus as of</span></div>
        <div class="stat-item"><span class="stat-value">DeBERTa-NLI</span><span class="stat-label">Citation guardrail</span></div>
    </div>
    """
        )
        gr.HTML('<p class="examples-header">Example questions</p>')
        with gr.Row():
            buttons = [gr.Button(q, elem_classes="example-btn") for q in EXAMPLES]
        with gr.Column(scale=1):
            question = gr.Textbox(
                lines=4,
                placeholder="Ask about the EU AI Act, fairness metrics, documentation…",
                show_label=False,
            )
            ask = gr.Button("ASK", elem_id="ask-button")
        answer = gr.Markdown()

        for btn, q in zip(buttons, EXAMPLES, strict=True):
            btn.click(lambda q=q: q, inputs=None, outputs=question)
        ask.click(process_query, inputs=[question], outputs=[answer])

        gr.HTML(
            "<hr><p style=\"font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--body-text-color-subdued);"
            'text-align:center;letter-spacing:0.08em;">NOT LEGAL ADVICE · ANSWERS ARE DRAFTS GROUNDED IN THE CITED TEXT</p>'
        )
    return demo


if __name__ == "__main__":
    init_models()
    build_ui().launch(
        share=False, css=css, theme=gr.themes.Default(primary_hue="orange", secondary_hue="slate")
    )
