
import gradio as gr
from pathlib import Path
import glob

from src.ingest import run_ingestion
from src.retriever import load_retriever, retrieve, rerank
from src.generator import generate_streaming

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
total_chunks = 0
total_papers = 0

def init_models():
    global embedder, reranker, collection, bm25_data, total_chunks, total_papers
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
        print("Loading ML Engine...")
        embedder, reranker, collection, bm25_data = load_retriever()
        total_chunks = collection.count()
        from src.config import PAPERS_DIR
        total_papers = len(glob.glob(str(Path(PAPERS_DIR) / "*.pdf")))
        print("ML Engine Ready!")


def process_query(question):
    if not question.strip():
        yield "<div class='answer-container'>Please enter a question.</div>"
        return

    yield "<div class='answer-container'><em>Retrieving legal & technical sources...</em></div>"

    retrieved = retrieve(question, embedder, collection, bm25_data=bm25_data)
    reranked = rerank(question, retrieved, reranker)

    if not reranked or reranked[0]["rerank_score"] < 0.0:
        yield """<div class="answer-container">
        This question appears to be outside the scope of the compliance corpus.
        Try asking about the EU AI Act, fairness metrics, or bias mitigation.
        </div>"""
        return

    # Start generator
    generator = generate_streaming(question, reranked)
    
    full_answer = ""
    for token in generator:
        full_answer += token
        yield f'<div class="answer-container">\n\n{full_answer}|\n\n</div>'
        
    final_output = f'<div class="answer-container">\n\n{full_answer}\n\n</div>'
    
    final_output += '\n<p class="sources-header">Sources retrieved</p>\n\n'
    
    for i, chunk in enumerate(reranked, 1):
        final_output += f"""<div class="source-card">
    <span class="source-number">{i}</span>
    <div class="source-info">
        <div class="source-title">{chunk['title']}</div>
        <div class="source-meta">
            Similarity: {chunk['similarity']} | Chunk {chunk.get('chunk_idx', 'N/A')}
        </div>
    </div>
    <span class="score-pill">score {chunk['rerank_score']}</span>
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

/* Sources Cards */
.sources-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--body-text-color-subdued); letter-spacing: 0.12em; text-transform: uppercase; margin: 1.5rem 0 0.8rem; }
.source-card { background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; align-items: flex-start; gap: 0.8rem; }
.source-number { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #b45309; min-width: 1.5rem; line-height: 1.3; }
.source-info { flex: 1; }
.source-title { font-size: 0.85rem; color: var(--body-text-color); font-weight: 500; line-height: 1.3; }
.source-meta { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--body-text-color-subdued); margin-top: 3px; }
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
            <span class="stat-value">2</span>
            <span class="stat-label">Stage retrieval</span>
        </div>
    </div>
    """)
    
    gr.HTML('<p class="examples-header">Example questions</p>')
    
    with gr.Row():
        ex1 = gr.Button("What are the mandatory requirements for high-risk AI systems?", elem_classes="example-btn")
        ex2 = gr.Button("How do we mitigate bias to comply with Article 10 of the EU AI Act?", elem_classes="example-btn")
        ex3 = gr.Button("What explainability metrics should we use for high-risk systems?", elem_classes="example-btn")
        ex4 = gr.Button("How should providers manage data governance for training datasets under the EU AI Act?", elem_classes="example-btn")
        ex5 = gr.Button("What are the accountability requirements when integrating third-party models?", elem_classes="example-btn")
        
    with gr.Column(scale=1):
        question_input = gr.Textbox(lines=4, placeholder="Ask anything about AI compliance, fairness algorithms, EU AI Act...", label="", show_label=False)
        ask_btn = gr.Button("ASK", elem_id="ask-button")
        
    answer_output = gr.Markdown()
    
    def set_q1(): return "What are the mandatory requirements for high-risk AI systems?"
    def set_q2(): return "How do we mitigate bias to comply with Article 10 of the EU AI Act?"
    def set_q3(): return "What explainability metrics should we use for high-risk systems?"
    def set_q4(): return "How should providers manage data governance for training datasets under the EU AI Act?"
    def set_q5(): return "What are the accountability requirements when integrating third-party models?"
    
    ex1.click(set_q1, inputs=None, outputs=question_input)
    ex2.click(set_q2, inputs=None, outputs=question_input)
    ex3.click(set_q3, inputs=None, outputs=question_input)
    ex4.click(set_q4, inputs=None, outputs=question_input)
    ex5.click(set_q5, inputs=None, outputs=question_input)
    
    ask_btn.click(process_query, inputs=[question_input], outputs=[answer_output])

    gr.HTML("""
    <hr>
    <p style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--body-text-color-subdued); text-align: center; letter-spacing: 0.08em;">
    LEGAL-TO-CODE RAG PIPELINE | GRADIO | BUILT FOR PORTFOLIO
    </p>
    """)

if __name__ == "__main__":
    demo.launch(share=False, css=css, theme=gr.themes.Default(primary_hue="orange", secondary_hue="slate"))
