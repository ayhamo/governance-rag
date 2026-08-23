"""
retriever.py
------------
Loads the embedding model and reranker once, exposes
retrieve() and rerank() for use by the app and generator.
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

import torch

from src.config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
    RERANK_MODEL, RETRIEVE_N, TOP_K, MAX_PER_PAPER
)


# ── load models once at module level ─────────────────────────
# Loading here means models are loaded once when the app starts,
# not on every query. In Streamlit we also use st.cache_resource.

def load_retriever(progress_callback=None):
    """
    Load embedding model, reranker, and ChromaDB collection.
    Returns (embedder, reranker, collection).
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. "
            "Run `python src/ingest.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if progress_callback:
        progress_callback(0.2, "Initializing Device...")

    if progress_callback:
        progress_callback(0.4, "Loading Embedding Model...")
    embedder  = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    if progress_callback:
        progress_callback(0.7, "Loading Cross-Encoder Reranker...")
    reranker  = CrossEncoder(RERANK_MODEL, device=device)
    
    if progress_callback:
        progress_callback(0.9, "Connecting to ChromaDB...")
    client    = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    if progress_callback:
        progress_callback(1.0, "Ready!")

    return embedder, reranker, collection


# ── retrieval ────────────────────────────────────────────────
def retrieve(question, embedder, collection, n=RETRIEVE_N):
    """
    Embed the question and find the top N most similar
    chunks in ChromaDB using cosine similarity.

    Returns a list of dicts:
        text, title, filename, chunk_idx, similarity
    """
    question_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings = [question_embedding],
        n_results        = n,
        include          = ["documents", "metadatas", "distances"]
    )

    return [
        {
            "text":       doc,
            "title":      meta["title"],
            "filename":   meta["filename"],
            "chunk_idx":  meta["chunk_idx"],
            "similarity": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ── reranking ────────────────────────────────────────────────
def rerank(question, chunks, reranker, top_k=TOP_K, max_per_paper=MAX_PER_PAPER):
    """
    Rerank chunks using a cross-encoder model.
    Cross-encoder scores (question, chunk) pairs together —
    more accurate than embedding similarity alone.

    Also enforces diversity: max max_per_paper chunks per paper.

    Returns top_k most relevant diverse chunks.
    """
    if not chunks:
        return []

    scores = reranker.predict([(question, c["text"]) for c in chunks])

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 3)

    paper_counts = {}
    diverse      = []

    for chunk in sorted(chunks, key=lambda x: x["rerank_score"], reverse=True):
        if chunk["rerank_score"] < -1:  # skip irrelevant chunks
            continue
        paper = chunk["filename"]
        if paper_counts.get(paper, 0) < max_per_paper:
            diverse.append(chunk)
            paper_counts[paper] = paper_counts.get(paper, 0) + 1
        if len(diverse) == top_k:
            break

    return diverse