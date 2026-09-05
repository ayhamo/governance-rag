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
        progress_callback(0.95, "Loading BM25 index...")
        
    bm25_data = None
    bm25_path = Path(CHROMA_DIR) / "bm25_index.pkl"
    if bm25_path.exists():
        import pickle
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)

    if progress_callback:
        progress_callback(1.0, "Ready!")

    return embedder, reranker, collection, bm25_data


# ── retrieval ────────────────────────────────────────────────
def retrieve(question, embedder, collection, bm25_data=None, n=RETRIEVE_N):
    """
    Embed the question and find the top N most similar
    chunks in ChromaDB using cosine similarity, combined with
    BM25 sparse retrieval using Reciprocal Rank Fusion (RRF).
    """
    question_embedding = embedder.encode(question).tolist()

    chroma_results = collection.query(
        query_embeddings = [question_embedding],
        n_results        = n,
        include          = ["documents", "metadatas", "distances"]
    )

    chroma_chunks = [
        {
            "id":         id_,
            "text":       doc,
            "title":      meta["title"],
            "filename":   meta["filename"],
            "chunk_idx":  meta["chunk_idx"],
            "similarity": round(1 - dist, 3),
        }
        for id_, doc, meta, dist in zip(
            chroma_results["ids"][0],
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
            chroma_results["distances"][0],
        )
    ]

    if bm25_data is None:
        return chroma_chunks
        
    bm25 = bm25_data["bm25"]
    bm25_scores = bm25.get_scores(question.lower().split())
    
    # get top N bm25 results
    # Use python's built-in sort and take top N indices
    top_n_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n]
    
    bm25_chunks = []
    for idx in top_n_idx:
        if bm25_scores[idx] <= 0:
            continue
        bm25_chunks.append({
            "id":         bm25_data["ids"][idx],
            "text":       bm25_data["documents"][idx],
            "title":      bm25_data["metadatas"][idx]["title"],
            "filename":   bm25_data["metadatas"][idx]["filename"],
            "chunk_idx":  bm25_data["metadatas"][idx]["chunk_idx"],
            "bm25_score": round(bm25_scores[idx], 3)
        })
        
    # RRF (Reciprocal Rank Fusion)
    k = 60
    rrf_scores = {}
    
    chunk_map = {}
    for rank, chunk in enumerate(chroma_chunks):
        cid = chunk["id"]
        chunk_map[cid] = chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        
    for rank, chunk in enumerate(bm25_chunks):
        cid = chunk["id"]
        if cid not in chunk_map:
            # We don't have the cosine similarity for this chunk if it wasn't in chroma top n
            chunk["similarity"] = "BM25 only"
            chunk_map[cid] = chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        
    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    # Take top N from combined list
    final_chunks = [chunk_map[cid] for cid in sorted_ids[:n]]
    
    return final_chunks


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