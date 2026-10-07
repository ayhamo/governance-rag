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
    RERANK_MODEL, RETRIEVE_N, TOP_K, MAX_PER_PAPER,
    ENFORCE_DUAL_DOMAIN
)

LEGAL_PAPERS = {
    "ADAPT_Centre_Contribution_on_Implementation_of_the_2503.05758v1.pdf",
    "AI_Governance_in_the_Context_of_the_EU_AI_Act__A_B_2502.03468v1.pdf",
    "Complying_with_the_EU_AI_Act_2307.10458v1.pdf",
    "Governing_What_the_EU_AI_Act_Excludes__Accountabil_2605.01091v1.pdf",
    "Mapping_the_Regulatory_Learning_Space_for_the_EU_A_2503.05787v2.pdf",
    "Navigating_the_EU_AI_Act__A_Methodological_Approac_2403.16808v2.pdf",
    "Qualifying_and_Quantifying_Risk_Under_the_EU_AI_Ac_2608.08564v2.pdf",
    "Red_Teaming_AI_Policy__A_Taxonomy_of_Avoision_and__2506.01931v1.pdf",
    "Sustainable_AI_Regulation_2306.00292v4.pdf",
    "The_EU_AI_Act_and_the_Rights_based_Approach_to_Tec_2603.22920v1.pdf",
}

TECHNICAL_PAPERS = {
    "An_Analysis_of_the_New_EU_AI_Act_and_A_Proposed_St_2510.01281v1.pdf",
    "Are_Bias_Mitigation_Techniques_for_Deep_Learning_E_2104.00170v4.pdf",
    "Assessing_Model_Agnostic_XAI_Methods_against_EU_AI_2604.09628v2.pdf",
    "Complying_with_the_EU_AI_Act__Innovations_in_Expla_2503.15528v1.pdf",
    "Equality_of_Opportunity_in_Supervised_Learning_1610.02413v1.pdf",
    "From_Bias_to_Accountability__How_the_EU_AI_Act_Con_2505.18236v1.pdf",
    "From_Obligation_to_Specification__A_Survey_on_Vali_2607.21608v1.pdf",
    "Operationalizing_the_EU_AI_Act_in_Agile_Software_D_2608.16526v1.pdf",
    "The_Case_for_ESM3_as_a_General_Purpose_AI_Model_wi_2605.01611v1.pdf",
    "The_EU_AI_Act_in_Development_Practice__A_Pro_justi_2504.20075v1.pdf",
}

def classify_domain(filename: str, title: str = "") -> str:
    """Classify paper into 'legal' or 'technical' domain."""
    if filename in LEGAL_PAPERS:
        return "legal"
    if filename in TECHNICAL_PAPERS:
        return "technical"

    text = f"{filename} {title}".lower()
    tech_keywords = [
        "bias", "fairness", "equal", "xai", "shap", "lime", "algorithm",
        "learning", "agile", "specification", "model", "deep learning"
    ]
    legal_keywords = [
        "governance", "act", "law", "regulation", "legal", "risk",
        "policy", "rights", "compliance", "obligation"
    ]
    tech_score = sum(1 for kw in tech_keywords if kw in text)
    legal_score = sum(1 for kw in legal_keywords if kw in text)
    return "technical" if tech_score > legal_score else "legal"


# ── load models once at module level ─────────────────────────
# Loading here means models are loaded once when the app starts,
# not on every query. In Streamlit we also use st.cache_resource.

def load_retriever(progress_callback=None):
    """
    Load embedding model, reranker, and ChromaDB collection.
    Returns (embedder, reranker, collection, bm25_data).
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
            "title":      meta.get("title", ""),
            "filename":   meta.get("filename", ""),
            "chunk_idx":  meta.get("chunk_idx", 0),
            "domain":     meta.get("domain") or classify_domain(meta.get("filename", ""), meta.get("title", "")),
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
    top_n_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n]
    
    bm25_chunks = []
    for idx in top_n_idx:
        if bm25_scores[idx] <= 0:
            continue
        meta = bm25_data["metadatas"][idx]
        bm25_chunks.append({
            "id":         bm25_data["ids"][idx],
            "text":       bm25_data["documents"][idx],
            "title":      meta.get("title", ""),
            "filename":   meta.get("filename", ""),
            "chunk_idx":  meta.get("chunk_idx", 0),
            "domain":     meta.get("domain") or classify_domain(meta.get("filename", ""), meta.get("title", "")),
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
            chunk["similarity"] = "BM25 only"
            chunk_map[cid] = chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        
    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    # Take top N from combined list
    final_chunks = [chunk_map[cid] for cid in sorted_ids[:n]]
    
    return final_chunks


# ── reranking ────────────────────────────────────────────────
def rerank(
    question,
    chunks,
    reranker,
    top_k=TOP_K,
    max_per_paper=MAX_PER_PAPER,
    enforce_dual_domain=ENFORCE_DUAL_DOMAIN
):
    """
    Rerank chunks using a cross-encoder model.
    Cross-encoder scores (question, chunk) pairs together —
    more accurate than embedding similarity alone.

    Enforces diversity:
    1. Maximum max_per_paper chunks per paper.
    2. Dual-Domain balance: ensures representation of both 'legal' and 'technical'
       perspectives when relevant chunks exist.

    Returns top_k most relevant diverse chunks.
    """
    if not chunks:
        return []

    scores = reranker.predict([(question, c["text"]) for c in chunks])

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 3)
        if "domain" not in chunk or not chunk["domain"]:
            chunk["domain"] = classify_domain(chunk.get("filename", ""), chunk.get("title", ""))

    paper_counts = {}
    candidates = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    diverse = []
    for chunk in candidates:
        if chunk["rerank_score"] < -1:  # skip irrelevant chunks
            continue
        paper = chunk["filename"]
        if paper_counts.get(paper, 0) < max_per_paper:
            diverse.append(chunk)
            paper_counts[paper] = paper_counts.get(paper, 0) + 1
        if len(diverse) == top_k:
            break

    # Dual-Domain balancing: ensure representation of both 'legal' and 'technical'
    if enforce_dual_domain and len(diverse) >= 2:
        domains_present = {c.get("domain") for c in diverse}
        if len(domains_present) == 1:
            present_domain = next(iter(domains_present))
            missing_domain = "technical" if present_domain == "legal" else "legal"

            # Find the best scoring chunk from the missing domain
            best_missing = None
            for chunk in candidates:
                if chunk.get("domain") == missing_domain and chunk["rerank_score"] >= -2.0:
                    best_missing = chunk
                    break

            if best_missing is not None and best_missing not in diverse:
                diverse[-1] = best_missing
                diverse.sort(key=lambda x: x["rerank_score"], reverse=True)

    return diverse