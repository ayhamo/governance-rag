"""
retriever.py
------------
Loads the embedding model, reranker, ChromaDB collection and BM25 index once;
exposes retrieve() (hybrid, RRF-fused) and rerank() (cross-encoder with
per-source and per-domain diversity).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from governance_rag.config import (
    BM25_INDEX_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    DOMAIN_QUOTA,
    EMBEDDING_MODEL,
    MAX_PER_SOURCE,
    RERANK_MODEL,
    RETRIEVE_N,
    TOP_K,
)

RRF_K = 60
RERANK_FLOOR = -1.0  # below this a chunk is treated as irrelevant
QUOTA_FLOOR = -2.0  # a domain-quota fill may dip this low
MAX_PINNED = 2  # explicit legal references guaranteed a slot in the top-K

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer shared by indexing and querying: 'Art. 10(2)(f)' → art, 10, 2, f."""
    return _TOKEN.findall(text.lower())


# ── explicit legal references in the question ────────────────
_REF_PATTERNS = [
    (
        "article",
        re.compile(r"\b(?:article|art\.?)\s*(\d{1,3})(?:\s*\(\s*(\d{1,2})\s*\))?", re.I),
    ),
    (
        "annex",
        re.compile(r"\bannex\s+([IVXLC]+)\b(?:\s*,?\s*(?:point|\()\s*(\d{1,2})\s*\)?)?", re.I),
    ),
    ("recital", re.compile(r"\brecitals?\s+(\d{1,3})", re.I)),
]


def parse_law_refs(question: str) -> list[dict]:
    """
    Find references like "Article 10(2)(f)", "Art. 6", "Annex III", "Annex IV, point 2",
    "Recital 60" in a question. Returns Chroma-metadata filters, most specific first.
    """
    refs: list[dict] = []
    for kind, pat in _REF_PATTERNS:
        for m in pat.finditer(question):
            if kind == "article":
                ref = {"unit_type": "article", "article": int(m.group(1))}
                if m.group(2):
                    ref["paragraph"] = int(m.group(2))
            elif kind == "annex":
                ref = {"unit_type": "annex", "annex": m.group(1).upper()}
                if m.group(2):
                    ref["item"] = m.group(2)
            else:
                ref = {"unit_type": "recital", "recital": int(m.group(1))}
            if ref not in refs:
                refs.append(ref)
    return refs


def lookup_refs(collection, refs: list[dict], source_id: str = "eu_ai_act") -> list[dict]:
    """Exact metadata lookup of referenced legal units; each result is flagged pinned."""
    out: list[dict] = []
    for ref in refs:
        where = {"$and": [{"source_id": source_id}] + [{k: v} for k, v in ref.items()]}
        try:
            res = collection.get(where=where, include=["documents", "metadatas"])
        except Exception:  # noqa: BLE001 — collection without these fields
            continue
        for id_, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True):
            out.append(_chunk(id_, doc, meta, similarity="exact ref", pinned=True))
    # deterministic order: by article/annex/recital then paragraph/item
    out.sort(
        key=lambda c: (
            str(c["meta"].get("article", c["meta"].get("annex", c["meta"].get("recital", "")))),
            str(c["meta"].get("paragraph", c["meta"].get("item", ""))),
        )
    )
    return out


# ── loading ──────────────────────────────────────────────────
def load_retriever(progress_callback=None):
    """Return (embedder, reranker, collection, bm25_data)."""
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. Run `governance-rag ingest` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if progress_callback:
        progress_callback(0.3, "Loading embedding model")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if progress_callback:
        progress_callback(0.6, "Loading reranker")
    reranker = CrossEncoder(RERANK_MODEL, device=device)
    if progress_callback:
        progress_callback(0.9, "Opening ChromaDB")
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)

    bm25_data = None
    if Path(BM25_INDEX_PATH).exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
    if progress_callback:
        progress_callback(1.0, "Ready")
    return embedder, reranker, collection, bm25_data


# ── retrieval ────────────────────────────────────────────────
def _chunk(id_: str, text: str, meta: dict, **extra) -> dict:
    """Normalise a Chroma/BM25 hit into the chunk dict the rest of the app uses."""
    return {
        "id": id_,
        "text": text,
        "title": meta.get("title", ""),
        "domain": meta.get("domain", "research"),
        "source_id": meta.get("source_id", ""),
        "citation_label": meta.get("citation_label", meta.get("title", "")),
        "source_url": meta.get("source_url", ""),
        "chunk_idx": meta.get("chunk_idx", 0),
        "meta": meta,
        **extra,
    }


def retrieve(
    question: str, embedder, collection, bm25_data=None, n: int = RETRIEVE_N
) -> list[dict]:
    """
    Dense top-n ∪ BM25 top-n, fused with Reciprocal Rank Fusion. Legal units the
    question names explicitly ("Article 10(2)") are looked up by metadata and
    placed first, flagged `pinned`, so the reranker cannot drop them.
    """
    pinned = lookup_refs(collection, parse_law_refs(question))
    q_emb = embedder.encode(question).tolist()
    res = collection.query(
        query_embeddings=[q_emb], n_results=n, include=["documents", "metadatas", "distances"]
    )
    dense = [
        _chunk(id_, doc, meta, similarity=round(1 - dist, 3))
        for id_, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        )
    ]
    if bm25_data is None:
        return _with_pinned(pinned, dense, n)

    scores = bm25_data["bm25"].get_scores(tokenize(question))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    sparse = [
        _chunk(
            bm25_data["ids"][i],
            bm25_data["documents"][i],
            bm25_data["metadatas"][i],
            bm25_score=round(float(scores[i]), 3),
        )
        for i in top_idx
        if scores[i] > 0
    ]

    rrf: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for rank, c in enumerate(dense):
        by_id[c["id"]] = c
        rrf[c["id"]] = rrf.get(c["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, c in enumerate(sparse):
        if c["id"] not in by_id:
            c["similarity"] = "BM25 only"
            by_id[c["id"]] = c
        rrf[c["id"]] = rrf.get(c["id"], 0.0) + 1.0 / (RRF_K + rank + 1)

    ordered = sorted(rrf, key=rrf.__getitem__, reverse=True)[:n]
    return _with_pinned(pinned, [by_id[cid] for cid in ordered], n)


def _with_pinned(pinned: list[dict], ranked: list[dict], n: int) -> list[dict]:
    """Pinned chunks first (deduplicated), then the fused ranking, capped at n + pinned."""
    seen = {c["id"] for c in pinned}
    rest = [c for c in ranked if c["id"] not in seen]
    return pinned + rest[: max(n - len(pinned), n // 2)]


# ── reranking ────────────────────────────────────────────────
def rerank(
    question: str,
    chunks: list[dict],
    reranker,
    top_k: int = TOP_K,
    max_per_source: int = MAX_PER_SOURCE,
    domain_quota: dict[str, int] | None = None,
) -> list[dict]:
    """
    Cross-encoder rerank with three rules:
      1. at most `max_per_source` chunks from one paper / instrument
      2. every domain in `domain_quota` gets at least its quota of slots,
         provided a candidate scores above QUOTA_FLOOR
      3. up to MAX_PINNED chunks flagged `pinned` (explicit legal references
         in the question) are kept regardless of score
    """
    if not chunks:
        return []
    domain_quota = DOMAIN_QUOTA if domain_quota is None else domain_quota

    scores = reranker.predict([(question, c["text"]) for c in chunks])
    for c, s in zip(chunks, scores, strict=True):
        c["rerank_score"] = round(float(s), 3)
    candidates = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    must_keep = [c for c in candidates if c.get("pinned")][:MAX_PINNED]

    def source_key(c):
        # law chunks: the instrument; research chunks: the paper
        return (
            c.get("source_id", "")
            if c.get("domain") == "law"
            else c.get("meta", {}).get("paper_id") or c.get("title")
        )

    chosen: list[dict] = list(must_keep)
    per_source: dict = {}
    for c in must_keep:
        per_source[source_key(c)] = per_source.get(source_key(c), 0) + 1
    for c in candidates:
        if c in chosen:
            continue
        if c["rerank_score"] < RERANK_FLOOR:
            continue
        k = source_key(c)
        if per_source.get(k, 0) < max_per_source:
            chosen.append(c)
            per_source[k] = per_source.get(k, 0) + 1
        if len(chosen) == top_k:
            break

    # domain quota: swap in the best unmet-domain candidate for the weakest over-represented one
    for domain, quota in domain_quota.items():
        have = [c for c in chosen if c.get("domain") == domain]
        while len(have) < quota:
            fill = next(
                (
                    c
                    for c in candidates
                    if c.get("domain") == domain
                    and c not in chosen
                    and c["rerank_score"] >= QUOTA_FLOOR
                ),
                None,
            )
            if fill is None:
                break
            others = [c for c in chosen if c.get("domain") != domain and not c.get("pinned")]
            if len(chosen) < top_k:
                chosen.append(fill)
            elif others:
                chosen.remove(min(others, key=lambda c: c["rerank_score"]))
                chosen.append(fill)
            else:
                break
            have.append(fill)

    chosen.sort(key=lambda c: (not c.get("pinned"), -c["rerank_score"]))
    return chosen
