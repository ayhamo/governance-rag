"""
api.py
------
Minimal HTTP surface over the engine. Models load at startup (lifespan).

    GET  /health     liveness + whether models are loaded
    GET  /corpus     manifest summary and as-of date
    POST /retrieve   {"question": ...} → reranked chunks
    POST /ask        {"question": ...} → SSE stream of tokens, then the audit
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from governance_rag.config import CHROMA_DIR, ENABLE_CITATION_VERIFICATION
from governance_rag.corpus.manifest import load_manifest
from governance_rag.generator import generate_streaming
from governance_rag.retriever import load_retriever, rerank, retrieve

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if Path(CHROMA_DIR).exists():
        embedder, reranker, collection, bm25 = load_retriever()
        state.update(embedder=embedder, reranker=reranker, collection=collection, bm25=bm25)
        if ENABLE_CITATION_VERIFICATION:
            from governance_rag.verifier import CitationVerifier

            state["verifier"] = CitationVerifier()
    else:
        print(
            f"Warning: {CHROMA_DIR} not found; run `governance-rag ingest`. Serving /health only."
        )
    yield
    state.clear()


app = FastAPI(title="GovernanceRAG API", lifespan=lifespan)


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "embedder" in state}


@app.get("/corpus")
def corpus():
    m = load_manifest()
    return {
        "as_of": m.as_of,
        "sources": [
            {
                "id": s.id,
                "domain": s.domain,
                "enabled": s.enabled,
                **{k: v for k, v in s.meta.items() if k != "papers"},
            }
            for s in m.sources.values()
        ],
    }


def _require_models():
    if "embedder" not in state:
        raise HTTPException(status_code=503, detail="Index not loaded; run `governance-rag ingest`")


@app.post("/retrieve")
def retrieve_endpoint(q: Question):
    _require_models()
    hits = retrieve(q.question, state["embedder"], state["collection"], bm25_data=state["bm25"])
    ranked = rerank(q.question, hits, state["reranker"])
    return [{k: v for k, v in c.items() if k != "meta"} | {"meta": c["meta"]} for c in ranked]


@app.post("/ask")
def ask(q: Question):
    _require_models()
    hits = retrieve(q.question, state["embedder"], state["collection"], bm25_data=state["bm25"])
    ranked = rerank(q.question, hits, state["reranker"])

    def events():
        answer = ""
        for token in generate_streaming(q.question, ranked):
            answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        audit = state["verifier"].verify_answer(answer, ranked) if "verifier" in state else None
        yield f"data: {json.dumps({'done': True, 'audit': audit, 'sources': [{'label': c['citation_label'], 'url': c['source_url'], 'domain': c['domain']} for c in ranked]})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
