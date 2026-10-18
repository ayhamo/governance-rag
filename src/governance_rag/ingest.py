"""
ingest.py
---------
Builds the index from every enabled source in corpus/manifest.yaml.

  law sources      → one chunk per recital / article paragraph / annex item
                     (governance_rag.corpus.eurlex), each with a structural
                     context header
  research sources → pymupdf4llm markdown, citation markers stripped,
                     1000-char overlapping chunks, optional LLM context header

Idempotent: chunks already in ChromaDB are skipped. `--rebuild` drops the
collection first (needed after changing chunking or metadata).

Usage:
    governance-rag ingest [--rebuild] [--llm-headers]
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re

import chromadb
import pymupdf4llm
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from governance_rag.config import (
    BATCH_SIZE,
    BM25_INDEX_PATH,
    CACHE_DIR,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from governance_rag.corpus.eurlex import parse_eurlex_html
from governance_rag.corpus.manifest import Manifest, Source, load_manifest
from governance_rag.retriever import tokenize


# ── citation cleaning (research papers) ──────────────────────
def remove_citations(text: str) -> str:
    """Strip inline citation markers, URLs and reference lists from paper text."""
    # bracket citations: [10], [1,2], [10-12], [10–12]
    text = re.sub(r"\[[\d,\s–\-]+\]", "", text)
    # author-year citations: (Author et al., 2022), (Smith, 2020)
    text = re.sub(r"\([A-Z][^)]{2,40}\d{4}\)", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # pymupdf4llm image placeholders
    text = re.sub(r"==>[^<]*intentionally omitted[^<]*<==", "", text)
    text = re.sub(r"https?://\S+", "", text)
    # reference list entries — "Author, A. (year). Title. Journal"
    text = re.sub(r"^[\-\*>]?\s*[A-Z][a-z]+,\s+[A-Z]\..*\(\d{4}\).*$", "", text, flags=re.MULTILINE)
    # standalone footnote numbers at start of line, blockquote markers
    text = re.sub(r"^\s*>\s*\d+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # references section onward
    text = re.sub(r"\n(References|Bibliography|REFERENCES)\n.*", "", text, flags=re.DOTALL)
    # "Section 3.2" cross references
    text = re.sub(r"\b(Sub)?[Ss]ection\s+\d+(\.\d+)*", "", text)
    return text.strip()


# ── research papers ──────────────────────────────────────────
def extract_documents(src: Source) -> list[dict]:
    """Extract clean markdown from each PDF listed under the source."""
    documents = []
    for paper in src.meta.get("papers", []):
        pdf_path = src.path / paper["file"]
        if not pdf_path.exists():
            print(f"⚠  missing {paper['file']} — run `governance-rag fetch`")
            continue
        try:
            text = remove_citations(pymupdf4llm.to_markdown(str(pdf_path)))
        except Exception as e:  # noqa: BLE001 — one bad PDF must not stop the run
            print(f"✗  failed {paper['file']}: {e}")
            continue
        if len(text) < 500:
            print(f"⚠  skipping {paper['file']} — too little text")
            continue
        documents.append(
            {
                "source_id": src.id,
                "paper_id": paper["id"],
                "filename": paper["file"],
                "title": paper["title"],
                "text": text,
            }
        )
        print(f"✓  {paper['title'][:70]}")
    print(f"Extracted: {len(documents)} documents from {src.id}")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split research documents into overlapping chunks. Empty input → []."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". ", "\n", " ", ""],
    )
    all_chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["text"])):
            chunk_text = re.sub(r"^[.\s#\-]+", "", chunk_text).strip()
            if len(chunk_text) < 100:
                continue
            all_chunks.append(
                {
                    "chunk_id": f"{doc['source_id']}:{doc['paper_id']}:chunk_{i:04d}",
                    "text": chunk_text,
                    "metadata": {
                        "domain": "research",
                        "source_id": doc["source_id"],
                        "paper_id": doc["paper_id"],
                        "filename": doc["filename"],
                        "title": doc["title"],
                        "short": doc["title"],
                        "citation_label": doc["title"],
                        "chunk_idx": i,
                        "source_url": f"https://arxiv.org/abs/{doc['paper_id']}",
                    },
                }
            )
    print(
        f"Research chunks: {len(all_chunks)} (avg {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars)"
    )
    return all_chunks


# ── optional LLM context headers (Anthropic "contextual retrieval") ──
_HEADER_PROMPT = """<document>
{doc}
</document>
Here is a chunk from that document:
<chunk>
{chunk}
</chunk>
Write one sentence (max 40 words) situating this chunk within the document, so it can be found by search. Reply with the sentence only."""


def add_llm_headers(chunks: list[dict], documents: list[dict]) -> None:
    """Prefix each research chunk with an LLM-generated context line. Cached on disk."""
    if not GROQ_API_KEY:
        print("⚠  --llm-headers set but GROQ_API_KEY is missing; skipping")
        return
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "llm_headers.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    doc_text = {d["paper_id"]: d["text"][:12000] for d in documents}  # first ~3k tokens

    for c in tqdm(chunks, desc="LLM headers"):
        key = hashlib.sha1((c["chunk_id"] + c["text"]).encode()).hexdigest()
        if key not in cache:
            prompt = _HEADER_PROMPT.format(doc=doc_text[c["metadata"]["paper_id"]], chunk=c["text"])
            try:
                r = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=80,
                )
                cache[key] = (r.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 — rate limits: keep what we have
                print(f"header failed for {c['chunk_id']}: {e}")
                continue
            cache_path.write_text(json.dumps(cache, indent=0))
        header = cache[key]
        if header:
            c["metadata"]["context_header"] = header
            c["text"] = f"{header}\n{c['text']}"


# ── law sources ──────────────────────────────────────────────
def law_chunks(src: Source) -> list[dict]:
    if src.kind != "eurlex_html":
        print(f"⚠  no chunker for kind={src.kind} ({src.id}); skipping")
        return []
    if not src.path.exists():
        print(f"⚠  missing {src.path.name} — run `governance-rag fetch`")
        return []
    parsed = parse_eurlex_html(
        src.path,
        source_id=src.id,
        instrument=src.meta["instrument"],
        short=src.short,
        link_base=src.meta.get("link_base", src.meta.get("url", "")),
        version_date=str(src.meta.get("version_date", "")),
    )
    out = []
    for lc in parsed:
        meta = {
            **lc.meta,
            "unit_type": lc.unit_type,
            "citation_label": lc.citation_label,
            "context_header": lc.header,
            "title": f"{src.short}, {lc.citation_label}",
        }
        out.append({"chunk_id": lc.chunk_id, "text": lc.text, "metadata": meta})
    kinds = {}
    for c in out:
        kinds[c["metadata"]["unit_type"]] = kinds.get(c["metadata"]["unit_type"], 0) + 1
    print(f"Law chunks from {src.id}: {len(out)} {kinds}")
    return out


# ── embedding + storage ──────────────────────────────────────
def embed_and_store(all_chunks: list[dict], rebuild: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Dropped collection {COLLECTION_NAME}")
        except Exception:  # noqa: BLE001 — collection may not exist
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    existing = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing]
    print(f"\nAlready embedded: {len(existing)}   New: {len(new_chunks)}")
    if not new_chunks:
        return collection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding with {EMBEDDING_MODEL} on {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding"):
        batch = new_chunks[i : i + BATCH_SIZE]
        embeddings = model.encode([c["text"] for c in batch], show_progress_bar=False).tolist()
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[_flat(c["metadata"]) for c in batch],
        )
    print(f"Total embeddings: {collection.count()}")
    return collection


def _flat(meta: dict) -> dict:
    """Chroma metadata must be flat str/int/float/bool."""
    return {
        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
        for k, v in meta.items()
        if v is not None
    }


# ── bm25 ─────────────────────────────────────────────────────
def build_bm25_index() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    if not data["documents"]:
        print("No documents in ChromaDB; BM25 index not built.")
        return
    bm25 = BM25Okapi([tokenize(doc) for doc in data["documents"]])
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "ids": data["ids"],
                "documents": data["documents"],
                "metadatas": data["metadatas"],
            },
            f,
        )
    print(f"BM25 index: {len(data['ids'])} docs → {BM25_INDEX_PATH.name}")


# ── main ─────────────────────────────────────────────────────
def collect_chunks(manifest: Manifest, llm_headers: bool = False) -> list[dict]:
    chunks: list[dict] = []
    for src in manifest.enabled():
        print(f"\n── {src.id} [{src.domain}] ──")
        if src.domain == "law":
            chunks.extend(law_chunks(src))
        elif src.kind == "arxiv_pdf_dir":
            docs = extract_documents(src)
            research = chunk_documents(docs)
            if llm_headers:
                add_llm_headers(research, docs)
            chunks.extend(research)
        else:
            print(f"⚠  unknown source kind {src.kind}; skipping")
    return chunks


def run_ingestion(
    rebuild: bool = False, llm_headers: bool = False, bm25_only: bool = False
) -> None:
    if bm25_only:
        build_bm25_index()
        return
    print("=" * 60)
    print("INGESTION PIPELINE")
    print("=" * 60)
    manifest = load_manifest()
    print(f"Corpus as of {manifest.as_of}; sources: {[s.id for s in manifest.enabled()]}")

    chunks = collect_chunks(manifest, llm_headers=llm_headers)
    if not chunks:
        print("\nNothing to index. Run `governance-rag fetch` first.")
        return
    embed_and_store(chunks, rebuild=rebuild)
    build_bm25_index()
    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
