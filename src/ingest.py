"""
ingest.py
---------
Extracts text from PDFs using pymupdf4llm (clean markdown output),
chunks it, embeds it, and stores in ChromaDB.

SCALABLE: if chroma_db already exists and has embeddings, skips already
processed chunks and only embeds new ones. Run freely — safe to rerun.

Usage:
    python src/ingest.py
"""

import re, os, pickle
from pathlib import Path

import fitz
import pymupdf4llm
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from src.config import (
    PAPERS_DIR, CHROMA_DIR, EMBEDDING_MODEL, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE, BM25_INDEX_PATH
)


# ── title extraction ─────────────────────────────────────────
def extract_title(pdf_path):
    """
    Extract title by finding the first heading that looks like
    a real paper title — skips journal headers, metadata blocks,
    and short non-title headings.
    """
    # hardcoded overrides for papers where automatic extraction fails
    overrides = {
        "2502.18359v1.pdf": "Responsible AI Agents",
    }
    if Path(pdf_path).name in overrides:
        return overrides[Path(pdf_path).name]

    try:
        text  = pymupdf4llm.to_markdown(str(pdf_path))
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line.startswith('#'):
                continue

            title = re.sub(r'^#+\s*', '', line).strip()
            title = re.sub(r'\*\*', '', title).strip()
            title = re.sub(r"[_']", '', title).strip()

            # remove leading OPEN / OPEN ACCESS artifact from journal badges
            if title.upper().startswith('OPEN '):
                title = title[5:].strip()

            if not (15 < len(title) < 200):
                continue
            if any(c in title for c in ['©', '@', 'http', 'DOI', 'doi']):
                continue
            skip_words = [
                'abstract', 'introduction', 'conclusion', 'references',
                'open access', 'edited by', 'reviewed by', 'draft',
                'published', 'received', 'accepted', 'contents',
                'supplementary', 'acknowledgement', 'appendix',
                'journal', 'editorial', 'keywords',
                'type review', 'acm reference', 'ccs concepts',
                'computers and education'
            ]
            if any(w in title.lower() for w in skip_words):
                continue
            # skip single or two word all-caps — always section headers
            if title.isupper() and len(title.split()) <= 2:
                continue
            # skip if just numbers or date
            if re.match(r'^[\d\s\.\-\/]+$', title):
                continue

            return title

    except Exception:
        pass

    return Path(pdf_path).stem


# ── citation cleaning ────────────────────────────────────────
def remove_citations(text):
    """
    Remove inline citation markers from extracted text.
    pymupdf4llm handles layout cleaning — we only need to strip citations.
    """
    # bracket citations: [10], [1,2], [10-12], [10–12]
    text = re.sub(r'\[[\d,\s–\-]+\]', '', text)
    # author-year citations: (Author et al., 2022), (Smith, 2020)
    text = re.sub(r'\([A-Z][^)]{2,40}\d{4}\)', '', text)
    # clean up extra whitespace left after removal
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'==>[^<]*intentionally omitted[^<]*<==', '', text)
    # remove pymupdf4llm image placeholders
    text = re.sub(r'==>[^<]*intentionally omitted[^<]*<==', '', text)

    # remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # remove reference list entries — "Author, A. (year). Title. Journal"
    text = re.sub(r'^[\-\*>]?\s*[A-Z][a-z]+,\s+[A-Z]\..*\(\d{4}\).*$', '', text, flags=re.MULTILINE)

    # remove standalone footnote numbers at start of line
    text = re.sub(r'^\s*>\s*\d+\s+', '', text, flags=re.MULTILINE)

    # remove markdown blockquote markers
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # remove references section
    text = re.sub(r'\n(References|Bibliography|REFERENCES)\n.*', '', text, flags=re.DOTALL)
    # remove subsection/section references like "Subsection 5.8", "Section 3.2"
    text = re.sub(r'\b(Sub)?[Ss]ection\s+\d+(\.\d+)*', '', text)
    return text.strip()


# ── PDF extraction ───────────────────────────────────────────
def extract_documents(papers_dir):
    """
    Extract clean markdown text from all PDFs using pymupdf4llm.
    pymupdf4llm handles headers, footers, columns, and layout automatically.
    """
    pdf_files = sorted(Path(papers_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {papers_dir}\n")

    documents = []
    for pdf_path in pdf_files:
        try:
            # pymupdf4llm extracts clean markdown — handles most PDF junk
            text = pymupdf4llm.to_markdown(str(pdf_path))

            # strip citation markers
            text = remove_citations(text)

            if len(text) < 500:
                print(f"⚠  Skipping {pdf_path.name} — too little text")
                continue

            title = extract_title(pdf_path)
            documents.append({
                "filename": pdf_path.name,
                "title":    title,
                "text":     text,
            })
            print(f"✓  {title[:70]}")

        except Exception as e:
            print(f"✗  Failed: {pdf_path.name} — {e}")

    print(f"\nExtracted: {len(documents)} documents")
    return documents


# ── chunking ─────────────────────────────────────────────────
def chunk_documents(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", ". ", "\n", " ", ""]
    )

    all_chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["text"])):
            chunk_text = re.sub(r'^[.\s#\-]+', '', chunk_text).strip()
            if not chunk_text or len(chunk_text) < 100:
                continue
            all_chunks.append({
                "chunk_id":  f"{doc['filename']}__chunk_{i:04d}",
                "filename":  doc["filename"],
                "title":     doc["title"],
                "chunk_idx": i,
                "text":      chunk_text,
            })

    print(f"Total chunks:            {len(all_chunks)}")
    print(f"Avg chunk size:          {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars")
    print(f"Avg chunks per document: {len(all_chunks) // len(documents)}")
    return all_chunks


# ── embedding + storage ──────────────────────────────────────
def embed_and_store(all_chunks):
    """
    Embed chunks and store in ChromaDB.
    Skips chunks already in the collection — safe to rerun.
    If collection doesn't exist yet, creates it from scratch.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection    = chroma_client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks   = [c for c in all_chunks if c["chunk_id"] not in existing_ids]

    print(f"\nAlready embedded: {len(existing_ids)}")
    print(f"New chunks:       {len(new_chunks)}")

    if not new_chunks:
        print("Nothing to do — all chunks already embedded.")
        return collection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} on device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE),
                  desc="Embedding", total=total_batches):
        batch      = new_chunks[i : i + BATCH_SIZE]
        embeddings = model.encode(
            [c["text"] for c in batch],
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids        = [c["chunk_id"]  for c in batch],
            embeddings = embeddings,
            documents  = [c["text"]      for c in batch],
            metadatas  = [{
                "title":     c["title"],
                "filename":  c["filename"],
                "chunk_idx": c["chunk_idx"],
            } for c in batch]
        )

    print(f"\nDone. Total embeddings in DB: {collection.count()}")
    return collection


# ── bm25 ─────────────────────────────────────────────────────
def build_bm25_index():
    print("\nBuilding BM25 index...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    
    if not documents:
        print("No documents found in ChromaDB to build BM25 index.")
        return
        
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    index_data = {
        "bm25": bm25,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas
    }
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
        
    print(f"BM25 index saved to {BM25_INDEX_PATH}")


# ── main ─────────────────────────────────────────────────────
def run_ingestion():
    print("=" * 60)
    print("INGESTION PIPELINE")
    print("=" * 60)

    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = client.get_collection(COLLECTION_NAME)
            count      = collection.count()
            if count > 0:
                print(f"\nVector store found — {count} embeddings already stored.")
                print("Running in incremental mode — only new papers will be embedded.\n")
            else:
                print("\nVector store exists but is empty. Running full ingestion.\n")
        except Exception:
            print("\nNo collection found. Running full ingestion.\n")
    else:
        print("\nNo vector store found. Running full ingestion.\n")

    documents  = extract_documents(PAPERS_DIR)
    all_chunks = chunk_documents(documents)
    embed_and_store(all_chunks)
    build_bm25_index()

    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()