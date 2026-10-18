import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
RAW_DIR = CORPUS_DIR / "raw"
CACHE_DIR = CORPUS_DIR / "cache"
MANIFEST_PATH = CORPUS_DIR / "manifest.yaml"
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_INDEX_PATH = CHROMA_DIR / "bm25_index.pkl"

# ── models ───────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
GROQ_MODEL = "openai/gpt-oss-20b"

# ── chromadb ─────────────────────────────────────────────────
COLLECTION_NAME = "governance_rag_v2"

# ── chunking (research papers only; law is chunked by article) ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
LAW_MAX_CHUNK_CHARS = 2500  # split a paragraph into its points above this

# ── ingestion ────────────────────────────────────────────────
BATCH_SIZE = 64

# ── retrieval ────────────────────────────────────────────────
RETRIEVE_N = 20
TOP_K = 5
MAX_PER_SOURCE = 2
# Minimum number of top-K slots reserved per domain when candidates exist.
DOMAIN_QUOTA = {"law": 1, "research": 1}

# ── citation verification ────────────────────────────────────
ENABLE_CITATION_VERIFICATION = True
NLI_CONFIDENCE_THRESHOLD = 0.50

# ── api keys ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
