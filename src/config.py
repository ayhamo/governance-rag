import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── paths ────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── models ───────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL      = "openai/gpt-oss-20b"

# ── chromadb ─────────────────────────────────────────────────
COLLECTION_NAME = "governance_rag_papers"

# ── chunking ─────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ── ingestion ────────────────────────────────────────────────
BATCH_SIZE = 64

# ── retrieval ────────────────────────────────────────────────
RETRIEVE_N    = 20
TOP_K         = 5
MAX_PER_PAPER = 2

# ── api keys ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
