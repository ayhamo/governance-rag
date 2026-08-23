from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import json

from src.config import CHROMA_DIR
from src.retriever import load_retriever, retrieve, rerank
from src.generator import generate_streaming
from pathlib import Path

from contextlib import asynccontextmanager

# Global variables for models
embedder = None
reranker = None
collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, reranker, collection
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        print("Warning: ChromaDB directory not found. Please run ingest.py first.")
    
    embedder, reranker, collection = load_retriever()
    print("Models and Vector DB loaded.")
    yield
    # We can add shutdown logic here if needed

app = FastAPI(title="AI Compliance Copilot API", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

class GenerateRequest(BaseModel):
    question: str
    chunks: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": embedder is not None}

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if not embedder or not collection:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    retrieved = retrieve(request.question, embedder, collection)
    reranked = rerank(request.question, retrieved, reranker)
    
    return reranked

@app.post("/generate")
def generate_endpoint(request: GenerateRequest):
    def event_generator():
        for token in generate_streaming(request.question, request.chunks):
            # We yield Server-Sent Events
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

