import pytest
from src.retriever import retrieve
from rank_bm25 import BM25Okapi

class DummyEmbedder:
    def encode(self, text):
        import numpy as np
        return np.array([0.1, 0.2, 0.3])

class DummyCollection:
    def query(self, query_embeddings, n_results, include):
        return {
            "ids": [["id1", "id2"]],
            "documents": [["Doc 1 text", "Doc 2 text"]],
            "metadatas": [[{"title": "T1", "filename": "f1.pdf", "chunk_idx": 0}, {"title": "T2", "filename": "f2.pdf", "chunk_idx": 1}]],
            "distances": [[0.1, 0.2]]
        }

def test_retrieve_without_bm25():
    embedder = DummyEmbedder()
    collection = DummyCollection()
    
    results = retrieve("query", embedder, collection, bm25_data=None, n=2)
    
    assert len(results) == 2
    assert results[0]["id"] == "id1"
    assert results[0]["similarity"] == 0.9  # 1 - 0.1
    assert results[1]["id"] == "id2"

def test_retrieve_with_bm25():
    embedder = DummyEmbedder()
    collection = DummyCollection()
    
    documents = ["Doc 1 text about something", "Doc 2 text about another", "Doc 3 has nothing relevant", "Doc 4 is highly relevant to query"]
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "bm25": bm25,
        "ids": ["id1", "id2", "id3", "id4"],
        "documents": documents,
        "metadatas": [
            {"title": "T1", "filename": "f1.pdf", "chunk_idx": 0},
            {"title": "T2", "filename": "f2.pdf", "chunk_idx": 1},
            {"title": "T3", "filename": "f3.pdf", "chunk_idx": 2},
            {"title": "T4", "filename": "f4.pdf", "chunk_idx": 3}
        ]
    }
    
    results = retrieve("relevant query", embedder, collection, bm25_data=bm25_data, n=2)
    
    # We expect some combined results from ChromaDB (id1, id2) and BM25 (id4 since it contains 'relevant query')
    assert len(results) <= 4
    # Because 'id4' has strong keyword match, it should be in the results.
    # The ChromaDB results ('id1', 'id2') should also be there.
    result_ids = [r["id"] for r in results]
    
    # id1, id2 are from ChromaDB dummy.
    # id4 is matched strongly by BM25.
    assert "id1" in result_ids
    assert "id4" in result_ids

