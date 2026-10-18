import numpy as np
from rank_bm25 import BM25Okapi

from governance_rag.retriever import rerank, retrieve


class DummyEmbedder:
    def encode(self, text):
        return np.array([0.1, 0.2, 0.3])


class DummyCollection:
    def query(self, query_embeddings, n_results, include):
        return {
            "ids": [["id1", "id2"]],
            "documents": [["Doc 1 text", "Doc 2 text"]],
            "metadatas": [
                [
                    {
                        "title": "T1",
                        "domain": "law",
                        "source_id": "eu_ai_act",
                        "citation_label": "Art. 1",
                    },
                    {
                        "title": "T2",
                        "domain": "research",
                        "source_id": "arxiv_papers",
                        "paper_id": "1",
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }


def test_retrieve_dense_only():
    results = retrieve("query", DummyEmbedder(), DummyCollection(), bm25_data=None, n=2)
    assert [r["id"] for r in results] == ["id1", "id2"]
    assert results[0]["similarity"] == 0.9
    assert results[0]["domain"] == "law"
    assert results[0]["citation_label"] == "Art. 1"


def test_retrieve_hybrid_rrf():
    documents = [
        "Doc 1 text about something",
        "Doc 2 text about another",
        "Doc 3 has nothing",
        "Doc 4 is highly relevant to query",
    ]
    bm25_data = {
        "bm25": BM25Okapi([d.lower().split() for d in documents]),
        "ids": ["id1", "id2", "id3", "id4"],
        "documents": documents,
        "metadatas": [{"title": f"T{i}", "domain": "research"} for i in range(1, 5)],
    }
    results = retrieve(
        "relevant query", DummyEmbedder(), DummyCollection(), bm25_data=bm25_data, n=3
    )
    ids = [r["id"] for r in results]
    assert "id1" in ids  # dense hit
    assert "id4" in ids  # BM25 hit
    assert next(r for r in results if r["id"] == "id4")["similarity"] == "BM25 only"


class ScoreByText:
    def __init__(self, table):
        self.table = table

    def predict(self, pairs):
        return [self.table.get(text, 0.0) for _, text in pairs]


def _chunk(cid, domain, text, source_id, paper_id=None):
    return {
        "id": cid,
        "domain": domain,
        "text": text,
        "source_id": source_id,
        "title": cid,
        "meta": {"paper_id": paper_id},
    }


def test_rerank_domain_quota_pulls_in_missing_domain():
    chunks = [
        _chunk("l1", "law", "law_1", "eu_ai_act"),
        _chunk("l2", "law", "law_2", "eu_ai_act"),
        _chunk("r1", "research", "res_1", "arxiv_papers", "p1"),
    ]
    reranker = ScoreByText({"law_1": 5.0, "law_2": 4.0, "res_1": 2.0})
    out = rerank(
        "q", chunks, reranker, top_k=2, max_per_source=2, domain_quota={"law": 1, "research": 1}
    )
    assert {c["domain"] for c in out} == {"law", "research"}
    assert out[0]["id"] == "l1"  # best law chunk kept, weakest law chunk swapped out


def test_rerank_max_per_source_limits_one_instrument():
    chunks = [_chunk(f"l{i}", "law", f"law_{i}", "eu_ai_act") for i in range(4)]
    reranker = ScoreByText({f"law_{i}": 4.0 - i for i in range(4)})
    out = rerank("q", chunks, reranker, top_k=3, max_per_source=2, domain_quota={})
    assert len(out) == 2


def test_rerank_drops_irrelevant():
    chunks = [
        _chunk("a", "law", "bad", "eu_ai_act"),
        _chunk("b", "research", "good", "arxiv_papers", "p"),
    ]
    out = rerank("q", chunks, ScoreByText({"bad": -5.0, "good": 1.0}), top_k=5, domain_quota={})
    assert [c["id"] for c in out] == ["b"]


def test_tokenize_splits_legal_references():
    from governance_rag.retriever import tokenize

    assert tokenize("Art. 10(2)(f) of the EU AI Act") == [
        "art",
        "10",
        "2",
        "f",
        "of",
        "the",
        "eu",
        "ai",
        "act",
    ]


def test_parse_law_refs():
    from governance_rag.retriever import parse_law_refs

    refs = parse_law_refs(
        "What does Article 10(2)(f) require, given Annex III, point 4 and Recital 67?"
    )
    assert {"unit_type": "article", "article": 10, "paragraph": 2} in refs
    assert {"unit_type": "annex", "annex": "III", "item": "4"} in refs
    assert {"unit_type": "recital", "recital": 67} in refs
    assert parse_law_refs("Annex IV") == [{"unit_type": "annex", "annex": "IV"}]
    assert parse_law_refs("art 6") == [{"unit_type": "article", "article": 6}]
    assert parse_law_refs("how do we mitigate bias?") == []


def test_rerank_keeps_pinned_despite_low_score():
    pinned = _chunk("art10", "law", "art_10", "eu_ai_act") | {"pinned": True}
    chunks = [
        pinned,
        _chunk("r1", "research", "res_1", "arxiv_papers", "p1"),
        _chunk("r2", "research", "res_2", "arxiv_papers", "p2"),
    ]
    out = rerank(
        "q",
        chunks,
        ScoreByText({"art_10": -3.0, "res_1": 5.0, "res_2": 4.0}),
        top_k=2,
        domain_quota={},
    )
    assert out[0]["id"] == "art10"
    assert len(out) == 2
