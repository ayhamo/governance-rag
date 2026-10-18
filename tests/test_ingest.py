from governance_rag.ingest import chunk_documents, remove_citations


def test_remove_citations():
    raw = "This is a sentence [10]. Another sentence (Author, 2020)."
    assert remove_citations(raw) == "This is a sentence . Another sentence ."
    assert (
        remove_citations("Check this https://github.com/ or http://example.com") == "Check this  or"
    )


def test_chunk_documents_empty_is_safe():
    assert chunk_documents([]) == []


def test_chunk_documents_metadata():
    doc = {
        "source_id": "arxiv_papers",
        "paper_id": "1234.5678",
        "filename": "x.pdf",
        "title": "A Paper",
        "text": "word " * 600,
    }
    chunks = chunk_documents([doc])
    assert chunks
    m = chunks[0]["metadata"]
    assert m["domain"] == "research"
    assert m["citation_label"] == "A Paper"
    assert m["source_url"].endswith("1234.5678")
    assert chunks[0]["chunk_id"].startswith("arxiv_papers:1234.5678:")
