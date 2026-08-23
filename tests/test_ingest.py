from src.ingest import remove_citations

def test_remove_citations():
    raw = "This is a sentence [10]. Another sentence (Author, 2020)."
    cleaned = remove_citations(raw)
    assert cleaned == "This is a sentence . Another sentence ."
    
    raw_with_urls = "Check this https://github.com/ or http://example.com"
    cleaned_urls = remove_citations(raw_with_urls)
    assert cleaned_urls == "Check this  or"
