import pytest
from src.verifier import (
    extract_citations_from_text,
    extract_claims_with_citations,
    segment_chunk_sentences,
    CitationVerifier
)

def test_extract_citations_from_text():
    assert extract_citations_from_text("Statement [Source 1].") == [1]
    assert extract_citations_from_text("Statement [Source 1, 3].") == [1, 3]
    assert extract_citations_from_text("Statement [Source 2][Source 4].") == [2, 4]
    assert extract_citations_from_text("No citation.") == []

def test_extract_claims_with_citations():
    markdown = """
    - High-risk AI systems in recruitment must undergo conformity assessment [Source 1].
    - Bias mitigation techniques such as Equalized Odds should be deployed [Source 2].
    - General ungrounded statement without citations.
    """
    claims = extract_claims_with_citations(markdown)
    assert len(claims) == 2
    assert claims[0]["citations"] == [1]
    assert "High-risk AI systems" in claims[0]["claim"]
    assert claims[1]["citations"] == [2]
    assert "Equalized Odds" in claims[1]["claim"]

def test_extract_claims_table_and_abbreviations():
    markdown = """
    | Requirement | Detail | Citation |
    |---|---|---|
    | Risk Assessment | Mandated under Art. 9(2)(a), (b), (c) AIA for continuous lifecycle management | [Source 1] |
    
    Deployers must check e.g. validation metrics according to Sec. 4 [Source 2].
    """
    claims = extract_claims_with_citations(markdown)
    assert len(claims) == 2
    # Check table row preservation
    assert claims[0]["citations"] == [1]
    assert "Art. 9(2)(a)" in claims[0]["claim"]
    assert "Risk Assessment" in claims[0]["claim"]
    # Check abbreviation preservation (e.g. should not split sentence)
    assert claims[1]["citations"] == [2]
    assert "e.g. validation metrics" in claims[1]["claim"]
    assert "Sec. 4" in claims[1]["claim"]

def test_segment_chunk_sentences():
    chunk = """
    ### Article 10 Requirements
    High-risk AI systems must implement continuous data governance.
    Training and testing datasets must be representative and free of bias.
    
    * Mitigations must be documented.
    """
    sentences = segment_chunk_sentences(chunk)
    assert len(sentences) >= 2
    assert any("data governance" in s for s in sentences)
    assert any("free of bias" in s for s in sentences)

def test_verifier_mock():
    # Test verifier with a mock model to ensure logic and aggregation are intact
    class MockModel:
        def predict(self, pairs):
            # If hypothesis contains "recruitment", return high entailment
            results = []
            for premise, hyp in pairs:
                if "recruitment" in hyp.lower() and "recruitment" in premise.lower():
                    results.append([-3.0, 4.0, -1.0])  # Entailment
                elif "exempt" in hyp.lower():
                    results.append([4.0, -3.0, -1.0])  # Contradiction
                else:
                    results.append([-2.0, -2.0, 3.0])  # Neutral
            return results

    verifier = CitationVerifier(backend="mock")
    verifier.model = MockModel()

    chunks = [
        {
            "id": "c1",
            "title": "EU AI Act Employment",
            "text": "Under Annex III, AI systems deployed in recruitment and hiring are considered high-risk systems."
        },
        {
            "id": "c2",
            "title": "Mitigation Paper",
            "text": "Equalized Odds is a post-processing algorithm that equalizes true positive rates."
        }
    ]

    answer = "Recruitment AI tools are classified as high-risk systems [Source 1]."
    audit = verifier.verify_answer(answer, chunks)

    assert audit["total_citations"] == 1
    assert audit["verified_count"] == 1
    assert audit["grounding_rate"] == 1.0
    assert audit["badge_type"] == "success"
    assert audit["source_summary"][1]["status"] == "verified"
    assert audit["source_summary"][2]["status"] == "unreferenced"

def test_verifier_real_model():
    verifier = CitationVerifier()
    if verifier.model is None:
        pytest.skip("Local NLI model not available")

    chunks = [
        {
            "id": "c1",
            "title": "EU AI Act Employment",
            "text": "Under Annex III of the EU AI Act, AI systems used in recruitment and CV screening are classified as high-risk systems."
        }
    ]

    answer = "Under the EU AI Act, CV screening systems are considered high-risk [Source 1]."
    audit = verifier.verify_answer(answer, chunks)

    assert audit["total_citations"] == 1
    assert audit["verified_count"] == 1
    assert audit["source_summary"][1]["status"] == "verified"
    assert audit["source_summary"][1]["confidence"] >= 0.50

