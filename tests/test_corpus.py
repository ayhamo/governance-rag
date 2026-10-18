"""Manifest loading and the EUR-Lex article-level chunker, run on the real Act."""

import pytest

from governance_rag.config import MANIFEST_PATH
from governance_rag.corpus.eurlex import applies_from, parse_eurlex_html
from governance_rag.corpus.manifest import load_manifest


def test_manifest_loads_and_labels_domains():
    m = load_manifest(MANIFEST_PATH)
    assert m.as_of
    assert m.domain_of("eu_ai_act") == "law"
    assert m.domain_of("arxiv_papers") == "research"
    assert m.sources["eu_ai_act"].enabled
    assert len(m.sources["arxiv_papers"].meta["papers"]) == 20


@pytest.fixture(scope="module")
def act_chunks():
    m = load_manifest(MANIFEST_PATH)
    src = m.sources["eu_ai_act"]
    if not src.path.exists():
        pytest.skip("EU AI Act HTML not fetched")
    return parse_eurlex_html(
        src.path,
        source_id="eu_ai_act",
        instrument=src.meta["instrument"],
        short=src.short,
        link_base=src.meta["link_base"],
        version_date=src.meta["version_date"],
    )


def test_structural_counts(act_chunks):
    by_type = {}
    for c in act_chunks:
        by_type.setdefault(c.unit_type, set())
    recitals = {c.meta["recital"] for c in act_chunks if c.unit_type == "recital"}
    articles = {c.meta["article"] for c in act_chunks if c.unit_type == "article"}
    annexes = {c.meta["annex"] for c in act_chunks if c.unit_type == "annex"}
    assert len(recitals) == 180
    assert len(articles) == 113
    assert len(annexes) == 13
    assert all(c.chunk_id.startswith("eu_ai_act:") for c in act_chunks)
    assert len({c.chunk_id for c in act_chunks}) == len(act_chunks)


def test_article_10_paragraph_2_has_bias_point(act_chunks):
    c = next(c for c in act_chunks if c.chunk_id == "eu_ai_act:art_10.2")
    assert c.citation_label == "Art. 10(2)"
    assert c.meta["article_title"] == "Data and data governance"
    assert c.meta["chapter"] == "III"
    assert c.header.startswith("EU AI Act, Article 10 (Data and data governance), paragraph 2")
    assert "(f) examination in view of possible biases" in c.body
    assert c.meta["source_url"].endswith("#art_10")


def test_annex_iv_has_nine_items(act_chunks):
    items = {
        int(c.meta["item"])
        for c in act_chunks
        if c.unit_type == "annex" and c.meta["annex"] == "IV" and "item" in c.meta
    }
    assert items == set(range(1, 10))
    item2 = next(c for c in act_chunks if c.chunk_id == "eu_ai_act:annex_IV.2")
    assert "data requirements" in item2.body.lower()


def test_annex_iii_lists_employment(act_chunks):
    item4 = next(c for c in act_chunks if c.chunk_id == "eu_ai_act:annex_III.4")
    assert "recruitment" in item4.body.lower()
    assert item4.citation_label == "Annex III(4)"


def test_definitions_article_is_split(act_chunks):
    art3 = [c for c in act_chunks if c.unit_type == "article" and c.meta["article"] == 3]
    assert len(art3) > 1
    assert all(len(c.body) <= 2600 for c in art3)


def test_applies_from_rules():
    assert applies_from("II", None, 5) == "2025-02-02"  # prohibited practices
    assert applies_from("III", "4", 28) == "2025-08-02"  # notifying authorities
    assert applies_from("V", None, 53) == "2025-08-02"  # GPAI
    assert applies_from("III", "2", 10) == "2026-08-02"  # high-risk requirements
    assert applies_from("V", None, 101) == "2026-08-02"  # GPAI fines: general date
