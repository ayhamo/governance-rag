# GovernanceRAG: EU AI Act → engineering

### Retrieval over the regulation text and ML-fairness research, with every citation checked by an NLI guardrail before you see it.

> **Live demo:** TBA
> **Status:** v0.2 — the regulation itself is now in the index, article by article.

---

## What's new in v0.2

- **The law is in the corpus.** Regulation (EU) 2024/1689 is fetched from EUR-Lex and chunked into **784 legal units**: 180 recitals, 533 article paragraphs (across all 113 articles), 71 annex items. Each unit carries its article/paragraph/annex number, chapter, the date it applies from (per Article 113), and a deep link back to EUR-Lex.
- **Citations name the law.** Sources render as `EU AI Act, Art. 10(2)` instead of a paper title, and link to the EUR-Lex anchor.
- **Source manifest.** [`corpus/manifest.yaml`](corpus/manifest.yaml) lists every source with its licence, URL and version date. `governance-rag fetch` downloads what is missing; nothing about sources is hard-coded.
- **Structural context headers.** Every law chunk is embedded with a header like *"EU AI Act, Article 10 (Data and data governance), paragraph 2"* so retrieval sees where it sits. Research chunks can get an LLM-generated header (`--llm-headers`, cached).
- **Reference pinning.** If the question names a unit ("Article 10(2)(f)", "Annex III, point 4", "Recital 60"), that unit is looked up by metadata and guaranteed a slot in the context; the reranker cannot drop it. The BM25 tokenizer also splits `10(2)(f)` into tokens so sparse search sees them.
- **Corpus as-of date** is stamped on every answer.
- **Three domains, configurable quota.** Reranking guarantees at least one `law` and one `research` chunk in the top-K; the quota is a config dict, ready for a third domain.
- **Tooling.** `pyproject.toml` + `uv`, `ruff`, a `governance-rag` CLI (`fetch` / `ingest` / `serve` / `api`), CI that actually runs the tests, a CPU Dockerfile.
- **Fixed:** the API crashed on startup; first run skipped the BM25 index; ingestion crashed on an empty corpus folder; Dockerfile still launched Streamlit.

Removed: the Colab notebook, the title-search paper downloader, `requirements.txt`, the hard-coded paper-to-domain lists.

---

## The problem

The EU AI Act requires providers of high-risk AI systems to produce technical documentation (Article 11, Annex IV) and to meet engineering obligations on data governance, logging, transparency, human oversight, accuracy and robustness (Articles 9–15). Non-compliance carries penalties of up to €35M or 7% of global turnover.

The people who have to do that work are ML engineers, and there is a translation gap:

* **Legal and compliance teams** read the 200-page regulation but cannot say which fairness metric, which log, or which test satisfies a given article.
* **ML engineers** can build the controls but do not know which article demands them or what Annex IV wants written down.

Existing "AI compliance" chatbots quote the law back at you and leave the engineer to guess. Worse, their citations are often plausible rather than true: the model paraphrases, cites, and nobody checks.

---

## What it does

Ask a question. GovernanceRAG retrieves from the **regulation text** and from **technical ML papers** at the same time, writes an answer that pairs each legal obligation with an engineering control, then runs every `[Source N]` through an NLI model against the exact chunk it cites. Claims the model cannot ground are flagged; you see the supporting sentence for the ones it can.

```
"What does Article 10(2)(f) require, and how do we test for it?"
                        │
          ┌─────────────┴─────────────┐
          │   Hybrid retrieval        │  BM25 ∪ dense (MiniLM) → RRF
          └─────────────┬─────────────┘
                        │
        [law]                       [research]
  EU AI Act, Art. 10(2)       Bias-mitigation & XAI papers
  Annex IV(2)(d) …            Equalized Odds, SHAP …
                        │
          Cross-encoder rerank · ≤2 chunks per source · domain quota
                        │
                 LLM (Groq) with [Source N] citations
                        │
          ┌─────────────┴─────────────┐
          │  NLI citation verifier    │  DeBERTa-v3 → entailed / neutral / contradicted
          └─────────────┬─────────────┘
                        │
  Answer + per-source verdict + supporting quote + EUR-Lex link + corpus as-of date
```

## Corpus

| Source | Domain | Units | Chunking |
|---|---|---|---|
| **Regulation (EU) 2024/1689** (EU AI Act), EUR-Lex OJ text of 12 July 2024 | `law` | 784 | One chunk per recital, article paragraph (with its points) and annex item. Over-long units (e.g. Art. 3 definitions) split at point boundaries. Metadata: `article`, `paragraph`, `annex`, `item`, `chapter`, `section`, `applies_from`, `source_url`. |
| **20 arXiv papers** on EU AI Act compliance, bias mitigation and explainability | `research` | 1,974 | `pymupdf4llm` markdown → citation markers stripped → 1000-char chunks, 200 overlap. |
| NIST AI RMF 1.0 | `law` | — | Downloaded, **not yet indexed** (chunker pending). |

Everything is declared in [`corpus/manifest.yaml`](corpus/manifest.yaml). The Act's HTML is committed (EUR-Lex permits reuse); the PDFs are not (`governance-rag fetch` gets them).

---

## Stack

| Component | Tool |
|---|---|
| Legal text parsing | `beautifulsoup4` + `lxml` over the EUR-Lex ELI/CONVEX HTML |
| PDF extraction | `pymupdf4llm` |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector store | ChromaDB, persistent, cosine |
| Sparse index | `rank_bm25` (Okapi) |
| Fusion | Reciprocal Rank Fusion, k = 60 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` with per-source cap and domain quota |
| LLM | `openai/gpt-oss-20b` via Groq |
| Citation guardrail | `cross-encoder/nli-deberta-v3-small`, LLM second opinion on neutral verdicts |
| UI / API | Gradio · FastAPI (`/ask` SSE, `/retrieve`, `/corpus`, `/health`) |
| Tooling | `uv`, `ruff`, `pytest`, GitHub Actions |

---

## Example

```
🛡️ NLI Citation Audit: Fully Grounded (4/4 citations verified) | Grounding: 100%

**Legal obligation — Article 10(2)(f)**
- Training, validation and testing data must be examined for biases likely to affect
  health and safety or fundamental rights [Source 1].
- Annex IV requires the technical documentation to describe the data sets used,
  including provenance and labelling procedures [Source 2].

**Technical mitigation**
- Measure group-conditional error rates; Equalized Odds post-processing balances
  TPR/FPR across protected groups [Source 3].
- …

Sources · corpus as of 2026-10-14
1  EU AI Act, Art. 10(2)   Data and data governance · applies from 2026-08-02   LAW   ✓ Verified (96%)
2  EU AI Act, Annex IV(2)  Technical documentation …                             LAW   ✓ Verified (91%)
3  Equality of Opportunity in Supervised Learning                           RESEARCH  ✓ Verified (88%)
```

Each source card links to EUR-Lex or arXiv and expands to show the sentence the verifier matched.

---

## Design decisions

1. **Article-level chunking for law, not fixed windows.** A legal citation is only useful if it names the unit. Chunk boundaries follow the regulation's own structure, and each chunk's embedded text starts with its structural header, which is the cheap half of "contextual retrieval".
2. **NLI entailment, not cosine similarity, for verification.** Contradictory sentences can sit close in embedding space. The verifier scores *(source sentence, claim)* pairs with a 3-way NLI model and only an LLM breaks ties on neutral verdicts.
3. **RRF over weighted score fusion.** BM25 keeps exact tokens like "Article 10" and "Annex III" from being lost; RRF combines without tuning weights.
4. **Domain quota in reranking.** Without it, a question about Article 10 returns five law chunks and no engineering guidance, or vice versa.
5. **Manifest over code.** Adding NIST, GDPR or a tooling-docs source is a YAML entry plus a chunker, not an edit to the retriever.

---

## Setup

```bash
git clone https://github.com/ayhamo/governance-rag.git
cd governance-rag
uv sync                      # CPU torch + dev tools. CUDA 12.8: uv sync --no-group cpu --group cu128
echo "GROQ_API_KEY=gsk_..." > .env   # free key at console.groq.com
uv run governance-rag fetch  # downloads the 20 PDFs (the Act's HTML is committed)
uv run governance-rag ingest # chunks, embeds, builds BM25 (a few minutes on CPU)
uv run governance-rag serve  # http://127.0.0.1:7860
```

Other entry points: `uv run governance-rag api` (FastAPI on :8000), `uv run governance-rag ingest --rebuild` after changing chunking, `--llm-headers` to add LLM context lines to research chunks.

### Tests

```bash
uv run pytest -m "not slow"   # ~20 s, no model downloads
uv run pytest -m slow         # real NLI model
```

Covers: manifest loading; the EUR-Lex chunker on the real Act (180 recitals, 113 articles, 13 annexes, Art. 10(2)(f), Annex III(4), Annex IV items 1–9, Article 113 application dates); hybrid retrieval and RRF; per-source cap and domain quota; citation parsing and NLI aggregation; API endpoints.

### Docker

```bash
docker build -t governance-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=... -v $PWD/chroma_db:/app/chroma_db governance-rag
```

The image does not contain the index yet; mount a built `chroma_db/` or run `fetch` + `ingest` inside the container.

---

*Not legal advice. Answers are drafts grounded in the cited text; verify with counsel.*
