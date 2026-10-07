# GovernanceRAG: Bridging AI Law and Engineering
### A production-grade RAG pipeline mapping the EU AI Act & NIST frameworks directly to technical mitigation algorithms.

> **Live demo:** TBA

---

## The Problem

Enterprise AI teams face a severe regulatory hurdle. Non-compliance with frameworks like the **EU AI Act** carries penalties of up to €35M or 7% of global annual revenue, while the **NIST AI Risk Management Framework** is becoming the de facto standard for enterprise procurement. 

However, there is a massive translation gap in the industry:
* **Legal and Compliance teams** read 200-page regulations but do not know how to write code to perform bias checks, data governance audits, or feature attribution.
* **Machine Learning Engineers** know how to code, but do not know which specific mathematical fairness metric (e.g., Equalized Odds vs. Disparate Impact) legally satisfies "Article 10" or what exact logs are required to prove compliance during an audit.

Existing "AI Compliance" chatbots only quote the law (e.g., *"You must mitigate bias"*), leaving engineers to guess how to implement it.

---

## The Solution

**GovernanceRAG** is a specialized Retrieval-Augmented Generation (RAG) system built to serve as a translation layer. It queries across two knowledge bases simultaneously to map legal requirements directly to technical implementations.

*Note: The current corpus focuses heavily on the EU AI Act and ML Fairness. I plan to expand this to include more global laws and policies (e.g., CPPA, GDPR, US State AI laws).*

**How it works:**
You describe your AI system (*"We are deploying an LLM for CV screening"*). The system:
1. Classifies the legal risk tier (e.g., *High-Risk under Annex III*).
2. Identifies mandatory legal checks (e.g., *Article 10 data governance*).
3. **Retrieves and recommends the specific technical algorithms** from academic papers required to pass an audit.
4. **Verifies citations using NLI guardrails**, ensuring that every claim in the response is mathematically and factually entailed by the source papers.

---

## Architecture

```
User: "We are deploying an LLM for CV screening. What are our requirements?"
                                      │
                         ┌────────────┴────────────┐
                         │      Hybrid Search      │
                         │   (BM25 + ChromaDB)     │
                         └────────────┬────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
               [Official Law]             [Technical Papers]
               EU AI Act Annex III         Debiasing & Fairness Literature
               High-Risk Classification    Equalized Odds / Disparate Impact
                         │                         │
                         └────────────┬────────────┘
                                      │
                             Cross-Encoder Reranking
                           (Dual-Domain Diversity)
                                      │
                            LLM (Qwen via Groq API)
                                      │
                         ┌────────────┴────────────┐
                         │  NLI Citation Verifier  │
                         │  (DeBERTa-v3 Guardrail) │
                         └────────────┬────────────┘
                                      │
Answer with Auditable Citations:
- Legal Classification: High-Risk (EU AI Act Article 6 / Annex III) [Source 1: Verified (95%)]
- Mandatory Checks: Bias testing on demographic groups (Article 10) [Source 2: Verified (98%)]
- Recommended Technical Algorithm: Equalized Odds Post-Processing [Source 3: Verified (92%)]
- Explainability Requirements: SHAP/LIME feature attribution logs [Source 4: Verified (94%)]
```

**Key Innovations:**
* **Two-Stage Dual-Domain Retrieval:** Fast hybrid retrieval (BM25 + ChromaDB fused via RRF) narrows down chunks; a cross-encoder scores each (query, chunk) pair while guaranteeing that both *Legal* and *Technical* domains are represented in the top-$K$ prompt context.
* **Hybrid Search (BM25 + Semantic):** Dense vectors capture semantic intent, while sparse BM25 guarantees exact regulatory acronyms (e.g., *"ISO 42001"*, *"Article 10"*) are never missed.
* **Auditable Citations (NLI Anti-Hallucination Guardrail):** Every `[Source N]` claim is post-audited by a Natural Language Inference (NLI) model (`cross-encoder/nli-deberta-v3-small`) to ensure statements strictly entail the evidence text, flagging ungrounded claims with visual confidence badges.
* **Interactive UI Evidence Quotes:** Click-to-expand evidence quotes directly in the Gradio UI reveal the exact sentences from the literature that substantiate each statement.
* **Automated Evaluation:** Scored via DeepEval / Ragas for Faithfulness and Context Precision. *(Phase 4)*

---

## Stack

| Component | Tool | Purpose |
|---|---|---|
| **PDF Extraction** | `pymupdf4llm` | Clean Markdown extraction preserving layout & structure |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Overlapping semantic chunking (1000 chars, 200 overlap) |
| **Dense Embeddings** | `all-MiniLM-L6-v2` | Fast, high-accuracy semantic vector representations |
| **Vector Store** | ChromaDB (persistent) | Local persistent cosine similarity search |
| **Sparse Index** | `rank_bm25` (Okapi BM25) | Exact keyword matching for legal articles & acronyms |
| **Retrieval Fusion** | Reciprocal Rank Fusion (RRF) | Rank-based combination of sparse and dense candidates |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pairwise query-document reranking with dual-domain balancing |
| **LLM** | `openai/gpt-oss-20b` / `Qwen` (via Groq API) | Ultra-fast compliance structured generation |
| **Citation Guardrail** | `cross-encoder/nli-deberta-v3-small` + Groq Judge | 3-way NLI classification (`Entailment`, `Neutral`, `Contradiction`) |
| **UI** | Gradio | Responsive UI with real-time streaming & interactive audit badges |
| **Evaluation (TBA)** | DeepEval / Ragas | Automated CI/CD quality gates & benchmark metrics |

---

## Corpus

The vector database is built on a curated collection of **20 technical and regulatory research papers** sourced from arXiv, balanced between official legal analyses of the EU AI Act and machine learning bias mitigation/explainability literature:

| Domain | Representative Papers |
|---|---|
| **Legal & Governance (10 Papers)** | *Complying with the EU AI Act*, *Qualifying and Quantifying Risk Under the EU AI Act*, *AI Governance in the Context of the EU AI Act*, *Navigating the EU AI Act*, *Red Teaming AI Policy*, etc. |
| **Technical & Engineering (10 Papers)** | *Equality of Opportunity in Supervised Learning* (Equalized Odds), *Are Bias Mitigation Techniques for Deep Learning Effective?*, *Assessing Model-Agnostic XAI Methods (SHAP/LIME)*, *Operationalizing the EU AI Act in Agile Software Development*, etc. |

> **Automated Ingestion:** Use `python scripts/download_papers.py` to fetch all 20 curated PDFs from the arXiv API.

---

## Example Output & Citation Audit

> **Note:** The system structures outputs for enterprise engineering teams, clearly distinguishing the Legal Obligation from the Technical Mitigation, followed by an NLI Citation Audit badge.

```markdown
🛡️ NLI Citation Audit: Fully Grounded (3/3 citations verified via NLI) | Grounding: 100%

Legal Obligation (Article 10 – EU AI Act)
- Data Governance & Quality: High-risk AI systems must be developed using training, validation, and testing data sets that are relevant, sufficiently representative, free of errors, and complete for the intended purpose [Source 1].
- Bias Examination: Article 10(2)(f) explicitly requires that data sets be examined for biases likely to affect health and safety or to have a negative impact on fundamental rights [Source 2].

Technical Mitigation (Engineering Implementation)
- Pre-processing: Apply statistical checks for missing values and disparate impact across protected attributes [Source 1].
- In-processing / Post-processing: Incorporate Equalized Odds post-processing or adversarial debiasing to balance TPR/FPR across groups [Source 3].
- Documentation: Maintain data provenance logs and SHAP explainability summaries for conformity assessment audits [Source 2].
```

In the UI, each source card displays:
* Domain Tag (`[LEGAL]` or `[TECHNICAL]`)
* Rerank Score (`score 8.375`)
* Verification Status (`✓ Verified (95%)` or `⚠️ Weak Grounding`)
* Expandable accordion revealing the exact supporting quote from the research paper.

---

## Key Design Decisions

1. **Two-Tier NLI Citation Verification**:
   * *Problem*: Cosine similarity is insufficient for legal verification because contradictory statements can share high semantic overlap. Strict MNLI models can also produce false-neutral classifications on slight vocabulary variations.
   * *Design*: We employ a two-tier strategy. First, candidate chunk sentences are processed through `cross-encoder/nli-deberta-v3-small` for instant, local verification. If classified as neutral due to strict phrasing nuances, the system consults an LLM-as-a-judge via Groq to confirm semantic grounding.
2. **Dual-Domain Balanced Reranking**:
   * Standard vector search often returns clusters of only regulatory papers or only ML papers. GovernanceRAG classifies all papers into `legal` and `technical` domains and enforces that both perspectives are represented in the top-$K$ reranked context window.
3. **Reciprocal Rank Fusion (RRF)**:
   * Combines sparse BM25 scores with dense cosine similarity scores without requiring arbitrary weight tuning:
     $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$

---

## Testing

The project includes an automated test suite verifying ingestion, hybrid retrieval, dual-domain balancing, and NLI citation verification:

```bash
# Run all tests
python -m pytest tests/ -v
```

**Test Coverage (11 tests):**
* `test_api.py`: FastAPI / healthcheck endpoint tests
* `test_ingest.py`: Citation cleaning and layout extraction
* `test_retriever.py`: Dense search, BM25 keyword fusion, domain classification, and dual-domain balanced reranking
* `test_verifier.py`: Citation regex parsing, atomic claim segmentation, chunk sentence splitting, and real-model NLI inference

---

## Roadmap

- [x] **Phase 1**: Core RAG Pipeline (ChromaDB, Cross-Encoder, Gradio UI, arXiv corpus ingestion)
- [x] **Phase 2**: Hybrid Search Implementation (BM25 Sparse + Dense Vectors + RRF)
- [x] **Phase 3**: Auditable & Verifiable Citations (NLI Entailment, Evidence Quotes & Dual-Domain Balancing)
- [ ] **Phase 4**: Automated RAG Evaluation Suite (DeepEval / Ragas with CI/CD quality gates)
- [ ] **Phase 5**: Production Dockerfile, Hugging Face Spaces & Automated GitHub Actions CI/CD

---

## Setup & Running the Application

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/ayhamo/governance-rag.git
cd governance-rag

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies (PyTorch CPU / CUDA)
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Get a free key at [console.groq.com](https://console.groq.com))*

### 3. Run the Application
Start the Gradio web interface:
```bash
python app.py
```

* The app will initialize the embedding model, cross-encoder, ChromaDB vector store, and the DeBERTa-v3 NLI citation verifier.
* Open your browser and navigate to **`http://127.0.0.1:7860`**.
* Ask any compliance question (e.g., *"How do we mitigate bias to comply with Article 10 of the EU AI Act?"*) to see real-time streaming, dual-domain sources, and the automated NLI Citation Audit in action!
