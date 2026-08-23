# GovernanceRAG: Bridging AI Law and Engineering
### A production-grade RAG pipeline mapping the EU AI Act & NIST frameworks directly to technical mitigation algorithms.

> **Live demo:** TBA

---

## The Problem

Enterprise AI teams are facing an unprecedented regulatory cliff. Non-compliance with frameworks like the **EU AI Act** carries catastrophic fines (up to €35M or 7% of global revenue), while the **NIST AI Risk Management Framework** is becoming the de-facto standard for enterprise procurement. 

However, there is a massive translation gap in the industry:
* **Legal and Compliance teams** read 200-page regulations but do not know how to write code to perform bias checks or feature attribution.
* **Machine Learning Engineers** know how to code, but do not know which specific mathematical fairness metric (e.g., Equalized Odds vs. Disparate Impact) legally satisfies "Article 10" or what exact logs are required to prove compliance during an audit.

Existing "AI Compliance" chatbots only quote the law (e.g., *"You must mitigate bias"*), leaving engineers to guess how to implement it.

---

## The Solution

**GovernanceRAG** is a specialized Retrieval-Augmented Generation (RAG) system built to serve as a translation layer. It queries across two distinct knowledge bases simultaneously to map legal requirements directly to technical implementations.

*Note: The current corpus focuses heavily on the EU AI Act and ML Fairness. We plan to expand this to include more global laws and policies (e.g., CPPA, GDPR, US State AI laws).*

**How it works:**
You describe your AI system (e.g., *"We are deploying an LLM for CV screening"*). The system:
1. Classifies the legal risk tier (e.g., *High-Risk under Annex III*).
2. Identifies mandatory legal checks.
3. **Retrieves and recommends the specific technical algorithms** from academic papers required to pass an audit.
4. Provides verifiable, strict inline citations linking back to both the legal article and the academic paper.

---

## Architecture

```
User: "We are deploying an LLM for CV screening. What are our requirements?"
                                      │
                         ┌────────────┴────────────┐
                         │      Hybrid Search      │
                         │   (BM25 + ChromaDB)     │ <-- (TBA)
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
                            (Diversity Enforced)
                                      │
                           LLM (Qwen via Groq API)
                                      │
Answer:
- Legal Classification: High-Risk (EU AI Act Article 6 / Annex III).
- Mandatory Checks: Bias testing on demographic groups (Article 10).
- Recommended Technical Algorithm: Equalized Odds Post-Processing [Source 1].
- Explainability Requirements: SHAP/LIME feature attribution logs [Source 2].
```

**Key Innovations:**
* **Two-Stage Retrieval:** Fast approximate search narrows chunks; a cross-encoder scores each (query, chunk) pair for true relevance.
* **Hybrid Search (BM25 + Semantic):** Dense vectors capture semantic intent, while sparse BM25 guarantees we don't miss exact regulatory acronyms. *(TBA)*
* **Auditable Citations:** LLM outputs are post-verified to ensure claims strictly match the retrieved chunk, preventing legal hallucination. *(TBA)*
* **Automated Evaluation:** Scored via Ragas/DeepEval for Faithfulness and Context Precision. *(TBA)*

---

## Stack (Still being updated)

| Component | Tool | Status |
|---|---|---|
| PDF extraction | pymupdf4llm | Active |
| Chunking | LangChain RecursiveCharacterTextSplitter | Active |
| Embeddings | `all-MiniLM-L6-v2` | Active |
| Vector store | ChromaDB (persistent) | Active |
| Sparse Index | BM25 | *TBA* |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Active |
| LLM | Qwen (via Groq) | Active |
| UI | Streamlit | Active |
| Evaluation | Ragas / DeepEval | *TBA* |

---

## Corpus

The vector database is built on a highly curated collection of **20 technical and regulatory research papers** sourced directly from arXiv. These papers specifically bridge the gap between EU AI Act regulations and Machine Learning bias mitigation algorithms (e.g., *Operationalizing the EU AI Act in Agile Software Development*, *Equality of Opportunity in Supervised Learning*).

Note: There is a plan to add more.

> **Automated Ingestion:** The repository includes an automated downloader script (`scripts/download_papers.py`) that uses the arXiv API to fetch the exact PDFs needed to build the database locally.

---

## Example Output

> **Note:** The system is heavily prompted to structure its outputs for enterprise engineering teams, separating the Legal Obligation from the Technical Mitigation.

**Q: How do we mitigate bias to comply with Article 10 of the EU AI Act?**

```
Retrieving legal & technical sources...
```

> **Legal Obligation (Article 10)**
> Under the EU AI Act, providers of high-risk AI systems must ensure that training, validation, and testing datasets are relevant, representative, and free of errors to prevent discriminatory impacts [Source 3]. The regulation specifically mandates that datasets must be examined for "possible biases" and that appropriate mitigation measures must be implemented [Source 1].
>
> **Technical Mitigation**
> To satisfy these legal requirements, engineering teams should implement a multi-stage approach:
> 
> 1. **Pre-processing (Data Level):**
>    Before training, audit the dataset for demographic parity and label noise. Techniques such as reweighing or oversampling can be applied to balance representation across protected classes [Source 1].
> 
> 2. **In-processing (Model Level):**
>    During training, incorporate fairness constraints directly into the loss function. For deep learning models, adversarial debiasing or applying regularization penalties can enforce fairness metrics like Equalized Odds without significantly degrading overall accuracy [Source 5].
> 
> 3. **Post-processing (Output Level):**
>    If the model cannot be retrained, apply threshold adjustments to the output probabilities. Calibrating decision thresholds separately for different demographic groups can ensure disparate impact is minimized [Source 2].

---

## Key Design Decisions

*(TBA - Architectural decisions regarding Hybrid Search and Verifiable Citations will be documented here as they are implemented).*

---

## Known Limitations

- **Fixed-size chunking** can cut mid-sentence on long complex sentences. Semantic Chunking is planned.
- **Table Extraction** from regulatory PDFs can occasionally lose formatting, requiring LLM inference to reconstruct relationships.

---

## Roadmap (still being worked on)

- [x] Core RAG Pipeline (ChromaDB, Cross-Encoder, Streamlit UI)
- [ ] Hybrid Search Implementation (BM25 Sparse + Dense Vectors)
- [ ] Auditable / Verifiable Citations (Anti-hallucination guardrails)
- [ ] Automated RAG Evaluation Suite (Ragas / DeepEval)
- [ ] Docker containerization
- [ ] Deploy on Hugging Face Spaces
- [ ] GitHub Actions CI/CD

---

## Setup

**1. Clone and install dependencies:**
```bash
git clone https://github.com/ayhamo/governance-rag.git
cd responsible-ai-rag
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
pip install -r requirements.txt
```

**2. Set your API key:**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_key_here
```
*(Get a free key at [console.groq.com](https://console.groq.com))*

**3. Download the Corpus:**
Run the automated script to fetch the 20 curated regulatory/ML papers from arXiv:
```bash
python scripts/download_papers.py
```

**4. Run the Application:**
Launch the Streamlit UI. On the first run, the system will automatically parse the PDFs, chunk the text, and build the ChromaDB vector store.
```bash
streamlit run app.py
```
