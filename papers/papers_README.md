# Papers Corpus (GovernanceRAG)

This folder contains the 20 PDF papers used to build the RAG corpus. 
Because of GitHub file size limits and potential copyright restrictions, the raw PDFs are **not** committed to this repository.

### How to download the papers:
We have provided an automated script that uses the arXiv API to fetch exactly the 20 curated papers needed for this project.

1. Ensure you are in the root directory of the project.
2. Run the downloader script:
   ```bash
   python scripts/download_papers.py
   ```
3. The script will automatically fetch the PDFs and place them in this folder.
4. Run `python src/ingest.py` or start the app with `streamlit run app.py` to embed the new PDFs into ChromaDB!

---

## Corpus — 20 Papers on AI Regulation & Engineering

The corpus is curated to bridge the gap between Legal Regulation (e.g., EU AI Act) and Technical ML Engineering (e.g., Bias Mitigation Algorithms).

1. Are Bias Mitigation Techniques for Deep Learning Effective?
2. Operationalizing the EU AI Act in Agile Software Development: A Guideline-Based Approach
3. From Obligation to Specification: A Survey on Validating EU AI Act Requirements in RE
4. Mapping the Regulatory Learning Space for the EU AI Act
5. AI Governance in the Context of the EU AI Act: A Bibliometric and Literature Review Approach
6. Red Teaming AI Policy: A Taxonomy of Avoision and the EU AI Act
7. Navigating the EU AI Act: A Methodological Approach to Compliance for Safety-critical Products
8. ADAPT Centre Contribution on Implementation of the EU AI Act and Fundamental Right Protection
9. From Bias to Accountability: How the EU AI Act Confronts Challenges in European GeoAI Auditing
10. Complying with the EU AI Act
11. The EU AI Act in Development Practice: A Pro-justice Approach
12. Qualifying and Quantifying Risk Under the EU AI Act
13. The Case for ESM3 as a General-Purpose AI Model with Systemic Risk Under the EU AI Act
14. An Analysis of the New EU AI Act and A Proposed Standardization Framework for Machine Learning Fairness
15. The EU AI Act and the Rights-based Approach to Technological Governance
16. Assessing Model-Agnostic XAI Methods against EU AI Act Explainability Requirements
17. Complying with the EU AI Act: Innovations in Explainable and User-Centric Hand Gesture Recognition
18. Sustainable AI Regulation
19. Governing What the EU AI Act Excludes: Accountability for Autonomous AI Agents in Smart City Critical Infrastructure
20. Equality of Opportunity in Supervised Learning (The foundational Equalized Odds paper)

---

## Topics Covered

- High-Risk AI Classification under the EU AI Act (Annex III)
- Article 10 Compliance (Data Governance and Fairness Constraints)
- Explainable AI (XAI) mappings to Regulatory Demands (SHAP, LIME)
- Bias Mitigation Algorithms (Pre-processing, In-processing, Post-processing)
- Systemic Risk and General Purpose AI Models (GPAI)
- Automated red-teaming and compliance auditing protocols
