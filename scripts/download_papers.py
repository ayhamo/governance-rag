import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# 20 highly relevant papers bridging the EU AI Act, AI regulation, and technical ML mitigation
PAPERS = [
    "Are Bias Mitigation Techniques for Deep Learning Effective?",
    "Operationalizing the EU AI Act in Agile Software Development: A Guideline-Based Approach",
    "From Obligation to Specification: A Survey on Validating EU AI Act Requirements in RE",
    "Mapping the Regulatory Learning Space for the EU AI Act",
    "AI Governance in the Context of the EU AI Act: A Bibliometric and Literature Review Approach",
    "Red Teaming AI Policy: A Taxonomy of Avoision and the EU AI Act",
    "Navigating the EU AI Act: A Methodological Approach to Compliance for Safety-critical Products",
    "ADAPT Centre Contribution on Implementation of the EU AI Act and Fundamental Right Protection",
    "From Bias to Accountability: How the EU AI Act Confronts Challenges in European GeoAI Auditing",
    "Complying with the EU AI Act",
    "The EU AI Act in Development Practice: A Pro-justice Approach",
    "Qualifying and Quantifying Risk Under the EU AI Act",
    "The Case for ESM3 as a General-Purpose AI Model with Systemic Risk Under the EU AI Act",
    "An Analysis of the New EU AI Act and A Proposed Standardization Framework for Machine Learning Fairness",
    "The EU AI Act and the Rights-based Approach to Technological Governance",
    "Assessing Model-Agnostic XAI Methods against EU AI Act Explainability Requirements",
    "Complying with the EU AI Act: Innovations in Explainable and User-Centric Hand Gesture Recognition",
    "Sustainable AI Regulation",
    "Governing What the EU AI Act Excludes: Accountability for Autonomous AI Agents in Smart City Critical Infrastructure",
    "Equality of Opportunity in Supervised Learning" # The famous Equalized Odds paper!
]

def main():
    # Empty the papers directory if we want a fresh start
    os.makedirs("papers", exist_ok=True)
    print(f"Starting download of {len(PAPERS)} papers for the Compliance Copilot...")

    for i, title in enumerate(PAPERS, 1):
        print(f"\n[{i}/{len(PAPERS)}] Searching for: {title}")
        query = urllib.parse.quote(f'ti:"{title}"')
        url = f'http://export.arxiv.org/api/query?search_query={query}&max_results=1'

        try:
            req = urllib.request.urlopen(url)
            xml_data = req.read()
            root = ET.fromstring(xml_data)

            # Parse arXiv response
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            if entry is not None:
                id_url = entry.find('{http://www.w3.org/2005/Atom}id').text
                arxiv_id = id_url.split('/abs/')[-1]
                pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

                # Clean filename
                safe_title = "".join([c if c.isalnum() else "_" for c in title])
                safe_title = safe_title[:50] # Trim long names
                filename = os.path.join("papers", f"{safe_title}_{arxiv_id}.pdf")
                
                if os.path.exists(filename):
                    print("  --> Already exists, skipping.")
                    continue

                print(f"  --> Found on arXiv! Downloading {pdf_url}...")
                urllib.request.urlretrieve(pdf_url, filename)
                print("  --> Saved successfully.")
            else:
                print("  --> Not found via exact arXiv API query (may be on ACM/Springer or publisher page).")
        except Exception as e:
            print(f"  --> Error: {e}")

    print("\nDone! Check your papers/ folder.")
    print("Don't forget to delete the old irrelevant PDFs and run `python src/ingest.py` to rebuild ChromaDB!")

if __name__ == "__main__":
    main()

