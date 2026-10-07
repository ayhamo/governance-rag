"""
verifier.py
-----------
Audits and verifies LLM-generated citations ([Source N]) against retrieved source chunks
using Natural Language Inference (NLI) with a cross-encoder model.

Classifies claims into:
- Entailment (Grounded / Verified)
- Neutral (Ambiguous / Unsupported)
- Contradiction (Conflicting / Hallucinated)

Isolates the most supportive sentence quote from each source chunk to provide
transparent, auditable compliance proofs.
"""

import re
import logging
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import CrossEncoder

from src.config import (
    NLI_MODEL,
    NLI_CONFIDENCE_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MODEL
)

logger = logging.getLogger(__name__)


# ── sentence & citation extraction ────────────────────────────
def extract_citations_from_text(text: str) -> List[int]:
    """
    Extract all referenced source numbers from a text fragment.
    Supports formats like [Source 1], [Source 1, 2], [Source 1][Source 2].
    """
    matches = re.findall(r'\[Source\s*([^\]]+)\]', text, flags=re.IGNORECASE)
    sources = set()
    for m in matches:
        digits = re.findall(r'\d+', m)
        sources.update(int(d) for d in digits)
    return sorted(list(sources))


def extract_claims_with_citations(answer: str) -> List[Dict[str, Any]]:
    """
    Parses a generated Markdown answer into atomic claim statements associated
    with their inline [Source N] citations.
    Handles:
    - Markdown table rows (| col | col | [Source N] |)
    - Abbreviations like Art., Recital., Sec., e.g., i.e., vs.
    - Citations placed before or after punctuation, narrow non-breaking spaces
    - Multiple citations per claim
    """
    if not answer or not answer.strip():
        return []

    lines = [line.strip() for line in answer.split('\n') if line.strip()]
    claims = []

    for line in lines:
        # Check if line contains citations
        if not re.search(r'\[Source\s*\d+', line, flags=re.IGNORECASE):
            continue

        # Handle Markdown table rows: preserve row context without splitting on internal periods
        if line.startswith('|') and line.endswith('|'):
            citations = extract_citations_from_text(line)
            if citations:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                text_cells = []
                for c in cells:
                    clean_c = re.sub(r'\[Source\s*[^\]]+\]', '', c, flags=re.IGNORECASE).strip()
                    if clean_c and clean_c != '---' and not re.match(r'^:?-+:?$', clean_c):
                        text_cells.append(clean_c)
                claim_text = " — ".join(text_cells).strip(' \u202f-*•>#\t\r\n.')
                if len(claim_text) >= 10:
                    claims.append({
                        "original_sentence": line,
                        "claim": claim_text,
                        "citations": citations,
                    })
            continue

        # Protect common legal & editorial abbreviations from sentence splitting
        protected_line = re.sub(r'\b(Art|art|Recital|Sec|Para|e\.g|i\.e|vs|No)\.\s*', r'\1_DOT_ ', line)

        # Split on sentence boundaries, keeping trailing citations attached
        raw_segments = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\"\'\-\*])', protected_line)

        merged_sentences = []
        for seg in raw_segments:
            seg_unprotected = seg.replace('_DOT_', '.')
            seg_stripped = seg_unprotected.strip()
            # If segment is just citation tags (e.g. "[Source 1]" or "\u202f[Source 1]"), merge with previous sentence
            if re.match(r'^[\u202f\s]*\[Source\s*[^\]]+\]', seg_stripped) and merged_sentences:
                merged_sentences[-1] += " " + seg_stripped
            else:
                merged_sentences.append(seg_unprotected)

        for s in merged_sentences:
            citations = extract_citations_from_text(s)
            if citations:
                # Strip citation brackets to obtain clean statement text
                clean_claim = re.sub(r'\[Source\s*[^\]]+\]', '', s, flags=re.IGNORECASE)
                clean_claim = clean_claim.strip(' \u202f-*•>#\t\r\n.')
                if len(clean_claim) >= 10:
                    claims.append({
                        "original_sentence": s.strip(),
                        "claim": clean_claim,
                        "citations": citations,
                    })

    return claims



def segment_chunk_sentences(chunk_text: str) -> List[str]:
    """
    Segments a retrieved document chunk into individual candidate premise sentences.
    Cleans Markdown headers and filters out ultra-short noise fragments.
    """
    if not chunk_text:
        return []

    # Clean Markdown headings and bold markers
    cleaned = re.sub(r'^#+\s*', '', chunk_text, flags=re.MULTILINE)
    cleaned = re.sub(r'\*\*', '', cleaned)
    
    # Split by paragraphs and sentence terminators
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', cleaned)
    
    valid_sentences = []
    for s in raw_sentences:
        s_clean = s.strip(' -*•#\t\r\n')
        # Skip very short fragments, formulas without words, or table remnants
        if len(s_clean) >= 25 and any(c.isalpha() for c in s_clean):
            valid_sentences.append(s_clean)

    return valid_sentences if valid_sentences else [chunk_text.strip()]


# ── citation verifier class ───────────────────────────────────
class CitationVerifier:
    """
    NLI-powered verifier that validates citation grounding in RAG responses.
    """

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        threshold: float = NLI_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        backend: str = "auto"
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CrossEncoder] = None
        self.label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}

        if self.backend in ("auto", "cross_encoder"):
            self._load_model()

    def _load_model(self):
        """Loads the CrossEncoder NLI model into memory."""
        try:
            logger.info(f"Loading NLI verifier model: {self.model_name} on {self.device}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            # Sync label mapping from model config if present
            if hasattr(self.model.model, "config") and hasattr(self.model.model.config, "id2label"):
                cfg_labels = self.model.model.config.id2label
                self.label_map = {int(k): v.lower() for k, v in cfg_labels.items()}
        except Exception as e:
            logger.warning(f"Could not load local CrossEncoder NLI model: {e}. Falling back to Groq API.")
            self.model = None

    def verify_citation(self, claim: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies a single claim against a retrieved source chunk.
        Uses a two-tier strategy: fast local NLI cross-encoder first, with an
        LLM-as-a-judge second opinion to resolve strict MNLI neutral ambiguities.
        """
        chunk_text = chunk.get("text", "")
        candidate_sentences = segment_chunk_sentences(chunk_text)

        if not candidate_sentences:
            return {
                "verdict": "neutral",
                "status": "unsupported",
                "confidence": 0.0,
                "best_quote": "",
                "probabilities": {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
            }

        # If local model is loaded, compute inference
        if self.model is not None:
            res = self._verify_with_model(claim, candidate_sentences)
            # If directly verified or contradicted, return immediately
            if res["status"] in ("verified", "contradicted"):
                return res
            # If neutral (ambiguous due to strict MNLI vocabulary mismatch), query Groq judge
            if GROQ_API_KEY:
                groq_res = self._verify_with_groq_single(claim, chunk_text)
                if groq_res["status"] == "verified":
                    return groq_res
            return res

        # Fallback to Groq LLM-as-a-judge if local model is not loaded
        return self._verify_with_groq_single(claim, chunk_text)


    def _verify_with_model(self, claim: str, candidate_sentences: List[str]) -> Dict[str, Any]:
        """Runs batch inference across candidate sentences using CrossEncoder NLI."""
        # Premise is the retrieved sentence; hypothesis is the generated claim
        pairs = [(sent, claim) for sent in candidate_sentences]
        scores = self.model.predict(pairs)
        
        probs = torch.softmax(torch.tensor(scores), dim=-1).tolist()

        best_entail_prob = -1.0
        best_quote = candidate_sentences[0]
        best_probs = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}

        max_contra_prob = 0.0

        for sent, p in zip(candidate_sentences, probs):
            p_dict = {self.label_map[i]: float(p[i]) for i in range(len(p))}
            entail = p_dict.get("entailment", 0.0)
            contra = p_dict.get("contradiction", 0.0)

            if contra > max_contra_prob:
                max_contra_prob = contra

            if entail > best_entail_prob:
                best_entail_prob = entail
                best_quote = sent
                best_probs = p_dict

        # Determine verdict and status
        if best_entail_prob >= self.threshold:
            verdict = "entailment"
            status = "verified"
            confidence = round(best_entail_prob, 3)
        elif max_contra_prob >= 0.70 and max_contra_prob > best_entail_prob:
            verdict = "contradiction"
            status = "contradicted"
            confidence = round(max_contra_prob, 3)
        else:
            verdict = "neutral"
            status = "unsupported"
            confidence = round(best_probs.get("neutral", 0.0), 3)

        return {
            "verdict": verdict,
            "status": status,
            "confidence": confidence,
            "best_quote": best_quote,
            "probabilities": {k: round(v, 4) for k, v in best_probs.items()}
        }

    def _verify_with_groq_single(self, claim: str, chunk_text: str) -> Dict[str, Any]:
        """Fallback verification using Groq API as an LLM judge."""
        if not GROQ_API_KEY:
            return {
                "verdict": "neutral",
                "status": "unsupported",
                "confidence": 0.0,
                "best_quote": "",
                "probabilities": {}
            }

        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            prompt = f"""You are an AI Compliance Citation Auditor.
Determine whether the following Claim is substantiated by the Provided Evidence.

Evidence:
\"\"\"{chunk_text[:1200]}\"\"\"

Claim:
\"{claim}\"

Respond in valid JSON only with keys:
"verdict": "entailment" (if strictly supported), "neutral" (if not covered or vague), or "contradiction" (if conflicting),
"confidence": float between 0.0 and 1.0,
"best_quote": exact sentence from Evidence supporting the claim (or empty if not supported).
"""
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            data = json.loads(response.choices[0].message.content or "{}")
            verdict = data.get("verdict", "neutral").lower()
            confidence = float(data.get("confidence", 0.5))
            status = "verified" if (verdict == "entailment" and confidence >= self.threshold) else ("contradicted" if verdict == "contradiction" else "unsupported")
            
            return {
                "verdict": verdict,
                "status": status,
                "confidence": confidence,
                "best_quote": data.get("best_quote", ""),
                "probabilities": {verdict: confidence}
            }
        except Exception as e:
            logger.warning(f"Groq verification fallback error: {e}")
            return {
                "verdict": "neutral",
                "status": "unsupported",
                "confidence": 0.0,
                "best_quote": "",
                "probabilities": {}
            }

    def verify_answer(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audits all inline citations across the generated answer.
        Returns detailed claim audits and aggregate grounding metrics.
        """
        claims = extract_claims_with_citations(answer)
        claim_audits = []
        source_audits: Dict[int, List[Dict[str, Any]]] = {}

        total_citations = 0
        verified_count = 0
        unsupported_count = 0
        contradicted_count = 0

        for item in claims:
            claim_text = item["claim"]
            for src_idx in item["citations"]:
                total_citations += 1

                # Check if cited index is within retrieved bounds
                if src_idx < 1 or src_idx > len(chunks):
                    audit_res = {
                        "claim": claim_text,
                        "source_idx": src_idx,
                        "verdict": "invalid",
                        "status": "invalid_citation",
                        "confidence": 0.0,
                        "best_quote": "",
                        "paper_title": "Unknown Source"
                    }
                    unsupported_count += 1
                else:
                    chunk = chunks[src_idx - 1]
                    res = self.verify_citation(claim_text, chunk)
                    audit_res = {
                        "claim": claim_text,
                        "source_idx": src_idx,
                        "verdict": res["verdict"],
                        "status": res["status"],
                        "confidence": res["confidence"],
                        "best_quote": res["best_quote"],
                        "paper_title": chunk.get("title", f"Source {src_idx}")
                    }

                    if res["status"] == "verified":
                        verified_count += 1
                    elif res["status"] == "contradicted":
                        contradicted_count += 1
                    else:
                        unsupported_count += 1

                claim_audits.append(audit_res)
                source_audits.setdefault(src_idx, []).append(audit_res)

        grounding_rate = (verified_count / total_citations) if total_citations > 0 else 0.0

        # Build human-readable audit badge
        if total_citations == 0:
            badge = "⚠️ No Citations Found in Response"
            badge_type = "warning"
        elif contradicted_count > 0:
            badge = f"⛔ Contradiction Detected ({contradicted_count} claim conflicting with sources)"
            badge_type = "danger"
        elif unsupported_count > 0:
            badge = f"⚠️ Partial Grounding ({verified_count}/{total_citations} citations verified)"
            badge_type = "warning"
        else:
            badge = f"🛡️ Fully Grounded ({verified_count}/{total_citations} citations verified via NLI)"
            badge_type = "success"

        # Summarize per-source verification status for UI cards
        source_summary = {}
        for i, chunk in enumerate(chunks, 1):
            audits_for_src = source_audits.get(i, [])
            if not audits_for_src:
                source_summary[i] = {
                    "cited": False,
                    "status": "unreferenced",
                    "best_quote": "",
                    "confidence": 0.0
                }
            else:
                has_contra = any(a["status"] == "contradicted" for a in audits_for_src)
                all_verified = all(a["status"] == "verified" for a in audits_for_src)
                best_conf = max(a["confidence"] for a in audits_for_src)
                best_q = next((a["best_quote"] for a in audits_for_src if a["best_quote"]), "")

                status = "contradicted" if has_contra else ("verified" if all_verified else "partial")
                source_summary[i] = {
                    "cited": True,
                    "status": status,
                    "best_quote": best_q,
                    "confidence": best_conf
                }

        return {
            "total_citations": total_citations,
            "verified_count": verified_count,
            "unsupported_count": unsupported_count,
            "contradicted_count": contradicted_count,
            "grounding_rate": round(grounding_rate, 3),
            "badge": badge,
            "badge_type": badge_type,
            "claim_audits": claim_audits,
            "source_summary": source_summary
        }
