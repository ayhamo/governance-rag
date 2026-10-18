"""
generator.py
------------
Builds the prompt from reranked chunks and streams a cited answer from the LLM.
"""

from __future__ import annotations

from groq import Groq

from governance_rag.config import GROQ_API_KEY, GROQ_MODEL

MAX_TOKENS = 1536  # 1024 cut structured answers off mid "Key takeaway"


def source_label(chunk: dict) -> str:
    """'EU AI Act, Art. 10(2)' for law chunks; the paper title for research."""
    if chunk.get("domain") == "law":
        return f"{chunk['meta'].get('short', 'Law')}, {chunk.get('citation_label', '')}"
    return chunk.get("title", "")


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        domain = c.get("domain", "")
        parts.append(f"[Source {i}: {source_label(c)} | {domain}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """You are GovernanceRAG, a compliance copilot that maps EU AI Act obligations to machine-learning engineering practice.
Your context contains two kinds of sources, labelled by domain:
- law: the text of the regulation itself (articles, annexes, recitals)
- research: technical ML papers on fairness, explainability and compliance engineering

ANSWER STRUCTURE
- Separate the Legal Obligation (cite law sources) from the Technical Mitigation (cite research sources).
- When a law source is an article or annex, name it in prose ("Article 10(2)(f)") as well as citing it.
- Use bullet points and bold for readability. Professional tone for an engineering audience.
- Give concrete steps or metrics when the research sources contain them (e.g. Equalized Odds, SHAP).

CITATION RULES
- Cite with standard ASCII brackets: [Source N]. Never [Source N: text], never Unicode brackets.
- Put [Source N] at the END of the sentence it supports. Never on its own line, never as a
  "Citation:" field, never as prose like "Source 3 describes". Every bullet that states a fact
  ends with its [Source N].
- Use ONLY the provided context. If the context does not answer the question, say so explicitly.
- Never invent citations."""


def build_user_prompt(question: str, context: str) -> str:
    return f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer with inline citations:"


def _client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env)")
    return Groq(api_key=GROQ_API_KEY)


def generate(question: str, chunks: list[dict]) -> str:
    r = _client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, build_context(chunks))},
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
    )
    return (r.choices[0].message.content or "").replace("【", "[").replace("】", "]")


def generate_streaming(question: str, chunks: list[dict]):
    stream = _client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, build_context(chunks))},
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for part in stream:
        token = part.choices[0].delta.content
        if token:
            yield token.replace("【", "[").replace("】", "]")


def generate_with_audit(question: str, chunks: list[dict], verifier=None):
    """Generate, then audit citations. Returns (answer, audit or None)."""
    answer = generate(question, chunks)
    if verifier is None:
        from governance_rag.config import ENABLE_CITATION_VERIFICATION

        if ENABLE_CITATION_VERIFICATION:
            from governance_rag.verifier import CitationVerifier

            verifier = CitationVerifier()
    return answer, (verifier.verify_answer(answer, chunks) if verifier else None)
