"""
generator.py
------------
Sends retrieved chunks as context to the LLM and generates
a grounded answer with inline citations.

Supports both regular and streaming generation.
"""

from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL


# ── context builder ──────────────────────────────────────────
def build_context(chunks):
    """Format reranked chunks into a numbered context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['title']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ── prompts ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the AI Compliance Copilot, bridging the gap between Legal Regulation (EU AI Act) and Machine Learning Engineering.
You have been given context from a curated set of official AI regulations and technical ML research papers.
Your goal is to provide precise, structured, and strictly cited answers that map legal requirements to technical algorithms.

ANSWER QUALITY:
- When asked about a requirement, clearly separate the Legal Obligation from the Technical Mitigation.
- Use structured formats (bullet points, bold text) for readability.
- Connect ideas from different papers (e.g., matching a legal article to a fairness paper).
- Use a highly professional, authoritative tone suitable for enterprise ML teams.
- If asked about "how", provide concrete ML steps or mathematical metrics mentioned in the text (e.g., Equalized Odds, SHAP).

CITATION RULES:
- Always cite using [Source N] notation inline.
- Citation format is strictly [Source N] — never [Source N: anything].
- Every single claim MUST end with a citation to the specific source chunk it came from.
- [Source N] is the complete citation format — nothing else goes inside the brackets.

STRICT LIMITS:
- Use ONLY the provided context — never external knowledge.
- Never fabricate or hallucinate citations.
- If the sources do not contain the answer, explicitly state that the compliance corpus does not cover it."""

def build_user_prompt(question, context):
    return f"""Context from research papers:

{context}

Question: {question}

Answer with inline citations:"""


# ── standard generation ──────────────────────────────────────
def generate(question, chunks):
    """
    Generate a grounded answer from reranked chunks.
    Returns the full answer string.
    """
    context  = build_context(chunks)
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(question, context)},
        ],
        temperature = 0.1,
        max_tokens  = 1024,
    )
    return response.choices[0].message.content


# ── streaming generation ─────────────────────────────────────
def generate_streaming(question, chunks):
    """
    Generator function for streaming responses.
    Yields tokens one by one as they arrive from the API.
    Use in Streamlit with: for token in generate_streaming(...): ...
    """
    context = build_context(chunks)
    client  = Groq(api_key=GROQ_API_KEY)
    stream  = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(question, context)},
        ],
        temperature = 0.1,
        max_tokens  = 1024,
        stream      = True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
