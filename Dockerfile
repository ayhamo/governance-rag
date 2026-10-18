# CPU image for the Gradio UI. The index is NOT built here yet: mount or copy
# a prebuilt chroma_db/ into /app/chroma_db, or run `governance-rag fetch` and
# `governance-rag ingest` inside the container first. Baking the index into
# the image is a roadmap item (implementation_plan.md, phase 6).

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/opt/hf
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY app.py ./
COPY corpus/manifest.yaml corpus/manifest.yaml
COPY corpus/raw/eu_ai_act corpus/raw/eu_ai_act
RUN uv sync --frozen --no-dev

# Pre-download the three local models so first start does not hit the network.
RUN /opt/venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
CrossEncoder('cross-encoder/nli-deberta-v3-small')"

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["/opt/venv/bin/python", "app.py"]
