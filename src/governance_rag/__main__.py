"""
Command line entry point.

    governance-rag fetch              download missing sources listed in the manifest
    governance-rag ingest [--rebuild] [--llm-headers]
                                      chunk, embed and index every enabled source
    governance-rag serve              start the Gradio UI
    governance-rag api                start the FastAPI server
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="governance-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download missing sources from corpus/manifest.yaml")

    p_ing = sub.add_parser("ingest", help="build or update the index")
    p_ing.add_argument("--rebuild", action="store_true", help="drop the collection first")
    p_ing.add_argument(
        "--llm-headers",
        action="store_true",
        help="generate LLM context headers for research chunks (needs GROQ_API_KEY; cached)",
    )

    p_ing.add_argument(
        "--bm25-only", action="store_true", help="rebuild only the BM25 index from ChromaDB"
    )

    sub.add_parser("serve", help="run the Gradio UI")
    sub.add_parser("api", help="run the FastAPI server")

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        from governance_rag.corpus.fetch import fetch_all

        fetch_all()
    elif args.cmd == "ingest":
        from governance_rag.ingest import run_ingestion

        run_ingestion(rebuild=args.rebuild, llm_headers=args.llm_headers, bm25_only=args.bm25_only)
    elif args.cmd == "serve":
        import runpy

        runpy.run_path("app.py", run_name="__main__")
    elif args.cmd == "api":
        import uvicorn

        uvicorn.run("governance_rag.api:app", host="0.0.0.0", port=8000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
