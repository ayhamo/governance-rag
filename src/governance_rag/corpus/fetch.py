"""
fetch.py
--------
Downloads any source listed in corpus/manifest.yaml that is missing on disk.
Safe to rerun: existing files are skipped.
"""

from __future__ import annotations

import httpx

from governance_rag.corpus.manifest import Manifest, Source, load_manifest

ARXIV_PDF = "https://arxiv.org/pdf/{id}"


def _download(url: str, dest, client: httpx.Client) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def fetch_source(src: Source, client: httpx.Client) -> list[str]:
    """Returns a list of human-readable lines describing what happened."""
    lines = []
    if src.kind == "arxiv_pdf_dir":
        for paper in src.meta.get("papers", []):
            dest = src.path / paper["file"]
            if dest.exists():
                lines.append(f"  skip  {paper['file']}")
                continue
            _download(ARXIV_PDF.format(id=paper["id"]), dest, client)
            lines.append(f"  get   {paper['file']}")
        return lines

    if src.path.exists():
        return [f"  skip  {src.path.name}"]
    url = src.meta.get("url")
    if not url:
        return [f"  !!    {src.id}: no url in manifest and file missing"]
    _download(url, src.path, client)
    return [f"  get   {src.path.name}"]


def fetch_all(manifest: Manifest | None = None, include_disabled: bool = False) -> None:
    manifest = manifest or load_manifest()
    with httpx.Client(headers={"User-Agent": "governance-rag/0.2"}) as client:
        for src in manifest.sources.values():
            if not src.enabled and not include_disabled:
                continue
            print(f"{src.id} ({src.domain})")
            for line in fetch_source(src, client):
                print(line)
