"""
manifest.py
-----------
Reads corpus/manifest.yaml: the single list of what the index is built from,
with domain, licence, URLs and version dates per source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from governance_rag.config import CORPUS_DIR, MANIFEST_PATH


@dataclass
class Source:
    id: str
    domain: str
    kind: str
    path: Path
    enabled: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def short(self) -> str:
        return self.meta.get("short") or self.meta.get("instrument") or self.id


@dataclass
class Manifest:
    as_of: str
    sources: dict[str, Source]

    def enabled(self) -> list[Source]:
        return [s for s in self.sources.values() if s.enabled]

    def domain_of(self, source_id: str) -> str | None:
        src = self.sources.get(source_id)
        return src.domain if src else None


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    sources: dict[str, Source] = {}
    for sid, cfg in raw["sources"].items():
        cfg = dict(cfg)
        sources[sid] = Source(
            id=sid,
            domain=cfg.pop("domain"),
            kind=cfg.pop("kind"),
            path=CORPUS_DIR / cfg.pop("path"),
            enabled=bool(cfg.pop("enabled", True)),
            meta=cfg,
        )
    return Manifest(as_of=str(raw["as_of"]), sources=sources)
