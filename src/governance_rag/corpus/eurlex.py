"""
eurlex.py
---------
Chunks an EUR-Lex Official Journal HTML export of a regulation into legal
units: one chunk per recital, per article paragraph (with its points), and
per annex item. Each chunk carries a structural context header
("Article 10 (Data and data governance), paragraph 2") and metadata that
lets a citation say "Art. 10(2)" and link to the EUR-Lex anchor.

Written against the 2024 CONVEX/ELI HTML layout used for
Regulation (EU) 2024/1689. Element ids follow the ELI scheme:
    rct_N          recital
    cpt_X          chapter          cpt_X.sct_N   section
    art_N          article          NNN.PPP       paragraph P of article N
    anx_X          annex
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from governance_rag.config import LAW_MAX_CHUNK_CHARS

# EUR-Lex serves XHTML; the HTML parser handles it fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── application dates (Article 113 of the Act) ───────────────
# Which parts of the Act apply from when. Used to stamp `applies_from` on
# every chunk so an answer can say whether an obligation is live yet.
GENERAL_APPLICATION = date(2026, 8, 2)
_APPLIES_FROM_RULES: list[tuple[str, date]] = [
    # (description, date) — matched by chapter / section / article below
    ("chapters I and II", date(2025, 2, 2)),
    ("chapter III section 4, chapter V, chapter VII, chapter XII, article 78", date(2025, 8, 2)),
    ("article 6(1) and corresponding obligations", date(2027, 8, 2)),
]


def applies_from(chapter: str | None, section: str | None, article: int | None) -> str:
    """Return the ISO date from which a unit applies, per Article 113."""
    if article == 101:  # penalties for GPAI providers: general date
        return GENERAL_APPLICATION.isoformat()
    if chapter in {"I", "II"}:
        return date(2025, 2, 2).isoformat()
    if (chapter == "III" and section == "4") or chapter in {"V", "VII", "XII"} or article == 78:
        return date(2025, 8, 2).isoformat()
    return GENERAL_APPLICATION.isoformat()


# ── data model ───────────────────────────────────────────────
@dataclass
class LawChunk:
    chunk_id: str
    text: str  # header + body, what gets embedded
    body: str  # legal text only
    header: str  # structural context line
    unit_type: str  # recital | article | annex
    citation_label: str  # "Art. 10(2)", "Annex IV(2)", "Recital 12"
    anchor: str  # EUR-Lex fragment, e.g. "art_10"
    meta: dict = field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────
_WS = re.compile(r"[ \t  ]+")


def _clean(s: str) -> str:
    s = s.replace(" ", " ").replace(" ", " ")
    s = _WS.sub(" ", s)
    return s.strip()


def _text_of(el: Tag) -> str:
    """Flatten an element to text, keeping point markers like (a) inline."""
    parts: list[str] = []
    for p in el.find_all("p"):
        t = _clean(p.get_text(" "))
        if t:
            parts.append(t)
    return _merge_markers(parts)


def _merge_markers(parts: list[str]) -> str:
    """EUR-Lex puts '(a)' and its text in separate <p>s; join them."""
    out: list[str] = []
    marker = re.compile(r"^(\(\w{1,4}\)|\d{1,3}\.|[ivx]{1,5}\.|—|-)$")
    i = 0
    while i < len(parts):
        if marker.match(parts[i]) and i + 1 < len(parts):
            out.append(f"{parts[i]} {parts[i + 1]}")
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return "\n".join(out)


def _direct_ps(el: Tag) -> list[str]:
    return [_clean(p.get_text(" ")) for p in el.find_all("p", recursive=False)]


def _split_points(body: str, max_chars: int) -> list[str]:
    """Split an over-long body on its point markers, keeping the lead-in."""
    if len(body) <= max_chars:
        return [body]
    lines = body.split("\n")
    lead = lines[0]
    pieces: list[str] = []
    cur = lead
    for line in lines[1:]:
        if len(cur) + len(line) + 1 > max_chars and cur != lead:
            pieces.append(cur)
            cur = lead
        cur += "\n" + line
    if cur != lead:
        pieces.append(cur)
    return pieces or [body]


# ── parser ───────────────────────────────────────────────────
def parse_eurlex_html(
    html_path: Path,
    source_id: str,
    instrument: str,
    short: str,
    link_base: str,
    version_date: str,
    max_chunk_chars: int = LAW_MAX_CHUNK_CHARS,
) -> list[LawChunk]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "lxml")
    base_meta = {
        "domain": "law",
        "source_id": source_id,
        "instrument": instrument,
        "short": short,
        "version_date": version_date,
    }
    chunks: list[LawChunk] = []

    def make(chunk_id, header, body, unit_type, label, anchor, **extra) -> LawChunk:
        meta = {**base_meta, **extra, "source_url": f"{link_base}#{anchor}"}
        return LawChunk(
            chunk_id=f"{source_id}:{chunk_id}",
            text=f"{header}\n{body}",
            body=body,
            header=header,
            unit_type=unit_type,
            citation_label=label,
            anchor=anchor,
            meta=meta,
        )

    # ── recitals ──
    for div in soup.select("div.eli-subdivision[id^=rct_]"):
        n = int(div["id"].split("_")[1])
        body = _text_of(div)
        header = f"{short}, Recital {n}"
        chunks.append(
            make(
                f"recital_{n}",
                header,
                body,
                "recital",
                f"Recital {n}",
                div["id"],
                recital=n,
                applies_from=date(2024, 8, 1).isoformat(),
            )
        )

    # ── articles ──
    for art in soup.select("div.eli-subdivision[id^=art_]"):
        n = int(art["id"].split("_")[1])
        title_el = art.select_one("div.eli-title p")
        title = _clean(title_el.get_text(" ")) if title_el else ""
        chapter, section = _chapter_section(art)
        live = applies_from(chapter, section, n)
        art_meta = dict(
            article=n,
            article_title=title,
            chapter=chapter or "",
            section=section or "",
            applies_from=live,
        )

        para_divs = [
            d
            for d in art.find_all("div", recursive=False)
            if re.match(r"^\d{3}\.\d{3}$", d.get("id", ""))
        ]
        if para_divs:
            for d in para_divs:
                p_no = int(d["id"].split(".")[1])
                body = _text_of(d)
                header = f"{short}, Article {n} ({title}), paragraph {p_no}"
                label = f"Art. {n}({p_no})"
                for i, piece in enumerate(_split_points(body, max_chunk_chars)):
                    suffix = f".{i + 1}" if i else ""
                    chunks.append(
                        make(
                            f"art_{n}.{p_no}{suffix}",
                            header,
                            piece,
                            "article",
                            label,
                            art["id"],
                            paragraph=p_no,
                            **art_meta,
                        )
                    )
        else:
            # Article with no numbered paragraphs (e.g. Art. 3 definitions):
            # all content apart from the title lines.
            body_parts = []
            for child in art.children:
                if not isinstance(child, Tag):
                    continue
                if "oj-ti-art" in child.get("class", []) or "eli-title" in child.get("class", []):
                    continue
                body_parts.append(
                    _text_of(child) if child.find("p") else _clean(child.get_text(" "))
                )
            body = "\n".join(p for p in body_parts if p)
            header = f"{short}, Article {n} ({title})"
            for i, piece in enumerate(_split_points(body, max_chunk_chars)):
                suffix = f".{i + 1}" if i else ""
                chunks.append(
                    make(
                        f"art_{n}{suffix}",
                        header,
                        piece,
                        "article",
                        f"Art. {n}",
                        art["id"],
                        **art_meta,
                    )
                )

    # ── annexes ──
    for anx in soup.find_all("div", id=re.compile(r"^anx_[IVXLC]+$")):
        roman = anx["id"].split("_")[1]
        titles = [_clean(p.get_text(" ")) for p in anx.find_all("p", class_="oj-doc-ti")]
        title = titles[1] if len(titles) > 1 else (titles[0] if titles else "")
        intro = " ".join(_direct_ps(anx))
        items = _top_level_items(anx)
        if not items:
            body = _text_of(anx)
            header = f"{short}, Annex {roman} ({title})"
            for i, piece in enumerate(_split_points(body, max_chunk_chars)):
                suffix = f".{i + 1}" if i else ""
                chunks.append(
                    make(
                        f"annex_{roman}{suffix}",
                        header,
                        piece,
                        "annex",
                        f"Annex {roman}",
                        anx["id"],
                        annex=roman,
                        annex_title=title,
                        applies_from=GENERAL_APPLICATION.isoformat(),
                    )
                )
            continue
        for section, section_title, item_no, body in items:
            # Annexes with lettered/numbered sections (I, VIII, XI) restart item numbering per section.
            sec_id = f".{section}" if section else ""
            sec_label = f", Section {section}" if section else ""
            header = f"{short}, Annex {roman} ({title}){sec_label}"
            if section_title:
                header += f" ({section_title})"
            header += f", point {item_no}"
            if intro:
                header += f" — {intro}"
            for i, piece in enumerate(_split_points(body, max_chunk_chars)):
                suffix = f".{i + 1}" if i else ""
                chunks.append(
                    make(
                        f"annex_{roman}{sec_id}.{item_no}{suffix}",
                        header,
                        piece,
                        "annex",
                        f"Annex {roman}{sec_label}, point {item_no}"
                        if section
                        else f"Annex {roman}({item_no})",
                        anx["id"],
                        annex=roman,
                        annex_title=title,
                        section=section or "",
                        item=item_no,
                        applies_from=GENERAL_APPLICATION.isoformat(),
                    )
                )

    return chunks


def _chapter_section(art: Tag) -> tuple[str | None, str | None]:
    chapter = section = None
    for parent in art.parents:
        pid = parent.get("id", "") if isinstance(parent, Tag) else ""
        m = re.match(r"^cpt_([IVXLC]+)(?:\.sct_(\d+))?$", pid)
        if m:
            chapter = chapter or m.group(1)
            if m.group(2):
                section = section or m.group(2)
    return chapter, section


def _top_level_items(anx: Tag) -> list[tuple[str, str, str, str]]:
    """
    Annex items are direct-child tables whose first cell is '1.', '2.', …
    Some annexes group them under 'Section A.' / 'Section 1' headings
    (p.oj-ti-grseq-1), after which numbering restarts; the section is tracked
    so chunk ids stay unique. Returns (section, section_title, item_no, body).
    """
    items: list[tuple[str, str, str, str]] = []
    section = section_title = ""
    pending_title = False
    for el in anx.find_all(recursive=False):
        if el.name == "p" and "oj-ti-grseq-1" in (el.get("class") or []):
            text = _clean(el.get_text(" "))
            m = re.match(r"^Section\s+([A-Z0-9]+)\.?\s*(.*)$", text)
            if m:
                section, section_title = m.group(1), m.group(2)
                pending_title = not section_title  # "Section 1" + title on the next <p>
            elif pending_title:
                section_title, pending_title = text, False
            continue
        if el.name != "table":
            continue
        cells = el.find_all("td")
        if len(cells) < 2:
            continue
        marker = _clean(cells[0].get_text(" "))
        m = re.match(r"^(\d{1,3})\.$", marker)
        if not m:
            continue
        items.append((section, section_title, m.group(1), f"{marker} {_text_of(cells[1])}"))
    return items
