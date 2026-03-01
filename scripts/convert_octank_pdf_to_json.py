#!/usr/bin/env python3
"""Convert OctankFinancial 10-K PDF to JSON articles for indexing.

Run from project root:
    .venv/bin/python scripts/convert_octank_pdf_to_json.py
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pypdf

PDF_PATH = Path("data/raw/octank/octank_financial_10K.pdf")
OUT_DIR  = Path("data/raw/octank")

# Matches standard 10-K section headers: "ITEM 1.", "Item 1A.", "ITEM 7A. MD&A"
SECTION_RE = re.compile(r"^(item\s+\d+[a-z]?\b.{0,80})", re.IGNORECASE | re.MULTILINE)


def extract_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return (header, body) pairs; single section fallback if no headers found."""
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("OctankFinancial 10-K", text)]

    sections = []
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start  = m.end()
        end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body   = text[start:end].strip()
        if body:
            sections.append((header, body))
    return sections


def slugify(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")[:60]


def write_sections(sections: list[tuple[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Idempotent: remove stale JSON files before writing fresh ones
    for old in out_dir.glob("*.json"):
        old.unlink()

    for idx, (header, body) in enumerate(sections):
        slug = f"{idx:02d}_{slugify(header)}"
        article = {
            "title": f"OctankFinancial 10-K — {header}",
            "pageid": f"octank_{slug}",
            "url": f"file://{PDF_PATH}",
            "content": body,
            "summary": "",
            "categories": ["10-K", "OctankFinancial", "Annual Report"],
            "links": [],
            "word_count": len(body.split()),
            "fetched_at": now,
        }
        out_path = out_dir / f"{slug}.json"
        out_path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{idx+1}/{len(sections)}] {out_path.name}  ({article['word_count']} words)")


def main() -> None:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting text from {PDF_PATH} ...")
    text = extract_text(PDF_PATH)
    print(f"  Total chars: {len(text):,}")

    sections = split_into_sections(text)
    print(f"  Detected {len(sections)} section(s)")

    write_sections(sections, OUT_DIR)
    print(f"\nDone. {len(sections)} JSON files written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
