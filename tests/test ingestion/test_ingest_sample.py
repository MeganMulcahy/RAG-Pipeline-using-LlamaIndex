"""
Sample ingest inspection — show what the pipeline actually extracted per page.

Picks up to:
  - 1 native_text page
  - 3 table_heavy pages
  - 3 image_dominant pages (Tesseract / unstructured)

Usage:
  python test_ingest_sample.py "test blob file2.pdf"
  python test_ingest_sample.py "test blob file2.pdf" --max-chars 2500
  python test_ingest_sample.py "test blob file2.pdf" --full -o ingest_sample_report.txt
"""

import argparse
import io
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import fitz
from dotenv import load_dotenv

load_dotenv(override=True)

from rag_pipeline import classify_pages_only, extract_page_content

SEP = "=" * 72


def _pick_samples(pages):
    buckets = {"native_text": [], "table_heavy": [], "image_dominant": []}
    for p in pages:
        if p.content_type in buckets:
            buckets[p.content_type].append(p)

    samples = []
    if buckets["native_text"]:
        samples.append(("TEXT (native)", buckets["native_text"][:1]))
    if buckets["table_heavy"]:
        samples.append(("TABLE (table_heavy)", buckets["table_heavy"][:3]))
    if buckets["image_dominant"]:
        samples.append(("HARD / SCANNED (image_dominant)", buckets["image_dominant"][:3]))
    return samples


def _clip(s: str, max_chars: int | None) -> str:
    if max_chars is None or len(s) <= max_chars:
        return s
    return s[:max_chars]


def _print_page_report(label: str, data: dict, max_chars: int | None) -> None:
    p1 = data["page_num"] + 1
    print(SEP)
    print(f"{label}  |  PDF page {p1} (0-based index {data['page_num']})")
    print(SEP)
    print(f"  content_type:    {data['content_type']}")
    print(f"  fitz char_count: {data['char_count_fitz']}")
    print(f"  extractor:       {data['extractor']}")
    print(f"  extracted chars: {len(data['text'] or '')}")
    print()

    fitz_raw = (data.get("fitz_raw") or "").strip()
    if fitz_raw:
        body = _clip(fitz_raw, max_chars)
        suffix = "" if max_chars is None else f" ({len(body)} of {len(fitz_raw)} chars)"
        print(f"--- fitz raw text{suffix} ---")
        print(body)
        if max_chars is not None and len(fitz_raw) > max_chars:
            print(f"\n... [{len(fitz_raw) - max_chars} more chars in fitz layer]")
        print()

    text = (data.get("text") or "").strip()
    if text:
        body = _clip(text, max_chars)
        suffix = "" if max_chars is None else f" ({len(body)} of {len(text)} chars)"
        print(f"--- extracted text used for RAG{suffix} ---")
        print(body)
        if max_chars is not None and len(text) > max_chars:
            print(f"\n... [{len(text) - max_chars} more chars]")
        print()
    else:
        print("--- extracted text: (empty) ---\n")

    html = data.get("table_html")
    if html:
        body = _clip(html, max_chars)
        suffix = "" if max_chars is None else f" ({len(body)} of {len(html)} chars)"
        print(f"--- table HTML (structured){suffix} ---")
        print(body)
        if max_chars is not None and len(html) > max_chars:
            print(f"\n... [{len(html) - max_chars} more chars]")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ingest output for sample pages")
    parser.add_argument("pdf", nargs="?", default="test blob file2.pdf", help="PDF path")
    parser.add_argument("--max-chars", type=int, default=2000, help="Max chars per section (ignored with --full)")
    parser.add_argument("--full", action="store_true", help="Print complete text with no truncation")
    parser.add_argument("-o", "--output", help="Write report to this file (UTF-8)")
    args = parser.parse_args()

    max_chars: int | None = None if args.full else args.max_chars

    if args.output:
        sys.stdout = open(args.output, "w", encoding="utf-8", errors="replace")

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\nPDF: {pdf_path.name}")
    print("Pass 1: classifying all pages (fitz only)…\n")

    path, pages = classify_pages_only(str(pdf_path))
    counts = {}
    for p in pages:
        counts[p.content_type] = counts.get(p.content_type, 0) + 1
    print("Page counts by content_type:")
    for ct, n in sorted(counts.items()):
        print(f"  {ct}: {n}")

    samples = _pick_samples(pages)
    if not samples:
        print("\nNo pages classified — check PDF path.")
        sys.exit(1)

    print(f"\nPass 2: extracting {sum(len(s[1]) for s in samples)} sample page(s)…\n")

    doc = fitz.open(path)
    try:
        for label, page_list in samples:
            for pinfo in page_list:
                data = extract_page_content(path, doc, pinfo)
                _print_page_report(label, data, max_chars)
    finally:
        doc.close()

    print(SEP)
    print("Done.")


if __name__ == "__main__":
    main()
