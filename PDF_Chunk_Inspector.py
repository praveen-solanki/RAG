#!/usr/bin/env python3
"""
PDF_Chunk_Inspector.py
======================
Standalone debugging / inspection tool for the V3 RAG ingestion pipeline.

Purpose
-------
This script takes a single PDF file, runs the full V3 page-classification and
chunking pipeline on it, and writes a detailed human-readable report so you can
see exactly:

  • How the classifier (heuristic + optional Ollama vision) labels each page.
  • Which pages are skipped (TOC / boilerplate) and why.
  • How sections are detected inside each page's extracted text.
  • How many parent chunks and child chunks come from each page individually,
    including their character ranges and text previews.
  • Per-page-type statistics and a summary table at the end.
  • Anomaly detection: pages with extracted text but zero chunks.

This file does NOT write anything to Qdrant or modify any existing code.
It imports classes directly from Qdrant_Database_Generation_V3 and calls the
same methods the real ingestion pipeline uses, so what you see here is exactly
what would be stored in the vector database.

Design note on per-page vs global chunking
-------------------------------------------
The real ingestion pipeline calls detect_sections + chunk on the FULL concatenated
document text (so chunks can span page boundaries, which is intentional for better
recall).  This inspector runs the same chunker on each PAGE INDIVIDUALLY so that
the report can clearly show "page N produced X parent + Y child chunks".
The per-page numbers are therefore close to — but not identical to — the numbers
the real ingestion produces.  The final summary section reports the GLOBAL run
(same as real ingestion) for comparison.

Usage
-----
    python PDF_Chunk_Inspector.py --pdf path/to/document.pdf

Options
    --pdf               Path to the PDF to inspect  (required)
    --output            Path for the output report file
                        (default: <pdf-stem>_inspection.txt next to the PDF)
    --no-vision         Skip Ollama vision classifier; use heuristics only
    --ollama-url        Ollama base URL (default: http://localhost:11434)
    --vision-model      Ollama vision model for classification
    --vision-desc-model Ollama model for image/diagram description
    --top-k-preview N   Chars of chunk text to preview in the report (default: 300)
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Import all classes from the ingestion module (no modifications made there) ──
try:
    from Qdrant_Database_Generation_V3 import (
        # Data classes
        PageType,
        DocumentSection,
        ParentChunk,
        ChildChunk,
        # Processing classes
        OllamaVisionClassifier,
        ContentTypeExtractor,
        EnhancedPDFLoader,
        AdvancedDocumentLoader,
        SemanticChunker,
        ParentChildBuilder,
        # Config constants (used as sensible defaults)
        CHILD_CHUNK_SIZE,
        CHILD_CHUNK_OVERLAP,
        PARENT_CHUNK_SIZE,
        PARENT_CHUNK_OVERLAP,
        OLLAMA_URL               as DEFAULT_OLLAMA_URL,
        OLLAMA_VISION_MODEL      as DEFAULT_VISION_MODEL,
        OLLAMA_VISION_DESC_MODEL as DEFAULT_VISION_DESC_MODEL,
        ENABLE_PAGE_CLASSIFICATION,
    )
except ImportError as exc:
    sys.exit(
        f"ERROR: Could not import from Qdrant_Database_Generation_V3.py.\n"
        f"Make sure both files are in the same directory.\nDetail: {exc}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════

_PAGE_TYPE_EMOJI: Dict[str, str] = {
    PageType.TEXT:     "📝 text",
    PageType.TABLE:    "📊 table",
    PageType.IMAGE:    "🖼️  image",
    PageType.DIAGRAM:  "📐 diagram",
    PageType.EQUATION: "🔢 equation",
    PageType.MIXED:    "🔀 mixed",
    PageType.COVER:    "📄 cover",
    PageType.TOC:      "📋 toc",
    PageType.UNKNOWN:  "❓ unknown",
}

_W = 90  # report width


def _bar(label: str, value: int, total: int, width: int = 30) -> str:
    if total == 0:
        pct, filled = 0.0, 0
    else:
        pct = value / total
        filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:<15} [{bar}] {value:>4}  ({pct*100:5.1f} %)"


def _divider(char: str = "─", width: int = _W) -> str:
    return char * width


def _header(title: str, char: str = "═", width: int = _W) -> str:
    pad = max(0, width - len(title) - 4)
    left = pad // 2
    right = pad - left
    return f"{char * (left + 2)}  {title}  {char * (right + 2)}"


# ═══════════════════════════════════════════════════════════════════════════
# Parse per-page blocks from the concatenated text produced by EnhancedPDFLoader
# ═══════════════════════════════════════════════════════════════════════════

def _parse_page_blocks(full_text: str) -> List[Tuple[int, str, str]]:
    """
    Split the concatenated page text into (page_number, page_type, page_body) triples.
    EnhancedPDFLoader.load() embeds  \\n[Page N | type=<type>]\\n  before each page.
    """
    tag_re = re.compile(r'\n\[Page\s+(\d+)\s*\|\s*type=(\w+)\]')
    parts  = tag_re.split(full_text)
    # layout: [preamble, pn, pt, body, pn, pt, body, …]
    blocks: List[Tuple[int, str, str]] = []
    i = 1
    while i + 2 < len(parts):
        blocks.append((int(parts[i]), parts[i + 1], parts[i + 2]))
        i += 3
    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# Main inspection logic
# ═══════════════════════════════════════════════════════════════════════════

def inspect_pdf(
    pdf_path: str,
    output_path: str,
    use_vision: bool = True,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    vision_model: str = DEFAULT_VISION_MODEL,
    vision_desc_model: str = DEFAULT_VISION_DESC_MODEL,
    top_k_preview: int = 300,
) -> None:
    """
    Run the full V3 pipeline on *pdf_path* and write a detailed report to
    *output_path*.  Nothing is written to Qdrant.
    """
    filename = Path(pdf_path).name
    lines: List[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    t_start = time.time()

    # ── Header ────────────────────────────────────────────────────────────
    emit(_header(f"PDF CHUNK INSPECTOR — {filename}"))
    emit(f"  File    : {os.path.abspath(pdf_path)}")
    emit(f"  Report  : {os.path.abspath(output_path)}")
    emit(f"  Vision  : {'ENABLED — ' + vision_model if use_vision else 'DISABLED (heuristics only)'}")
    emit(f"  Chunking: child={CHILD_CHUNK_SIZE} chars  overlap={CHILD_CHUNK_OVERLAP}"
         f"  |  parent={PARENT_CHUNK_SIZE} chars  overlap={PARENT_CHUNK_OVERLAP}")
    emit()

    # ── Initialise classifier (optional) ─────────────────────────────────
    classifier        = None
    content_extractor = None
    if use_vision and ENABLE_PAGE_CLASSIFICATION:
        emit("▶  Initialising vision classifier …")
        try:
            classifier        = OllamaVisionClassifier(
                ollama_url     = ollama_url,
                classify_model = vision_model,
                describe_model = vision_desc_model,
            )
            content_extractor = ContentTypeExtractor(
                ollama_url     = ollama_url,
                classify_model = vision_model,
                describe_model = vision_desc_model,
            )
            emit(f"   ✓ Connected to Ollama at {ollama_url}")
        except Exception as exc:
            emit(f"   ⚠ Classifier unavailable ({exc}) — heuristics only.")
            classifier = content_extractor = None
    else:
        emit("▶  Heuristic classifier only (--no-vision).")

    # ── Load the full PDF ─────────────────────────────────────────────────
    emit()
    emit(f"▶  Loading & classifying: {filename} …")
    t0 = time.time()
    full_text, doc_meta = AdvancedDocumentLoader.load(
        pdf_path,
        classifier        = classifier,
        content_extractor = content_extractor,
    )
    load_elapsed = time.time() - t0

    if not full_text:
        emit("  ✗ No text could be extracted from this PDF. Aborting.")
        _write_report(output_path, lines)
        return

    total_pages         = doc_meta.get("num_pages", 0)
    page_types_meta: Dict[int, str] = doc_meta.get("page_types", {})

    emit(f"   ✓ Done in {load_elapsed:.1f}s  |  "
         f"{total_pages} pages  |  "
         f"{len(full_text):,} chars total")
    emit()

    # ── Parse into per-page blocks ────────────────────────────────────────
    page_blocks = _parse_page_blocks(full_text)

    if not page_blocks:
        emit("  ⚠ Could not parse per-page blocks from extracted text.")
        emit("    (The PDF may use a non-standard text layout.)")
        emit("    Proceeding with global-only analysis.")

    # ── Initialise chunker & builder ──────────────────────────────────────
    chunker = SemanticChunker(chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_CHUNK_OVERLAP)
    builder = ParentChildBuilder(chunker)

    # ═══════════════════════════════════════════════════════════════════════
    # PER-PAGE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    emit(_header("PER-PAGE CLASSIFICATION & CHUNKING BREAKDOWN"))
    emit()

    # Accumulate totals for the summary
    type_counter: Dict[str, int]  = {}
    skipped_pages: List[int]      = []
    total_per_page_parents: int   = 0
    total_per_page_children: int  = 0

    # Collect per-page data for the cross-table
    cross_table: List[Tuple[int, str, int, int, bool]] = []

    for page_num, page_type, page_body in page_blocks:
        emoji_label = _PAGE_TYPE_EMOJI.get(page_type, "❓ " + page_type)
        type_counter[page_type] = type_counter.get(page_type, 0) + 1

        is_skipped = page_type in (PageType.TOC, PageType.COVER)
        if is_skipped:
            skipped_pages.append(page_num)

        # ── Page header ───────────────────────────────────────────────────
        emit(_divider())
        skip_note = "  ⊘ NOT INDEXED" if is_skipped else ""
        emit(f"PAGE {page_num:>4}  |  {emoji_label.upper()}{skip_note}")
        emit(_divider())

        # ── Classification details ────────────────────────────────────────
        # page_types_meta is populated by OllamaVisionClassifier when it runs;
        # if a page number appears in the dict the classifier was active for that page.
        classified_by_vision = use_vision and (page_num in page_types_meta)
        has_tables_text  = "[Table " in page_body
        has_image_text   = "[IMAGE " in page_body
        has_diagram_text = "[DIAGRAM " in page_body
        emit(f"  Classifier result : {page_type}  "
             f"({'vision LLM' if classified_by_vision else 'heuristic'})")
        emit(f"  Text extracted    : {len(page_body):,} chars")
        emit(f"  Embedded tables   : {'YES — pdfplumber pipe-delimited' if has_tables_text else 'none'}")
        emit(f"  Embedded images   : {'YES' if has_image_text else 'none'}")
        emit(f"  Embedded diagrams : {'YES' if has_diagram_text else 'none'}")

        if is_skipped:
            emit()
            emit(f"  This page type ({page_type}) is not indexed — skipping chunk analysis.")
            cross_table.append((page_num, page_type, 0, 0, True))
            emit()
            continue

        if not page_body.strip():
            emit()
            emit("  No text content — no chunks produced.")
            cross_table.append((page_num, page_type, 0, 0, False))
            emit()
            continue

        # ── Run chunking on this page's text only ─────────────────────────
        sections = chunker.detect_sections(page_body)
        parents, children = builder.build(sections, file_hash_value=f"{filename}_p{page_num}")

        total_per_page_parents  += len(parents)
        total_per_page_children += len(children)
        cross_table.append((page_num, page_type, len(parents), len(children), False))

        # ── Section breakdown ─────────────────────────────────────────────
        emit()
        emit(f"  SECTIONS DETECTED:  {len(sections)}")
        for si, sec in enumerate(sections, 1):
            hier = " > ".join(sec.section_hierarchy) if sec.section_hierarchy else sec.title
            emit(f"    [{si:>3}] type={sec.section_type:<8}  "
                 f"level={sec.level}  "
                 f"len={len(sec.content):>5} chars")
            emit(f"          hierarchy : {hier[:80]}")
            emit(f"          preview   : {repr(sec.content[:100])}")

        # ── Parent chunks ─────────────────────────────────────────────────
        emit()
        emit(f"  PARENT CHUNKS:  {len(parents)}  "
             "(large context window — stored in _parents collection)")
        for pi, p in enumerate(parents, 1):
            hier = " > ".join(p.section_hierarchy) if p.section_hierarchy else p.section_title
            emit(f"    Parent {pi:>3}:  "
                 f"chars [{p.start_char:>6} … {p.end_char:>6}]  "
                 f"({p.end_char - p.start_char:>5} chars, {p.word_count} words)")
            emit(f"      section   : {hier[:75]}")
            emit(f"      page_type : {p.page_type}  |  chunk_type: {p.chunk_type}")
            emit(f"      parent_id : {p.parent_id}")
            emit(f"      preview   : {repr(p.text[:top_k_preview])}")

        # ── Child chunks ──────────────────────────────────────────────────
        emit()
        emit(f"  CHILD CHUNKS:  {len(children)}  "
             "(precision retrieval units — stored in _children collection)")
        for ci, c in enumerate(children, 1):
            hier = " > ".join(c.section_hierarchy) if c.section_hierarchy else c.section_title
            emit(f"    Child {ci:>4}:  "
                 f"chars [{c.start_char:>6} … {c.end_char:>6}]  "
                 f"({c.end_char - c.start_char:>5} chars)  "
                 f"words={c.word_count}  sents={c.sentence_count}  "
                 f"chunk_type={c.chunk_type}")
            emit(f"      parent_id : {c.parent_id[:32]}…")
            emit(f"      section   : {hier[:75]}")
            emit(f"      preview   : {repr(c.text[:top_k_preview])}")

        emit()

    # ═══════════════════════════════════════════════════════════════════════
    # GLOBAL RUN (whole document processed as one unit — mirrors real ingestion)
    # ═══════════════════════════════════════════════════════════════════════
    emit()
    emit(_header("GLOBAL CHUNKING (mirrors real ingestion — chunks may span pages)"))
    emit()
    emit("  Running section detection and parent-child build on the full document …")
    t0 = time.time()
    global_sections = chunker.detect_sections(full_text)
    global_parents, global_children = builder.build(global_sections, file_hash_value=filename)
    global_elapsed  = time.time() - t0

    emit(f"  Done in {global_elapsed:.1f}s")
    emit()
    emit(f"  Sections detected (global)  : {len(global_sections)}")
    emit(f"  Parent chunks (global)      : {len(global_parents)}"
         f"  → stored in _parents collection")
    emit(f"  Child  chunks (global)      : {len(global_children)}"
         f"  → stored in _children collection")
    emit()

    if global_parents:
        sizes = [p.end_char - p.start_char for p in global_parents]
        emit(f"  Parent size  min={min(sizes):>5}  max={max(sizes):>5}  "
             f"avg={sum(sizes)//len(sizes):>5} chars")
    if global_children:
        sizes = [c.end_char - c.start_char for c in global_children]
        emit(f"  Child  size  min={min(sizes):>5}  max={max(sizes):>5}  "
             f"avg={sum(sizes)//len(sizes):>5} chars")

    emit()
    emit("  Global section list (title | type | level | chars):")
    for si, sec in enumerate(global_sections, 1):
        hier = " > ".join(sec.section_hierarchy) if sec.section_hierarchy else sec.title
        emit(f"    [{si:>4}]  type={sec.section_type:<8}  "
             f"lvl={sec.level}  "
             f"len={len(sec.content):>6} chars  "
             f"│ {hier[:60]}")

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE × CHUNK CROSS-TABLE
    # ═══════════════════════════════════════════════════════════════════════
    emit()
    emit(_header("PAGE × CHUNK COUNT  (per-page independent analysis)"))
    emit()
    col = [6, 14, 10, 10, 14]
    emit(f"  {'Page':>{col[0]}}  "
         f"{'Type':<{col[1]}}  "
         f"{'Parents':>{col[2]}}  "
         f"{'Children':>{col[3]}}  "
         f"{'Note':<{col[4]}}")
    emit("  " + _divider("-", sum(col) + 8))
    for pn, pt, np_, nc, sk in cross_table:
        note = "⊘ skipped" if sk else ("— no text" if (np_ == 0 and nc == 0) else "")
        emit(f"  {pn:>{col[0]}}  "
             f"{pt:<{col[1]}}  "
             f"{np_:>{col[2]}}  "
             f"{nc:>{col[3]}}  "
             f"{note:<{col[4]}}")
    emit()
    emit(f"  TOTAL (per-page analysis)  parents={total_per_page_parents}  "
         f"children={total_per_page_children}")
    emit(f"  TOTAL (global  analysis)   parents={len(global_parents)}  "
         f"children={len(global_children)}")
    emit()
    emit("  Note: totals differ because global chunking allows chunks to span page")
    emit("  boundaries, while per-page analysis treats each page independently.")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    emit()
    emit(_header("SUMMARY"))
    emit()
    emit(f"  PDF                   : {filename}")
    emit(f"  Total pages           : {total_pages}")
    emit(f"  Skipped pages (TOC/cover): {len(skipped_pages)}"
         + (f"  →  pages {skipped_pages}" if skipped_pages else ""))
    emit(f"  Indexed pages         : {total_pages - len(skipped_pages)}")
    emit()
    emit("  Page-type breakdown:")
    total_classified = sum(type_counter.values())
    for pt in [PageType.TEXT, PageType.TABLE, PageType.IMAGE, PageType.DIAGRAM,
               PageType.EQUATION, PageType.MIXED, PageType.COVER, PageType.TOC,
               PageType.UNKNOWN]:
        count = type_counter.get(pt, 0)
        if count:
            emit(_bar(_PAGE_TYPE_EMOJI.get(pt, pt), count, total_classified))
    emit()
    emit(f"  Sections (global)     : {len(global_sections)}")
    emit(f"  Parent chunks (global): {len(global_parents)}  "
         f"(stored in _parents collection, ~{PARENT_CHUNK_SIZE} chars each)")
    emit(f"  Child  chunks (global): {len(global_children)}  "
         f"(stored in _children collection, ~{CHILD_CHUNK_SIZE} chars each)")

    # ═══════════════════════════════════════════════════════════════════════
    # ANOMALY CHECK
    # ═══════════════════════════════════════════════════════════════════════
    emit()
    emit(_header("ANOMALY CHECK"))
    emit()
    anomalies: List[str] = []
    for pn, pt, np_, nc, sk in cross_table:
        if sk:
            continue
        body = next((b for n, t, b in page_blocks if n == pn), "")
        body_len = len(body.strip())
        if nc == 0 and body_len > 200:
            anomalies.append(
                f"  ⚠  Page {pn:>3} ({pt}): {body_len} chars extracted but 0 child chunks."
                f"  Content may be mostly boilerplate or short lines."
            )
        if pt in (PageType.TABLE, PageType.MIXED) and np_ == 0:
            anomalies.append(
                f"  ⚠  Page {pn:>3} ({pt}): table/mixed page produced 0 parent chunks."
            )
        if pt in (PageType.IMAGE, PageType.DIAGRAM) and body_len < 50:
            anomalies.append(
                f"  ℹ  Page {pn:>3} ({pt}): very short text ({body_len} chars) — "
                f"vision description may be needed for useful retrieval."
            )
    if anomalies:
        for a in anomalies:
            emit(a)
    else:
        emit("  ✓  No anomalies detected.")
    emit()

    # ── Footer ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    emit(_header(f"Inspection complete in {total_elapsed:.1f}s"))
    emit()

    _write_report(output_path, lines)


def _write_report(output_path: str, lines: List[str]) -> None:
    """Write the collected report lines to a UTF-8 text file."""
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n✓ Report written to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="PDF_Chunk_Inspector",
        description=(
            "Inspect how Qdrant_Database_Generation_V3.py would classify and chunk "
            "a given PDF — without writing anything to Qdrant."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python PDF_Chunk_Inspector.py --pdf document.pdf\n"
            "  python PDF_Chunk_Inspector.py --pdf document.pdf --no-vision\n"
            "  python PDF_Chunk_Inspector.py --pdf document.pdf --output report.txt\n"
            "  python PDF_Chunk_Inspector.py --pdf document.pdf --top-k-preview 500\n"
        ),
    )
    parser.add_argument(
        "--pdf", required=True, metavar="PATH",
        help="Path to the PDF file to inspect.",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help=(
            "Path for the output report (.txt). "
            "Default: <pdf-stem>_inspection.txt in the same directory as the PDF."
        ),
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Disable Ollama vision classifier; use heuristics only (much faster).",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL, metavar="URL",
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--vision-model", default=DEFAULT_VISION_MODEL, metavar="MODEL",
        help=f"Ollama vision model for page classification (default: {DEFAULT_VISION_MODEL}).",
    )
    parser.add_argument(
        "--vision-desc-model", default=DEFAULT_VISION_DESC_MODEL, metavar="MODEL",
        help=f"Ollama model for image/diagram descriptions (default: {DEFAULT_VISION_DESC_MODEL}).",
    )
    parser.add_argument(
        "--top-k-preview", type=int, default=300, metavar="N",
        help="Number of characters to preview for each chunk in the report (default: 300).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        sys.exit(f"ERROR: File not found: {args.pdf}")
    if not args.pdf.lower().endswith(".pdf"):
        sys.exit(f"ERROR: Expected a .pdf file, got: {args.pdf}")

    if args.output is None:
        stem = Path(args.pdf).stem
        args.output = str(Path(args.pdf).parent / f"{stem}_inspection.txt")

    inspect_pdf(
        pdf_path          = args.pdf,
        output_path       = args.output,
        use_vision        = not args.no_vision,
        ollama_url        = args.ollama_url,
        vision_model      = args.vision_model,
        vision_desc_model = args.vision_desc_model,
        top_k_preview     = args.top_k_preview,
    )


if __name__ == "__main__":
    main()
