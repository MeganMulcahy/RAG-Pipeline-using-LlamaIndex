"""
rag_pipeline.py  —  Gemini-powered RAG pipeline

Structure
---------
1. Config & setup
2. Data classes        (PageInfo, LogicalDocument)
3. PDF extraction      (blob_reader — fitz + OCR fallback)
4. Boundary detection  (heuristics — split multi-doc PDF blobs)
5. Document grouping   (group_logical_docs)
6. PDF loading         (load_pdf → LlamaIndex Documents)
7. Indexing            (build_index — SentenceSplitter)
8. Hybrid retrieval    (HybridRetriever — BM25 + vector)
9. Pipeline            (build_rag_pipeline)
10. CLI entry point
"""

import hashlib
import io
import logging
import os
import re
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from google import genai
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv(override=True)

PDF_EXTENSION = ".pdf"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Silence chatty third-party loggers
for _log in (
    "httpx", "huggingface_hub", "sentence_transformers",
    "pikepdf", "unstructured", "pdfminer", "PIL",
):
    logging.getLogger(_log).setLevel(logging.WARNING)


# =============================================================================
# 1. CONFIG
# =============================================================================

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "models/gemini-3.1-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 100
TOP_K           = 4
MAX_EXTRACT_WORKERS = int(os.getenv("MAX_EXTRACT_WORKERS", "4"))
TESSERACT_CMD       = os.getenv("TESSERACT_CMD", "")
QUERY_REWRITE       = os.getenv("QUERY_REWRITE", "true").lower() == "true"

_tesseract_available: Optional[bool] = None
_tesseract_warned    = False

_PAGE_RE = re.compile(r'\bpage\s+\d+\s+of\s+\d+\b', re.IGNORECASE)


def _heuristic_boundary(curr_text: str) -> Optional[bool]:
    """
    Returns True  → definitely same doc.
    Returns None  → uncertain (defaults to same document).
    """
    stripped = curr_text.strip()
    if len(stripped) < 120:
        return True   # blank / near-blank page → continuation
    if _PAGE_RE.search(stripped):
        return True   # "Page N of M" header/footer → continuation
    return None


# =============================================================================
# 2. GEMINI + EMBEDDING SETUP
# =============================================================================

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Add it to your .env file.")

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


class GeminiLLM(CustomLLM):
    """Minimal LlamaIndex-compatible wrapper around google.genai."""
    context_window: int = 8192
    num_output: int     = 1024
    model_name: str     = GEMINI_MODEL

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **_) -> CompletionResponse:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **_):
        yield self.complete(prompt)


_llm        = GeminiLLM()
_embed      = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
Settings.llm         = _llm
Settings.embed_model = _embed


# =============================================================================
# 3. DATA CLASSES
# =============================================================================

@dataclass
class PageInfo:
    page_num:     int
    text:         str
    page_in_doc:  int           = 0
    content_type: str          = "native_text"
    char_count:   int          = 0
    table_html:   Optional[str] = None


@dataclass
class LogicalDocument:
    doc_id:     str
    page_start: int
    page_end:   int
    text:       str


# =============================================================================
# 4. PDF EXTRACTION
# =============================================================================

def parse_manifest_entry(entry: str) -> dict:
    """Parse compact manifest: page|page_in_doc|file_id|file_name|content_type"""
    page, page_in_doc, file_id, file_name, content_type = entry.split("|", 4)
    return {
        "page":         int(page),
        "page_in_doc":  int(page_in_doc),
        "file_id":      file_id,
        "file_name":    file_name,
        "content_type": content_type,
    }


def _classify_page_content_type(page, char_count: int) -> str:
    """Pass 1: fitz-only signals → native_text | image_dominant | table_heavy."""
    page_area  = page.rect.width * page.rect.height or 1.0

    image_area = 0.0
    for block in page.get_text("blocks"):
        if block[6] == 1:
            x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
            image_area += (x1 - x0) * (y1 - y0)

    line_count = 0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") == 0:
            line_count += len(block.get("lines", []))

    rect_count  = len(page.get_drawings())
    image_ratio = image_area / page_area

    if rect_count > 20 and line_count > 10:
        return "table_heavy"
    if char_count > 100 and image_ratio < 0.6:
        return "native_text"
    if image_ratio >= 0.6 or char_count < 50:
        return "image_dominant"
    return "native_text"


def _configure_tesseract() -> bool:
    """Detect Tesseract once; set pytesseract path on Windows if needed."""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available

    candidates = []
    if TESSERACT_CMD:
        candidates.append(TESSERACT_CMD)
    which = shutil.which("tesseract")
    if which:
        candidates.append(which)
    candidates.extend([
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ])

    for cmd in candidates:
        if cmd and os.path.isfile(cmd):
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = cmd
                _tesseract_available = True
                logging.debug("Tesseract configured: %s", cmd)
                return True
            except Exception:
                continue

    _tesseract_available = False
    return False


def _warn_tesseract_missing() -> None:
    global _tesseract_warned
    if _tesseract_warned:
        return
    _tesseract_warned = True
    logging.warning(
        "Tesseract OCR is not installed or not on PATH. "
        "Install from https://github.com/UB-Mannheim/tesseract/wiki "
        "or set TESSERACT_CMD in .env. Scanned pages will use fitz text only."
    )


def _get_tesseract_cmd() -> str:
    """Resolved tesseract.exe path for main process and worker processes."""
    if _configure_tesseract():
        import pytesseract
        return pytesseract.pytesseract.tesseract_cmd
    return TESSERACT_CMD if TESSERACT_CMD and os.path.isfile(TESSERACT_CMD) else ""


def _apply_tesseract_cmd(cmd: str) -> None:
    """Configure tesseract in a worker process (ProcessPool children)."""
    if not cmd or not os.path.isfile(cmd):
        return
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = cmd
    tess_dir = os.path.dirname(cmd)
    os.environ["PATH"] = tess_dir + os.pathsep + os.environ.get("PATH", "")


def _log_page_extraction(pinfo: PageInfo, extractor: str) -> None:
    print(
        f"  Page {pinfo.page_num}: content_type={pinfo.content_type} "
        f"extractor={extractor} chars={len(pinfo.text)}"
    )


def _fitz_page_text(page) -> str:
    return page.get_text().strip()


def _ocr_page_text(page, page_num: int) -> str:
    """pytesseract fallback for scanned pages when unstructured is unavailable."""
    if not _configure_tesseract():
        _warn_tesseract_missing()
        return ""
    try:
        from PIL import Image
        import pytesseract
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception as e:
        logging.warning("Page %d: OCR failed — %s", page_num, e)
        return ""


def _finalize_slow_page(
    doc,
    pinfo: PageInfo,
    text: str,
    table_html: Optional[str],
    extractor: str,
) -> None:
    """fitz → OCR (if installed) so pages are not left empty."""
    if not text.strip():
        fitz_text = _fitz_page_text(doc[pinfo.page_num])
        if fitz_text:
            text = fitz_text
            extractor = "fitz_fallback"

    if not text.strip() and _configure_tesseract():
        text = _ocr_page_text(doc[pinfo.page_num], pinfo.page_num)
        if text.strip():
            extractor = "pytesseract"
    elif not text.strip():
        _warn_tesseract_missing()

    pinfo.text = text
    pinfo.table_html = table_html
    _log_page_extraction(pinfo, extractor if text.strip() else "none")


def _chunk_to_text(chunk) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text", "") or "").strip()
    return str(chunk).strip()


def _extract_native_text_batch(path: str, page_indices: List[int]) -> Dict[int, str]:
    """One pymupdf4llm call for all native_text pages."""
    if not page_indices:
        return {}
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(path, pages=page_indices, page_chunks=True)
    out: Dict[int, str] = {}

    if isinstance(chunks, list):
        for i, page_idx in enumerate(page_indices):
            if i < len(chunks):
                text = _chunk_to_text(chunks[i])
                if text:
                    out[page_idx] = text
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            meta = chunk.get("metadata") or {}
            pg = meta.get("page") if isinstance(meta, dict) else chunk.get("page")
            if pg is not None:
                text = _chunk_to_text(chunk)
                if text:
                    out[int(pg)] = text
    elif len(page_indices) == 1:
        out[page_indices[0]] = str(chunks).strip()

    return out


def _elements_to_text_and_html(elements) -> Tuple[str, Optional[str]]:
    texts: List[str] = []
    table_html: Optional[str] = None
    for el in elements:
        txt = str(getattr(el, "text", "") or "").strip()
        if txt:
            texts.append(txt)
        meta = getattr(el, "metadata", None)
        html = getattr(meta, "text_as_html", None) if meta else None
        if html and getattr(el, "category", "") == "Table":
            table_html = html
    return "\n".join(texts), table_html


def _extract_single_page(
    args: Tuple[str, int, str, int, str],
) -> Tuple[int, str, Optional[str], str]:
    """
    Top-level worker for ProcessPoolExecutor (must be picklable).
    args: (path, page_num, content_type, char_count, tesseract_cmd)
    """
    path, page_num, content_type, char_count, tesseract_cmd = args
    _apply_tesseract_cmd(tesseract_cmd)
    try:
        from unstructured.partition.auto import partition

        strategy = "hi_res" if char_count < 50 else "fast"
        kwargs = {
            "filename":     path,
            "strategy":     strategy,
            "page_numbers": [page_num + 1],
            "languages":    ["eng"],
        }
        if content_type == "table_heavy":
            kwargs["infer_table_structure"] = True

        elements = partition(**kwargs)
        text, table_html = _elements_to_text_and_html(elements)
        label = f"unstructured_{strategy}"
        if content_type == "table_heavy":
            label += "_table"
        return page_num, text, table_html, label
    except Exception as e:
        logging.warning("Page %d: unstructured worker failed — %s", page_num, e)
        return page_num, "", None, "unstructured_error"


def blob_reader(pdf_path) -> List[PageInfo]:
    """
    Open a PDF from a file path, file-like object, or dict with 'content' key.

    Pass 1: fitz-only page classification (native_text / image_dominant / table_heavy).
    Pass 2: routed extraction via pymupdf4llm or unstructured; OCR if needed.
    """
    temp_path = None
    if isinstance(pdf_path, dict) and "content" in pdf_path:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(pdf_path["content"])
        tmp.close()
        path = temp_path = tmp.name
        doc  = fitz.open(path)
    elif hasattr(pdf_path, "read"):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(pdf_path.read())
        tmp.close()
        path = temp_path = tmp.name
        doc  = fitz.open(path)
    else:
        path = str(pdf_path)
        doc  = fitz.open(path)

    pages: List[PageInfo] = []
    try:
        # Pass 1 — classify every page (fitz only)
        for i, page in enumerate(doc):
            char_count = len(page.get_text())
            content_type = _classify_page_content_type(page, char_count)
            pages.append(PageInfo(
                page_num=i,
                text="",
                content_type=content_type,
                char_count=char_count,
            ))

        # Pass 2a — batched pymupdf4llm (native_text + table_heavy pages with text layer)
        markdown_pages = [
            p.page_num for p in pages
            if p.content_type == "native_text"
            or (p.content_type == "table_heavy" and p.char_count >= 50)
        ]
        markdown_texts: Dict[int, str] = {}
        if markdown_pages:
            try:
                markdown_texts = _extract_native_text_batch(path, markdown_pages)
                logging.info(
                    "pymupdf4llm batch: %d page(s) in one call",
                    len(markdown_pages),
                )
            except Exception as e:
                logging.warning("pymupdf4llm batch failed — %s", e)

        for pinfo in pages:
            if pinfo.page_num not in markdown_pages:
                continue
            text = markdown_texts.get(pinfo.page_num, "")
            extractor = "pymupdf4llm"
            if not text.strip():
                text = _fitz_page_text(doc[pinfo.page_num])
                extractor = "fitz_fallback"
            pinfo.text = text
            _log_page_extraction(pinfo, extractor)

        tesseract_cmd = _get_tesseract_cmd()

        # image_dominant: OCR in main process (fast; avoids unstructured re-reading PDF)
        for pinfo in pages:
            if pinfo.content_type == "image_dominant":
                _finalize_slow_page(doc, pinfo, "", None, "pending")

        # table_heavy scanned pages only — unstructured in parallel workers
        slow_args = [
            (path, p.page_num, p.content_type, p.char_count, tesseract_cmd)
            for p in pages
            if p.content_type == "table_heavy" and p.char_count < 50
        ]
        slow_results: Dict[int, Tuple[str, Optional[str], str]] = {}
        if slow_args:
            workers = min(MAX_EXTRACT_WORKERS, len(slow_args))
            logging.info(
                "unstructured parallel: %d page(s), %d worker(s)",
                len(slow_args), workers,
            )
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_extract_single_page, args): args
                    for args in slow_args
                }
                for future in as_completed(futures):
                    page_num, text, table_html, extractor = future.result()
                    slow_results[page_num] = (text, table_html, extractor)

        for pinfo in pages:
            if pinfo.page_num in markdown_pages or pinfo.content_type != "table_heavy":
                continue
            text, table_html, extractor = slow_results.get(
                pinfo.page_num, ("", None, "unstructured_error"),
            )
            _finalize_slow_page(doc, pinfo, text, table_html, extractor)
    finally:
        doc.close()
        if temp_path and os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return pages


def classify_pages_only(pdf_path: str) -> Tuple[str, List[PageInfo]]:
    """Pass 1 only: return (pdf_path, PageInfo list with content_type, no extraction)."""
    doc = fitz.open(pdf_path)
    path = str(pdf_path)
    pages: List[PageInfo] = []
    try:
        for i, page in enumerate(doc):
            char_count = len(page.get_text())
            pages.append(PageInfo(
                page_num=i,
                text="",
                content_type=_classify_page_content_type(page, char_count),
                char_count=char_count,
            ))
    finally:
        doc.close()
    return path, pages


def extract_page_content(
    path: str,
    doc: fitz.Document,
    pinfo: PageInfo,
) -> Dict[str, object]:
    """
    Run Pass 2 extraction for a single page. Returns ingest details for inspection.
    """
    page = doc[pinfo.page_num]
    fitz_raw = page.get_text()
    text, table_html, extractor = "", None, "none"

    use_markdown = (
        pinfo.content_type == "native_text"
        or (pinfo.content_type == "table_heavy" and pinfo.char_count >= 50)
    )
    if use_markdown:
        try:
            batch = _extract_native_text_batch(path, [pinfo.page_num])
            text = batch.get(pinfo.page_num, "")
            if text.strip():
                extractor = "pymupdf4llm"
        except Exception as e:
            logging.warning("Page %d pymupdf4llm: %s", pinfo.page_num, e)
        if not text.strip():
            text = _fitz_page_text(page)
            extractor = "fitz_fallback"

    elif pinfo.content_type == "image_dominant":
        # Same as blob_reader: fitz → Tesseract, no unstructured
        text = _fitz_page_text(page)
        extractor = "fitz_fallback" if text.strip() else "none"
        if not text.strip() and _configure_tesseract():
            text = _ocr_page_text(page, pinfo.page_num)
            extractor = "pytesseract" if text.strip() else "none"
        elif not text.strip():
            _warn_tesseract_missing()

    elif pinfo.content_type == "table_heavy" and pinfo.char_count < 50:
        tcmd = _get_tesseract_cmd()
        args = (path, pinfo.page_num, pinfo.content_type, pinfo.char_count, tcmd)
        _, text, table_html, extractor = _extract_single_page(args)
        pinfo_tmp = PageInfo(
            page_num=pinfo.page_num, text="", content_type=pinfo.content_type,
            char_count=pinfo.char_count,
        )
        _finalize_slow_page(doc, pinfo_tmp, text, table_html, extractor)
        text = pinfo_tmp.text
        table_html = pinfo_tmp.table_html

    return {
        "page_num":       pinfo.page_num,
        "content_type":   pinfo.content_type,
        "char_count_fitz": pinfo.char_count,
        "extractor":      extractor,
        "text":           text,
        "table_html":     table_html,
        "fitz_raw":       fitz_raw,
    }


# =============================================================================
# 5. BOUNDARY DETECTION
# =============================================================================

def detect_document_boundary(prev_text: str, curr_text: str) -> bool:
    """
    Returns True if the two pages belong to the SAME logical document.
    Uses layout heuristics only (domain-agnostic).
    """
    if not prev_text or not curr_text:
        return False

    hint = _heuristic_boundary(curr_text)
    if hint is not None:
        return hint

    return True


# =============================================================================
# 6. DOCUMENT GROUPING
# =============================================================================

def group_logical_docs(pages: List[PageInfo]) -> List[LogicalDocument]:
    """Group pages into logical documents using page_in_doc boundaries."""
    logical_docs: List[LogicalDocument] = []
    current_pages: List[PageInfo] = []
    doc_counter = 0

    for page in pages:
        if not page.text.strip():
            continue

        if page.page_in_doc == 0 and current_pages:
            logical_docs.append(LogicalDocument(
                doc_id     = f"doc_{doc_counter}",
                page_start = current_pages[0].page_num,
                page_end   = current_pages[-1].page_num,
                text       = "\n\n".join(p.text for p in current_pages),
            ))
            doc_counter += 1
            current_pages = []

        current_pages.append(page)

    if current_pages:
        logical_docs.append(LogicalDocument(
            doc_id     = f"doc_{doc_counter}",
            page_start = current_pages[0].page_num,
            page_end   = current_pages[-1].page_num,
            text       = "\n\n".join(p.text for p in current_pages),
        ))

    return logical_docs


# =============================================================================
# 7. PDF LOADING  →  LlamaIndex Documents
# =============================================================================

page_manifest: List[str] = []   # populated by load_pdf; compact per-page metadata


def load_pdf(pdf_path) -> List[Document]:
    """
    Full ingestion pipeline for one PDF:
      blob_reader → heuristic boundaries → group → Documents
    """
    global page_manifest
    page_manifest = []

    pages     = blob_reader(pdf_path)
    file_name = (
        os.path.basename(pdf_path.name) if hasattr(pdf_path, "name")
        else os.path.basename(str(pdf_path))
    )
    file_id     = str(uuid.uuid4())
    total_pages = len(pages)
    non_empty   = [(i, p) for i, p in enumerate(pages) if p.text.strip()]

    print(f"\n  Segmenting {len(non_empty)}/{total_pages} pages (heuristic boundaries)…")

    page_in_doc = 0
    prev_page: Optional[PageInfo] = None

    for page in pages:
        if not page.text.strip():
            continue

        if prev_page is None:
            page_in_doc = 0
        else:
            same = detect_document_boundary(prev_page.text, page.text)
            page_in_doc = page_in_doc + 1 if same else 0

        page.page_in_doc = page_in_doc
        prev_page        = page

        print(
            f"  Page {page.page_num + 1}/{total_pages} | "
            f"{page.content_type:<14} | segment_page: {page_in_doc}"
        )

        page_manifest.append(
            f"{page.page_num}|{page_in_doc}|{file_id}|{file_name}|{page.content_type}"
        )

    logical_docs = group_logical_docs(pages)
    print(f"\n  {len(logical_docs)} logical document(s) identified.")

    return [
        Document(
            text     = doc.text,
            metadata = {
                "file_id":    file_id,
                "file_name":  file_name,
                "doc_id":     doc.doc_id,
                "page_start": doc.page_start,
                "page_end":   doc.page_end,
            },
        )
        for doc in logical_docs
    ]


def load_pdfs(paths: List[str]) -> List[Document]:
    """Load multiple files (PDF or other supported types)."""
    all_docs: List[Document] = []
    for path in paths:
        all_docs.extend(load_file(path))
    return all_docs


def _load_unstructured_file(path: Path) -> List[Document]:
    """Extract text from non-PDF files (Word, Excel, slides, HTML, images, txt, …)."""
    global page_manifest
    page_manifest = []

    from unstructured.partition.auto import partition

    _apply_tesseract_cmd(_get_tesseract_cmd())
    print(f"\n  Extracting {path.name} via unstructured…")
    elements = partition(filename=str(path), languages=["eng"])
    text, table_html = _elements_to_text_and_html(elements)
    if not text.strip():
        raise ValueError(f"No text extracted from {path.name}")

    file_id = str(uuid.uuid4())
    page_manifest.append(f"0|0|{file_id}|{path.name}|unstructured")
    metadata = {
        "file_id":      file_id,
        "file_name":    path.name,
        "doc_id":       "doc_0",
        "page_start":   0,
        "page_end":     0,
        "content_type": "unstructured",
    }
    if table_html:
        metadata["has_tables"] = True

    print(f"  Extracted {len(text)} characters from {path.name}")
    return [Document(text=text, metadata=metadata)]


def load_file(file_path) -> List[Document]:
    """
    Ingest one file. PDFs use the two-pass page pipeline; other types use
    unstructured (partition.auto) — docx, xlsx, pptx, html, images, txt, etc.
    """
    path = Path(file_path)
    if path.suffix.lower() == PDF_EXTENSION:
        return load_pdf(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return _load_unstructured_file(path)


# =============================================================================
# 9. QUERY REWRITE + DEDUPLICATION
# =============================================================================

def rewrite_query(query: str) -> str:
    """Fix typos / normalize vague queries before retrieval (Gemini)."""
    if not QUERY_REWRITE or not query.strip():
        return query
    prompt = (
        "You fix search queries for document retrieval.\n"
        "Correct spelling and grammar. Keep the same meaning and intent.\n"
        "Return ONLY the corrected query — no quotes, no explanation.\n\n"
        f"Query: {query.strip()}\n\n"
        "Corrected query:"
    )
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        fixed = (response.text or "").strip().split("\n")[0].strip().strip('"')
        if fixed and len(fixed) >= 2 and fixed.lower() != query.strip().lower():
            print(f"  Query rewrite: {query!r} → {fixed!r}")
            return fixed
    except Exception as e:
        logging.warning("Query rewrite failed: %s", e)
    return query


def _node_dedupe_key(node) -> tuple:
    m = node.metadata or {}
    text = (node.get_content() if hasattr(node, "get_content") else "")[:200]
    return (
        m.get("file_id"),
        m.get("file_name"),
        m.get("doc_id"),
        m.get("page_start"),
        m.get("page_end"),
        text,
    )


def dedupe_nodes(nodes: list) -> list:
    """Drop duplicate chunks (same segment + similar text)."""
    seen: set = set()
    unique = []
    for node in nodes:
        key = _node_dedupe_key(node)
        if key in seen:
            continue
        seen.add(key)
        unique.append(node)
    return unique


def dedupe_source_nodes(nodes: list) -> list:
    """Dedupe citation lines for CLI display (by page range + file + segment)."""
    seen: set = set()
    unique = []
    for node in nodes:
        m = node.metadata or {}
        key = (m.get("file_name"), m.get("page_start"), m.get("page_end"), m.get("doc_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(node)
    return unique


# =============================================================================
# 10. INDEXING
# =============================================================================

def build_index(documents: List[Document]) -> VectorStoreIndex:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = dedupe_nodes(nodes)
    logging.info("Indexing %d unique chunk(s) after deduplication", len(nodes))
    return VectorStoreIndex(nodes=nodes)


# =============================================================================
# 11. HYBRID RETRIEVER
# =============================================================================

class HybridRetriever(BaseRetriever):
    """Merges BM25 (keyword) and vector results, deduplicated by node ID."""

    def __init__(self, vector_retriever, bm25_retriever, top_k: int = TOP_K):
        self.vector_retriever = vector_retriever
        self.bm25_retriever   = bm25_retriever
        self.top_k            = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes  = self.vector_retriever.retrieve(query_bundle)
        keyword_nodes = self.bm25_retriever.retrieve(query_bundle)

        by_key: Dict[tuple, NodeWithScore] = {}
        for node in vector_nodes + keyword_nodes:
            key = _node_dedupe_key(node)
            prev = by_key.get(key)
            if prev is None or (node.score or 0) > (prev.score or 0):
                by_key[key] = node

        merged = sorted(
            by_key.values(),
            key=lambda n: n.score if n.score is not None else 0.0,
            reverse=True,
        )
        return merged[:self.top_k]


# =============================================================================
# 12. PIPELINE ASSEMBLY
# =============================================================================

def _make_engine(index: VectorStoreIndex, nodes: list) -> RetrieverQueryEngine:
    safe_k = min(TOP_K, max(1, len(nodes)))
    vector_retriever = index.as_retriever(similarity_top_k=safe_k)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=safe_k)
    hybrid         = HybridRetriever(vector_retriever, bm25_retriever, top_k=safe_k)

    postprocessors = []
    if len(nodes) > 1:
        try:
            postprocessors = [SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=min(TOP_K, len(nodes)),
            )]
        except Exception as e:
            logging.warning(f"Reranker unavailable: {e}")

    return RetrieverQueryEngine.from_args(
        retriever        = hybrid,
        llm              = _llm,
        node_postprocessors = postprocessors,
    )


def build_rag_pipeline(index: VectorStoreIndex) -> RetrieverQueryEngine:
    nodes = list(index.docstore.docs.values())
    return _make_engine(index, nodes)


# =============================================================================
# 13. INGEST
# =============================================================================

def ingest_file(file_path: str) -> Tuple[VectorStoreIndex, RetrieverQueryEngine]:
    """Load any supported file, chunk, embed, and build the query engine."""
    docs = load_file(file_path)
    index = build_index(docs)
    engine = build_rag_pipeline(index)
    return index, engine


# =============================================================================
# 14. CLI ENTRY POINT
# =============================================================================

def main():
    file_path = input(
        "File path (PDF, Word, Excel, slides, HTML, txt, images, …): "
    ).strip().strip('"')
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\nLoading {os.path.basename(file_path)}…")
    _, engine = ingest_file(file_path)
    seg = "page(s)" if Path(file_path).suffix.lower() == PDF_EXTENSION else "segment(s)"
    print(f"Ingested {len(page_manifest)} {seg}.")
    print('Ready — type your question or "exit" to quit.\n')

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        search_query = rewrite_query(query)
        response = engine.query(search_query)
        print(f"\n{response}\n")
        for node in dedupe_source_nodes(response.source_nodes):
            m = node.metadata
            print(
                f"  [p.{m.get('page_start')}-{m.get('page_end')}  "
                f"{m.get('file_name')}  {m.get('doc_id')}]"
            )
        print("-" * 60)


if __name__ == "__main__":
    main()
