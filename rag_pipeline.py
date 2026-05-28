"""
rag_pipeline.py  —  Gemini-powered RAG pipeline

Structure
---------
1. Config & setup
2. Data classes        (PageInfo, LogicalDocument)
3. PDF extraction      (blob_reader — fitz + OCR fallback)
4. Classification      (keyword labels; Gemini only if INGEST_LLM=true)
5. Boundary detection  (heuristics; Gemini only if INGEST_LLM=true)
6. Document grouping   (group_logical_docs)
7. PDF loading         (load_pdf → LlamaIndex Documents)
8. Indexing            (build_index — SentenceSplitter)
9. Hybrid retrieval    (HybridRetriever — BM25 + vector)
10. Pipeline           (build_rag_pipeline, build_filtered_engine)
11. Query routing      (optional keyword filter; full-index search by default)
12. CLI entry point
"""

import io
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Silence chatty third-party loggers
for _log in ("httpx", "huggingface_hub", "sentence_transformers"):
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

# Query: pre-filter by doc_type keywords (set true to enable). Default searches full index.
QUERY_DOC_TYPE_ROUTING = os.getenv("QUERY_DOC_TYPE_ROUTING", "false").lower() == "true"

_DOC_TYPES = [
    "Resume", "Contract", "Mortgage Contract", "Invoice", "Pay Slip",
    "Lender Fee Sheet", "Land Deed", "Bank Statement", "Tax Document",
    "Insurance", "Report", "Letter", "Form", "ID Document", "Medical", "Other",
]

_KEYWORD_RULES = [
    ("Resume",            ["resume", "curriculum vitae", " cv ", "work experience", "education", "skills", "objective"]),
    ("Lender Fee Sheet",  ["loan estimate", "closing disclosure", "origination charge", "apr", "annual percentage rate",
                           "discount point", "closing cost", "lender credit", "prepaid interest", "cash to close"]),
    ("Mortgage Contract", ["mortgage", "deed of trust", "promissory note", "deed of mortgage", "home loan"]),
    ("Contract",          ["agreement", "whereas", "hereinafter", "obligations", "parties agree", "terms and conditions"]),
    ("Invoice",           ["invoice", "bill to", "amount due", "payment terms", "subtotal", "purchase order"]),
    ("Pay Slip",          ["pay stub", "pay slip", "payslip", "gross pay", "net pay", "deductions", "ytd", "federal tax"]),
    ("Land Deed",         ["deed", "grantor", "grantee", "real property", "parcel", "legal description", "county recorder"]),
    ("Bank Statement",    ["account number", "statement period", "beginning balance", "ending balance", "deposits", "withdrawals"]),
    ("Tax Document",      ["form w-2", "form 1099", "form 1040", "taxable income", "withholding", "irs", "tax return"]),
    ("Insurance",         ["policy number", "insured", "coverage", "premium", "deductible", "beneficiary", "underwriter"]),
    ("Medical",           ["patient", "diagnosis", "physician", "prescription", "medical record", "treatment", "dosage"]),
]

_PAGE_RE = re.compile(r'\bpage\s+\d+\s+of\s+\d+\b', re.IGNORECASE)


def _keyword_classify(text: str) -> Optional[str]:
    """Return a doc type if 2+ keywords match, else None."""
    lower = text.lower()
    for doc_type, keywords in _KEYWORD_RULES:
        if sum(1 for kw in keywords if kw in lower) >= 2:
            return doc_type
    return None


def _heuristic_boundary(curr_text: str) -> Optional[bool]:
    """
    Returns True  → definitely same doc.
    Returns None  → uncertain (keyword mismatch rule decides).
    """
    stripped = curr_text.strip()
    if len(stripped) < 120:
        return True   # blank / near-blank page → continuation
    if _PAGE_RE.search(stripped):
        return True   # "Page N of M" header/footer → continuation
    return None


_QUERY_ROUTING_RULES = [
    ("Lender Fee Sheet",  ["origination", "closing cost", "discount point", "appraisal fee", "apr"]),
    ("Pay Slip",          ["salary", "gross pay", "net pay", "paycheck", "deductions", "ytd"]),
    ("Bank Statement",    ["account balance", "deposits", "withdrawals", "bank statement"]),
    ("Tax Document",      ["w-2", "1099", "taxable income", "irs", "tax return"]),
    ("Invoice",           ["invoice", "bill to", "amount due", "purchase order"]),
    ("Mortgage Contract", ["mortgage", "promissory note", "deed of trust", "escrow", "loan amount"]),
    ("Land Deed",         ["grantor", "grantee", "parcel", "legal description"]),
    ("Contract",          ["agreement", "obligations", "indemnification", "whereas"]),
    ("Resume",            ["resume", "work experience", "employment history"]),
    ("Insurance",         ["policy number", "premium", "coverage", "deductible"]),
    ("Medical",           ["diagnosis", "prescription", "patient", "physician"]),
]


# =============================================================================
# 2. GEMINI + EMBEDDING SETUP
# =============================================================================

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Add it to your .env file.")

_key_preview = f"{GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}" if len(GEMINI_API_KEY) > 8 else "too short"
logging.info("Gemini key loaded: %s (len=%d)", _key_preview, len(GEMINI_API_KEY))
genai.configure(api_key=GEMINI_API_KEY)
_gemini = genai.GenerativeModel(GEMINI_MODEL)


class GeminiLLM(CustomLLM):
    """Minimal LlamaIndex-compatible wrapper around google.generativeai."""
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
        text = _gemini.generate_content(prompt).text or ""
        return CompletionResponse(text=text.strip())

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
    page_num:   int
    text:       str
    doc_type:   Optional[str] = None
    page_in_doc: int          = 0


@dataclass
class LogicalDocument:
    doc_id:     str
    doc_type:   str
    page_start: int
    page_end:   int
    text:       str


# =============================================================================
# 4. PDF EXTRACTION
# =============================================================================

def blob_reader(pdf_path) -> List[PageInfo]:
    """
    Open a PDF from a file path, file-like object, or dict with 'content' key.
    Extracts text per page; falls back to pytesseract OCR for scanned pages.
    """
    if isinstance(pdf_path, dict) and "content" in pdf_path:
        doc = fitz.open(stream=pdf_path["content"], filetype="pdf")
    elif hasattr(pdf_path, "read"):
        doc = fitz.open(stream=pdf_path.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_path)

    pages: List[PageInfo] = []
    for i, page in enumerate(doc):
        text = page.get_text()

        if not text.strip():
            try:
                from PIL import Image
                import pytesseract
                pix  = page.get_pixmap()
                img  = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                print(f"  Page {i}: OCR extracted {len(text)} chars")
            except Exception as e:
                logging.warning(f"Page {i}: OCR failed — {e}")
                text = ""

        pages.append(PageInfo(page_num=i, text=text))

    doc.close()
    return pages


# =============================================================================
# 5. DOCUMENT CLASSIFICATION
# =============================================================================

def classify_document_type(text: str) -> str:
    """Keyword label for metadata."""
    fast = _keyword_classify(text)
    if fast:
        return fast
    return "Other"


# =============================================================================
# 6. BOUNDARY DETECTION
# =============================================================================

def detect_document_boundary(prev_text: str, curr_text: str,
                              doc_type: str = None) -> bool:
    """
    Returns True if the two pages belong to the SAME document.
    Heuristics first; then keyword mismatch rule.
    """
    if not prev_text or not curr_text:
        return False

    hint = _heuristic_boundary(curr_text)
    if hint is not None:
        return hint

    prev_kw = _keyword_classify(prev_text)
    curr_kw = _keyword_classify(curr_text)
    if prev_kw and curr_kw and prev_kw != curr_kw:
        return False
    return True


# =============================================================================
# 7. DOCUMENT GROUPING
# =============================================================================

def group_logical_docs(pages: List[PageInfo]) -> List[LogicalDocument]:
    """Group a flat list of classified PageInfo objects into logical documents."""
    logical_docs: List[LogicalDocument] = []
    current_pages: List[PageInfo]       = []
    current_type: Optional[str]         = None
    doc_counter                         = 0

    for page in pages:
        if not page.text.strip():
            continue

        if page.page_in_doc == 0 and current_pages:
            logical_docs.append(LogicalDocument(
                doc_id     = f"doc_{doc_counter}",
                doc_type   = current_type,
                page_start = current_pages[0].page_num,
                page_end   = current_pages[-1].page_num,
                text       = "\n\n".join(p.text for p in current_pages),
            ))
            doc_counter += 1
            current_pages = []

        current_pages.append(page)
        current_type = page.doc_type

    if current_pages:
        logical_docs.append(LogicalDocument(
            doc_id     = f"doc_{doc_counter}",
            doc_type   = current_type,
            page_start = current_pages[0].page_num,
            page_end   = current_pages[-1].page_num,
            text       = "\n\n".join(p.text for p in current_pages),
        ))

    return logical_docs


# =============================================================================
# 8. PDF LOADING  →  LlamaIndex Documents
# =============================================================================

pdf_metadata_store: List[Dict] = []   # populated by load_pdf; used for routing


def load_pdf(pdf_path) -> List[Document]:
    """
    Full ingestion pipeline for one PDF:
      blob_reader → keyword classify → heuristic boundaries → group → Documents
    """
    global pdf_metadata_store
    pdf_metadata_store = []

    pages     = blob_reader(pdf_path)
    file_name = (
        os.path.basename(pdf_path.name) if hasattr(pdf_path, "name")
        else os.path.basename(str(pdf_path))
    )
    file_id     = str(uuid.uuid4())
    total_pages = len(pages)
    non_empty   = [(i, p) for i, p in enumerate(pages) if p.text.strip()]

    print(f"\n  Labeling {len(non_empty)}/{total_pages} pages (keywords + continuity)…")
    for page in pages:
        if page.text.strip():
            # Keep raw keyword label only; if absent, decide later using boundary continuity.
            page.doc_type = _keyword_classify(page.text)

    page_in_doc = 0
    prev_page: Optional[PageInfo] = None

    for page in pages:
        if not page.text.strip():
            continue

        if prev_page is None:
            if not page.doc_type:
                page.doc_type = "Other"
            page_in_doc = 0
        else:
            prev_type = prev_page.doc_type or "Other"
            curr_type = page.doc_type
            same = detect_document_boundary(prev_page.text, page.text, prev_type)

            if curr_type is None:
                # No new keyword hit on this page: inherit type if boundary says continuation.
                if same:
                    page.doc_type = prev_type
                    page_in_doc += 1
                else:
                    page.doc_type = "Other"
                    page_in_doc = 0
            elif curr_type != prev_type:
                # Strong keyword shift starts a new logical document.
                page_in_doc = 0
            else:
                page_in_doc = page_in_doc + 1 if same else 0

        page.page_in_doc = page_in_doc
        prev_page        = page

        print(f"  Page {page.page_num + 1}/{total_pages} | {page.doc_type:<20} | page_in_doc: {page_in_doc}")

        pdf_metadata_store.append({
            "page":        page.page_num,
            "text":        page.text,
            "doc_type":    page.doc_type,
            "page_in_doc": page_in_doc,
            "file_id":     file_id,
            "file_name":   file_name,
        })

    logical_docs = group_logical_docs(pages)
    print(f"\n  {len(logical_docs)} logical document(s) identified.")

    return [
        Document(
            text     = doc.text,
            metadata = {
                "file_id":    file_id,
                "file_name":  file_name,
                "doc_type":   doc.doc_type,
                "page_start": doc.page_start,
                "page_end":   doc.page_end,
            },
        )
        for doc in logical_docs
    ]


def load_pdfs(paths: List[str]) -> List[Document]:
    """Load and classify multiple PDF files."""
    all_docs: List[Document] = []
    for path in paths:
        all_docs.extend(load_pdf(path))
    return all_docs


# =============================================================================
# 9. INDEXING
# =============================================================================

def build_index(documents: List[Document]) -> VectorStoreIndex:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return VectorStoreIndex.from_documents(documents, transformations=[splitter])


# =============================================================================
# 10. HYBRID RETRIEVER
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

        seen:   Dict[str, NodeWithScore] = {}
        for node in vector_nodes + keyword_nodes:
            if node.node_id not in seen:
                seen[node.node_id] = node

        return sorted(
            seen.values(),
            key=lambda n: n.score if n.score is not None else 0.0,
            reverse=True,
        )[:self.top_k]


# =============================================================================
# 11. PIPELINE ASSEMBLY
# =============================================================================

def _make_engine(
    index: VectorStoreIndex,
    nodes: list,
    doc_type_filter: Optional[str] = None,
) -> RetrieverQueryEngine:
    safe_k = min(TOP_K, max(1, len(nodes)))

    if doc_type_filter:
        filters         = MetadataFilters(filters=[MetadataFilter(key="doc_type", value=doc_type_filter)])
        vector_retriever = index.as_retriever(similarity_top_k=safe_k, filters=filters)
    else:
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


def build_filtered_engine(
    index: VectorStoreIndex, doc_type: str
) -> Optional[RetrieverQueryEngine]:
    all_nodes = list(index.docstore.docs.values())
    filtered  = [n for n in all_nodes if n.metadata.get("doc_type") == doc_type]
    return _make_engine(index, filtered, doc_type_filter=doc_type) if filtered else None


# =============================================================================
# 12. QUERY ROUTING
# =============================================================================

def fast_route_by_keywords(query: str) -> Optional[str]:
    """Map query terms to a loaded doc_type — no LLM. Returns None if no match."""
    available = {m["doc_type"] for m in pdf_metadata_store}
    if not available:
        return None
    if len(available) == 1:
        return next(iter(available))

    lower = query.lower()
    for doc_type, keywords in _QUERY_ROUTING_RULES:
        if doc_type not in available:
            continue
        if any(kw in lower for kw in keywords):
            return doc_type
    return None


def resolve_query_doc_type(query: str) -> Optional[str]:
    """Optional pre-filter; None means search the full index (default)."""
    if not QUERY_DOC_TYPE_ROUTING:
        return None
    return fast_route_by_keywords(query)


def retrieve_files_by_doc_type(doc_type: str) -> list:
    return [m for m in pdf_metadata_store if m["doc_type"] == doc_type]


# =============================================================================
# 13. CLI ENTRY POINT
# =============================================================================

def main():
    pdf_path = input("PDF path: ").strip().strip('"')
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"\nLoading {os.path.basename(pdf_path)}…")
    docs   = load_pdf(pdf_path)
    index  = build_index(docs)
    engine = build_rag_pipeline(index)

    types = sorted({m["doc_type"] for m in pdf_metadata_store})
    print(f"Types detected: {', '.join(types)}")
    print('Ready — type your question or "exit" to quit.\n')

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        predicted = resolve_query_doc_type(query)
        matched   = retrieve_files_by_doc_type(predicted) if predicted else []
        q_engine  = build_filtered_engine(index, predicted) if matched else engine
        if QUERY_DOC_TYPE_ROUTING and predicted:
            print(f"  Routing to: {predicted} ({len(matched)} page(s))")
        else:
            print("  Searching full document index")

        response = q_engine.query(query)
        print(f"\n{response}\n")
        for node in response.source_nodes:
            m = node.metadata
            print(f"  [{m.get('doc_type')}  p.{m.get('page_start')}-{m.get('page_end')}  {m.get('file_name')}]")
        print("-" * 60)


if __name__ == "__main__":
    main()
