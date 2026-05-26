import os
import re
import time
import uuid
import logging
from collections import deque
from dotenv import load_dotenv
from typing import List, Optional, Dict
from dataclasses import dataclass

import contextlib
import sys

import fitz  # PyMuPDF
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# Silence noisy third-party loggers (httpx HuggingFace requests, sentence-transformers)
for _noisy in ("httpx", "huggingface_hub", "huggingface_hub.file_download", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# --- CONFIG -------------------------------------------------------------------

MODEL_PATH      = "content/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
TEMPERATURE     = 0.0
MAX_NEW_TOKENS  = 512
CONTEXT_WINDOW  = 2048
GPU_LAYERS      = 20
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 100
TOP_K           = 3

# --- Security -----------------------------------------------------------------

MAX_PDF_SIZE_MB   = 50
MAX_PDF_PAGES     = 300
MAX_QUERY_LENGTH  = 500
LLM_CALLS_PER_MIN = 30   # max LLM invocations per 60-second window

class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: int = 60):
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: deque = deque()

    def check(self):
        now = time.monotonic()
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max_calls:
            raise RuntimeError(
                f"Rate limit exceeded: max {self._max_calls} LLM calls per "
                f"{self._window}s. Please wait before retrying."
            )
        self._timestamps.append(now)

_llm_rate_limiter = RateLimiter(max_calls=LLM_CALLS_PER_MIN)

_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions?|"
    r"disregard\s+(all\s+)?prior\s+(instructions?|context)|"
    r"you\s+are\s+now\s+a|"
    r"new\s+instructions?:|"
    r"system\s*:\s*you|"
    r"<\s*/?(?:system|user|assistant|prompt|instruction)\s*>)",
    re.IGNORECASE,
)

def sanitize_text(text: str, max_chars: int = 2000) -> str:
    """Strip prompt-injection patterns and control characters from text."""
    # remove null bytes and non-printable control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = _INJECTION_PATTERNS.sub("[REMOVED]", text)
    return text[:max_chars]

def validate_pdf_path(pdf_path: str) -> None:
    real = os.path.realpath(pdf_path)
    if not os.path.isfile(real):
        raise ValueError(f"File not found: {pdf_path}")
    if not real.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {pdf_path}")
    size_mb = os.path.getsize(real) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(f"PDF too large ({size_mb:.1f} MB). Limit: {MAX_PDF_SIZE_MB} MB")

def validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long ({len(query)} chars). Limit: {MAX_QUERY_LENGTH}.")
    return sanitize_text(query, max_chars=MAX_QUERY_LENGTH)

@contextlib.contextmanager
def _quiet_llm():
    """Redirect C-level stdout so llama.cpp raw tokens don't bleed into the terminal."""
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(sys.stdout.fileno())
        os.dup2(null_fd, sys.stdout.fileno())
        try:
            yield
        finally:
            os.dup2(saved, sys.stdout.fileno())
            os.close(null_fd)
            os.close(saved)
    except Exception:
        yield  # fallback: if fd tricks fail, just proceed normally

# --- Model Setup --------------------------------------------------------------

from llama_index.llms.llama_cpp import LlamaCPP

llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=TEMPERATURE,
    max_new_tokens=MAX_NEW_TOKENS,
    context_window=CONTEXT_WINDOW,
    model_kwargs={"n_gpu_layers": GPU_LAYERS},
    verbose=False,
)

embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

Settings.llm = llm
Settings.embed_model = embed_model

# Kill llama.cpp's C-level log callback so nothing leaks to the terminal
try:
    import llama_cpp as _llama_cpp
    _llama_cpp.llama_log_set(lambda *_: None, None)
except Exception:
    pass

# Populated by load_pdf — used for metadata-based routing
pdf_metadata_store = []

@dataclass
class PageInfo:
    """Stores information about a single page"""
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0

@dataclass
class LogicalDocument:
    """A contiguous group of pages identified as one document"""
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    chunks: List[Dict] = None

# Document Ingestion & Extraction
# 1. PDF Input
def get_pdf_path() -> str:
    while True:
        pdf_path = input("Enter the path to your PDF file: ").strip()
        try:
            validate_pdf_path(pdf_path)
            return pdf_path
        except ValueError as e:
            print(f"{e}. Please try again.")
# 2. PDF Loading
def blob_reader(pdf_path: str) -> List[PageInfo]:
    if isinstance(pdf_path, dict) and "content" in pdf_path:
        doc = fitz.open(stream=pdf_path["content"], filetype="pdf")
    elif hasattr(pdf_path, "read"):
        doc = fitz.open(stream=pdf_path.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_path)
    
    if len(doc) > MAX_PDF_PAGES:
        doc.close()
        raise ValueError(f"PDF has {len(doc)} pages. Limit: {MAX_PDF_PAGES}.")

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            print(f"No text available, performing OCR")
            try:
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                from PIL import Image
                import pytesseract
                import io

                img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(img)
                print(f"  Page {i}: OCR extracted {len(text)} characters")
            except Exception as e:
                print(f"  Page {i}: OCR failed - {e}")
                text = ""
        pages.append(PageInfo(page_num=i, text=text))

    doc.close()
    return pages

_DOC_TYPES = [
    "Resume", "Contract", "Mortgage Contract", "Invoice", "Pay Slip",
    "Lender Fee Sheet", "Land Deed", "Bank Statement", "Tax Document",
    "Insurance", "Report", "Letter", "Form", "ID Document", "Medical", "Other",
]

def classify_doc_type(text: str) -> str:
    _llm_rate_limiter.check()
    text_sample = sanitize_text(text, max_chars=1500)

    prompt = (
        "You are a document classifier. Output exactly one category name — nothing else.\n\n"
        "Categories:\n"
        "Resume, Contract, Mortgage Contract, Invoice, Pay Slip, Lender Fee Sheet, "
        "Land Deed, Bank Statement, Tax Document, Insurance, Report, Letter, Form, "
        "ID Document, Medical, Other\n\n"
        "IMPORTANT DISTINCTIONS:\n"
        "- Lender Fee Sheet = a FEE SCHEDULE listing loan costs (origination fee, appraisal, "
        "  title insurance, discount points, APR, closing costs). It is NOT a contract. "
        "  Look for: 'Loan Estimate', 'Closing Disclosure', 'Good Faith Estimate', "
        "  'origination charges', 'APR', 'discount points', 'lender credits', 'title service fees'.\n"
        "- Contract = a legal agreement with parties, obligations, clauses. "
        "  Look for: 'AGREEMENT', 'WHEREAS', 'IN WITNESS WHEREOF', 'the parties agree', "
        "  'terms and conditions', 'obligations'.\n"
        "- Mortgage Contract = a HOME LOAN agreement (not a fee sheet). "
        "  Look for: 'mortgage', 'promissory note', 'deed of trust', 'borrower', 'principal'.\n\n"
        "Examples:\n"
        'Sample: "EMPLOYMENT AGREEMENT entered into between..." -> Contract\n'
        'Sample: "ANNEXURE A forms part of the Agreement dated..." -> Contract\n'
        'Sample: "LEASE AGREEMENT between Landlord and Tenant..." -> Contract\n'
        'Sample: "NON-DISCLOSURE AGREEMENT parties agree to keep confidential..." -> Contract\n'
        'Sample: "Loan Estimate  Origination Charges $1,500  Appraisal Fee $450  APR 6.75%..." -> Lender Fee Sheet\n'
        'Sample: "Closing Disclosure  Total Loan Costs  Discount Points 0.5%  Lender Credits..." -> Lender Fee Sheet\n'
        'Sample: "Good Faith Estimate  Loan origination fee 1%  Credit report $35  Title insurance $800..." -> Lender Fee Sheet\n'
        'Sample: "Lender Fee Worksheet  Processing fee $500  Underwriting $895  Recording fee $125..." -> Lender Fee Sheet\n'
        'Sample: "PAY STUB  Employee: John Smith  Gross Pay: $3,200  Net Pay: $2,450..." -> Pay Slip\n'
        'Sample: "MORTGAGE AGREEMENT  This Home Loan is made between Borrower and Lender..." -> Mortgage Contract\n\n'
        f"Document sample:\n{text_sample}\n\n"
        "Category:"
    )
    with _quiet_llm():
        response = str(llm.complete(prompt)).strip().split('\n')[0]
    # Longest names first so "Mortgage Contract" is checked before "Contract"
    for known in sorted(_DOC_TYPES, key=len, reverse=True):
        if known.lower() in response.lower():
            return known
    return "Other"

def is_same_document(prev_text: str, curr_text: str, doc_type: str = None) -> bool:
    if not prev_text or not curr_text:
        return False

    _llm_rate_limiter.check()
    prev_sample = sanitize_text(prev_text[-500:] if len(prev_text) > 500 else prev_text, max_chars=500)
    curr_sample = sanitize_text(curr_text[:500] if len(curr_text) > 500 else curr_text, max_chars=500)

    prompt = (
        "You are a document boundary detector. "
        f"The current document type is: {doc_type or 'Unknown'}.\n\n"
        f"End of previous page:\n...{prev_sample}\n\n"
        f"Start of current page:\n{curr_sample}...\n\n"
        "Do these two pages belong to the same document? "
        "Reply with a single word — Yes or No — and nothing else."
    )
    with _quiet_llm():
        response = str(llm.complete(prompt)).strip()
    return response.lower().startswith('yes')

def group_logical_docs(metadata_store: list) -> List[LogicalDocument]:
    logical_docs = []
    doc_counter = 0
    current_pages = []
    current_doc_type = None

    for page in metadata_store:
        if page["page_in_doc"] == 0 and current_pages:
            logical_docs.append(LogicalDocument(
                doc_id=f"doc_{doc_counter}",
                doc_type=current_doc_type,
                page_start=current_pages[0]["page"],
                page_end=current_pages[-1]["page"],
                text="\n\n".join(p["text"] for p in current_pages),
            ))
            doc_counter += 1
            current_pages = []

        current_pages.append(page)
        current_doc_type = page["doc_type"]

    if current_pages:
        logical_docs.append(LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=current_doc_type,
            page_start=current_pages[0]["page"],
            page_end=current_pages[-1]["page"],
            text="\n\n".join(p["text"] for p in current_pages),
        ))

    return logical_docs


def load_pdf(pdf_path: str) -> List[Document]:
    global pdf_metadata_store
    pdf_metadata_store = []

    doc_pages = blob_reader(pdf_path)
    total_pages = len(doc_pages)
    file_name = os.path.basename(pdf_path)
    file_id = str(uuid.uuid4())

    current_doc_type = None
    page_in_doc = 0
    prev_text = None

    for page in doc_pages:
        if not page.text.strip():
            continue
        if prev_text is None:
            current_doc_type = classify_doc_type(page.text)
            page_in_doc = 0
        elif is_same_document(prev_text, page.text, current_doc_type):
            page_in_doc += 1
        else:
            current_doc_type = classify_doc_type(page.text)
            page_in_doc = 0

        print(f"  Page {page.page_num}/{total_pages} | {current_doc_type:<18} | page_in_doc: {page_in_doc}")

        pdf_metadata_store.append({
            "page":        page.page_num,
            "text":        page.text,
            "doc_type":    current_doc_type,
            "page_in_doc": page_in_doc,
            "file_id":     file_id,
            "file_name":   file_name,
        })
        prev_text = page.text

    # Summary table
    print(f"\n{'page':>5}  {'doc_type':<18} {'page_in_doc'}")
    print("-" * 40)
    for m in pdf_metadata_store:
        print(f"  {m['page']:>3}  {m['doc_type']:<18} {m['page_in_doc']}")

    logical_docs = group_logical_docs(pdf_metadata_store)
    print(f"\n  {len(logical_docs)} logical document(s) identified.\n")

    documents = []
    for doc in logical_docs:
        metadata = {
            "file_id":    file_id,
            "file_name":  file_name,
            "doc_type":   doc.doc_type,
            "page_start": doc.page_start,
            "page_end":   doc.page_end,
        }
        documents.append(Document(text=doc.text, metadata=metadata))

    return documents


# --- 3. Indexing --------------------------------------------------------------

def build_index(documents: List[Document]) -> VectorStoreIndex:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return VectorStoreIndex.from_documents(documents, transformations=[splitter])

# --- 4. Hybrid Retriever ------------------------------------------------------

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever, top_k: int = 2):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        keyword_nodes = self.keyword_retriever.retrieve(query_bundle)
        unique_nodes = {}
        for node in vector_nodes + keyword_nodes:
            if node.node_id not in unique_nodes:
                unique_nodes[node.node_id] = node
        return sorted(
            unique_nodes.values(),
            key=lambda x: x.score if hasattr(x, "score") else 0.0,
            reverse=True
        )[:self.top_k]


# --- 5. Pipeline Assembly -----------------------------------------------------

def build_rag_pipeline(index: VectorStoreIndex) -> RetrieverQueryEngine:
    nodes = list(index.docstore.docs.values())
    num_nodes = len(nodes)
    safe_top_k = min(TOP_K, max(1, num_nodes))

    vector_retriever = index.as_retriever(similarity_top_k=safe_top_k)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=safe_top_k)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, top_k=safe_top_k)

    node_postprocessors = []
    if num_nodes > 1:
        try:
            node_postprocessors = [SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=min(TOP_K, num_nodes)
            )]
        except Exception as e:
            print(f"Reranker unavailable (skipping): {e}")

    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        node_postprocessors=node_postprocessors
    )


# --- 6. Query Routing ---------------------------------------------------------


def predict_query_doc_type(query: str) -> str:
    _llm_rate_limiter.check()
    available = list({m["doc_type"] for m in pdf_metadata_store})
    if not available:
        return "Other"
    categories = ", ".join(available)
    safe_query = sanitize_text(query, max_chars=300)
    prompt = (
        "You are a document router. Given the query below, pick the single most "
        "relevant document category from the available list.\n\n"
        f"Query: {safe_query}\n\n"
        f"Available categories: {categories}\n\n"
        "Reply with one category name only — no explanation, no punctuation, no code.\n"
        "Answer:"
    )
    with _quiet_llm():
        response = str(llm.complete(prompt)).strip().split('\n')[0]
    for known in sorted(_DOC_TYPES, key=len, reverse=True):
        if known.lower() in response.lower():
            return known
    return "Other"


def retrieve_files_by_doc_type(doc_type: str) -> list:
    return [m for m in pdf_metadata_store if m["doc_type"] == doc_type]

def load_pdfs(paths: List[str]) -> List[Document]:
    """Load and classify multiple PDF files, merging their metadata stores."""
    all_docs = []
    for path in paths:
        all_docs.extend(load_pdf(path))
    return all_docs

def build_filtered_engine(index: VectorStoreIndex, doc_type: str) -> RetrieverQueryEngine:
    all_nodes = list(index.docstore.docs.values())
    filtered_nodes = [n for n in all_nodes if n.metadata.get("doc_type") == doc_type]
    if not filtered_nodes:
        return None

    safe_top_k = min(TOP_K, max(1, len(filtered_nodes)))
    filters = MetadataFilters(filters=[MetadataFilter(key="doc_type", value=doc_type)])
    vector_retriever = index.as_retriever(similarity_top_k=safe_top_k, filters=filters)
    bm25_retriever = BM25Retriever.from_defaults(nodes=filtered_nodes, similarity_top_k=safe_top_k)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, top_k=safe_top_k)

    return RetrieverQueryEngine.from_args(retriever=hybrid_retriever, llm=llm)


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    pdf_path = get_pdf_path()
    pages = load_pdf(pdf_path)
    index = build_index(pages)
    engine = build_rag_pipeline(index)

    print('\nChat with the RAG engine. Type "exit" to quit.\n')

    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        try:
            user_query = validate_query(user_query)
        except ValueError as e:
            print(f"  Invalid query: {e}")
            continue

        predicted_type = predict_query_doc_type(user_query)
        matched = retrieve_files_by_doc_type(predicted_type)
        print(f"  Routing to: {predicted_type} ({len(matched)} document(s) matched)")

        query_engine = build_filtered_engine(index, predicted_type) if matched else None
        if query_engine is None:
            print("  No matching documents — searching all.")
            query_engine = engine

        response = query_engine.query(user_query)
        print(f"\n{response}\n")
        for node in response.source_nodes:
            m = node.metadata
            print(f"  [p.{m.get('page_start')}-{m.get('page_end')} | {m.get('doc_type')} | {m.get('file_name')}]")
        print("\n----------------------\n")
