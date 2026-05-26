import os
import re
import time
import uuid
import logging
from collections import deque
from dotenv import load_dotenv
from typing import List, Optional, Dict

import fitz  # PyMuPDF
import google.generativeai as genai
from dataclasses import dataclass
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv()

for _noisy in ("httpx", "huggingface_hub", "huggingface_hub.file_download", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# --- CONFIG -------------------------------------------------------------------

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = "models/gemini-3.1-flash-lite"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 100
TOP_K             = 4
MAX_PDF_SIZE_MB   = 50
MAX_PDF_PAGES     = 300
MAX_QUERY_LEN     = 500
LLM_CALLS_PER_MIN = 60

# --- Gemini Setup -------------------------------------------------------------

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
_gemini = genai.GenerativeModel(GEMINI_MODEL)

embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
Settings.embed_model = embed_model

class _GeminiLLM(CustomLLM):
    """Thin LlamaIndex-compatible wrapper around google.generativeai."""
    context_window: int = 8192
    num_output: int = 512
    model_name: str = GEMINI_MODEL

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **_) -> CompletionResponse:
        _rate_limiter.check()
        text = _gemini.generate_content(prompt).text or ""
        return CompletionResponse(text=text.strip())

    def stream_complete(self, prompt: str, **_):
        yield self.complete(prompt)

_llm = _GeminiLLM()
Settings.llm = _llm

# --- Security -----------------------------------------------------------------

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
                f"Rate limit exceeded: max {self._max_calls} calls per {self._window}s."
            )
        self._timestamps.append(now)

_rate_limiter = RateLimiter(max_calls=LLM_CALLS_PER_MIN)

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
    if len(query) > MAX_QUERY_LEN:
        raise ValueError(f"Query too long ({len(query)} chars). Limit: {MAX_QUERY_LEN}.")
    return sanitize_text(query, max_chars=MAX_QUERY_LEN)

# --- Dataclasses --------------------------------------------------------------

@dataclass
class PageInfo:
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0

@dataclass
class LogicalDocument:
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    chunks: Optional[List[Dict]] = None

_DOC_TYPES = [
    "Resume", "Contract", "Mortgage Contract", "Invoice", "Pay Slip",
    "Lender Fee Sheet", "Land Deed", "Bank Statement", "Tax Document",
    "Insurance", "Report", "Letter", "Form", "ID Document", "Medical", "Other",
]

# --- Gemini helper ------------------------------------------------------------

def _gemini_complete(prompt: str) -> str:
    _rate_limiter.check()
    try:
        return _gemini.generate_content(prompt).text.strip()
    except Exception as e:
        logging.warning(f"Gemini call failed: {e}")
        return ""

# --- PDF Extraction -----------------------------------------------------------

def blob_reader(pdf_path) -> List[PageInfo]:
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
            try:
                from PIL import Image
                import pytesseract
                import io
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            except Exception as e:
                logging.warning(f"Page {i}: OCR failed - {e}")
                text = ""
        pages.append(PageInfo(page_num=i, text=text))

    doc.close()
    return pages

# --- Classification & Boundary Detection -------------------------------------

def classify_document_type(text: str) -> str:
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
        "  Look for: 'AGREEMENT', 'WHEREAS', 'IN WITNESS WHEREOF', 'the parties agree'.\n"
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
    response = _gemini_complete(prompt)
    for known in sorted(_DOC_TYPES, key=len, reverse=True):
        if known.lower() in response.lower():
            return known
    return "Other"

def detect_document_boundary(prev_text: str, curr_text: str, doc_type: str = None) -> bool:
    if not prev_text or not curr_text:
        return False
    prev_sample = sanitize_text(prev_text[-500:], max_chars=500)
    curr_sample = sanitize_text(curr_text[:500], max_chars=500)
    prompt = (
        "You are a document boundary detector. "
        f"The current document type is: {doc_type or 'Unknown'}.\n\n"
        f"End of previous page:\n...{prev_sample}\n\n"
        f"Start of current page:\n{curr_sample}...\n\n"
        "Do these two pages belong to the same document? "
        "Reply with a single word — Yes or No — and nothing else."
    )
    response = _gemini_complete(prompt)
    return response.lower().startswith("yes")

# --- Logical Document Grouping ------------------------------------------------

def group_logical_docs(pages: List[PageInfo]) -> List[LogicalDocument]:
    logical_docs = []
    current_pages: List[PageInfo] = []
    current_type = None
    doc_counter = 0

    for page in pages:
        if not page.text.strip():
            continue
        if page.page_in_doc == 0 and current_pages:
            logical_docs.append(LogicalDocument(
                doc_id=f"doc_{doc_counter}",
                doc_type=current_type,
                page_start=current_pages[0].page_num,
                page_end=current_pages[-1].page_num,
                text="\n\n".join(p.text for p in current_pages),
            ))
            doc_counter += 1
            current_pages = []
        current_pages.append(page)
        current_type = page.doc_type

    if current_pages:
        logical_docs.append(LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=current_type,
            page_start=current_pages[0].page_num,
            page_end=current_pages[-1].page_num,
            text="\n\n".join(p.text for p in current_pages),
        ))
    return logical_docs

# --- PDF Loading --------------------------------------------------------------

pdf_metadata_store: List[Dict] = []

def load_pdf(pdf_path) -> List[Document]:
    global pdf_metadata_store
    pdf_metadata_store = []

    pages = blob_reader(pdf_path)
    file_name = (
        os.path.basename(pdf_path.name) if hasattr(pdf_path, "name")
        else os.path.basename(str(pdf_path))
    )
    file_id = str(uuid.uuid4())
    total_pages = len(pages)

    current_type = None
    page_in_doc = 0
    prev_text = None

    for page in pages:
        if not page.text.strip():
            continue
        if prev_text is None:
            current_type = classify_document_type(page.text)
            page_in_doc = 0
        elif detect_document_boundary(prev_text, page.text, current_type):
            page_in_doc += 1
        else:
            current_type = classify_document_type(page.text)
            page_in_doc = 0

        page.doc_type = current_type
        page.page_in_doc = page_in_doc

        print(f"  Page {page.page_num}/{total_pages} | {current_type:<18} | page_in_doc: {page_in_doc}")

        pdf_metadata_store.append({
            "page":        page.page_num,
            "text":        page.text,
            "doc_type":    current_type,
            "page_in_doc": page_in_doc,
            "file_id":     file_id,
            "file_name":   file_name,
        })
        prev_text = page.text

    logical_docs = group_logical_docs(pages)
    print(f"\n  {len(logical_docs)} logical document(s) identified.\n")

    documents = []
    for doc in logical_docs:
        documents.append(Document(
            text=doc.text,
            metadata={
                "file_id":    file_id,
                "file_name":  file_name,
                "doc_type":   doc.doc_type,
                "page_start": doc.page_start,
                "page_end":   doc.page_end,
            },
        ))
    return documents

def load_pdfs(paths: List[str]) -> List[Document]:
    all_docs = []
    for path in paths:
        all_docs.extend(load_pdf(path))
    return all_docs

# --- Index & Hybrid Retriever ------------------------------------------------

def build_index(documents: List[Document]) -> VectorStoreIndex:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return VectorStoreIndex.from_documents(documents, transformations=[splitter])

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever, top_k: int = TOP_K):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        keyword_nodes = self.bm25_retriever.retrieve(query_bundle)
        unique: Dict[str, NodeWithScore] = {}
        for node in vector_nodes + keyword_nodes:
            if node.node_id not in unique:
                unique[node.node_id] = node
        return sorted(
            unique.values(),
            key=lambda x: x.score if x.score is not None else 0.0,
            reverse=True,
        )[:self.top_k]

def _make_engine(index: VectorStoreIndex, nodes, doc_type: str = None) -> RetrieverQueryEngine:
    safe_k = min(TOP_K, max(1, len(nodes)))
    if doc_type:
        filters = MetadataFilters(filters=[MetadataFilter(key="doc_type", value=doc_type)])
        vector_retriever = index.as_retriever(similarity_top_k=safe_k, filters=filters)
    else:
        vector_retriever = index.as_retriever(similarity_top_k=safe_k)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=safe_k)
    hybrid = HybridRetriever(vector_retriever, bm25_retriever, top_k=safe_k)

    postprocessors = []
    if len(nodes) > 1:
        try:
            postprocessors = [SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=min(TOP_K, len(nodes)),
            )]
        except Exception as e:
            logging.warning(f"Reranker unavailable (skipping): {e}")

    kwargs = dict(retriever=hybrid, node_postprocessors=postprocessors)
    if _llm:
        kwargs["llm"] = _llm
    return RetrieverQueryEngine.from_args(**kwargs)

def build_rag_pipeline(index: VectorStoreIndex) -> RetrieverQueryEngine:
    nodes = list(index.docstore.docs.values())
    return _make_engine(index, nodes)

def build_filtered_engine(index: VectorStoreIndex, doc_type: str) -> Optional[RetrieverQueryEngine]:
    all_nodes = list(index.docstore.docs.values())
    filtered = [n for n in all_nodes if n.metadata.get("doc_type") == doc_type]
    if not filtered:
        return None
    return _make_engine(index, filtered, doc_type=doc_type)

# --- Query Routing ------------------------------------------------------------

def predict_query_doc_type(query: str) -> str:
    available = list({m["doc_type"] for m in pdf_metadata_store})
    if not available:
        return "Other"
    safe_query = sanitize_text(query, max_chars=300)
    prompt = (
        "You are a document router. Given the query below, pick the single most "
        "relevant document category from the available list.\n\n"
        f"Query: {safe_query}\n\n"
        f"Available categories: {', '.join(available)}\n\n"
        "Reply with one category name only — no explanation, no punctuation, no code.\n"
        "Answer:"
    )
    response = _gemini_complete(prompt)
    for known in sorted(_DOC_TYPES, key=len, reverse=True):
        if known.lower() in response.lower():
            return known
    return "Other"

def retrieve_files_by_doc_type(doc_type: str) -> list:
    return [m for m in pdf_metadata_store if m["doc_type"] == doc_type]

# --- Main ---------------------------------------------------------------------

def main():
    print("\n=== Document Q&A ===\n")
    pdf_path = input("Enter path to PDF file: ").strip().strip('"')
    try:
        validate_pdf_path(pdf_path)
    except ValueError as e:
        print(e)
        return

    from datetime import datetime
    start = datetime.now()
    documents = load_pdf(pdf_path)
    index     = build_index(documents)
    engine    = build_rag_pipeline(index)
    elapsed   = f"{(datetime.now() - start).total_seconds():.1f}s"

    doc_types = list({m["doc_type"] for m in pdf_metadata_store})
    print(f"\nProcessed: {os.path.basename(pdf_path)}")
    print(f"  Pages:   {len(pdf_metadata_store)}")
    print(f"  Types:   {', '.join(doc_types)}")
    print(f"  Time:    {elapsed}")
    print('\nReady. Type your question or "exit" to quit.\n')

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        try:
            question = validate_query(question)
        except ValueError as e:
            print(f"  {e}")
            continue

        predicted_type = predict_query_doc_type(question)
        matched        = retrieve_files_by_doc_type(predicted_type)
        q_engine       = build_filtered_engine(index, predicted_type) if matched else engine

        response = q_engine.query(question)
        print(f"\n{response}\n")
        for node in response.source_nodes:
            m = node.metadata
            print(f"  [{m.get('doc_type')}  p.{m.get('page_start')}-{m.get('page_end')}  {m.get('file_name')}]")
        print("-" * 60)


if __name__ == "__main__":
    main()
