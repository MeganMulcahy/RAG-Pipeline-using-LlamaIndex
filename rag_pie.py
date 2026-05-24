import os
import uuid
import logging
from dotenv import load_dotenv
from typing import List

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from llama_index.readers.file import PDFReader

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

# --- CONFIG -------------------------------------------------------------------

MODEL_PATH      = "content/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
TEMPERATURE     = 0.7
MAX_NEW_TOKENS  = 512
CONTEXT_WINDOW  = 2048
GPU_LAYERS      = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 100
TOP_K           = 3

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

# Populated by load_pdf — used for metadata-based routing
pdf_metadata_store = []


# --- 1. PDF Input -------------------------------------------------------------

def get_pdf_path() -> str:
    while True:
        pdf_path = input("Enter the path to your PDF file: ").strip()
        if not os.path.isfile(pdf_path):
            print(f"File not found: {pdf_path}. Please try again.")
        elif not pdf_path.lower().endswith(".pdf"):
            print(f"Not a PDF file: {pdf_path}. Please try again.")
        else:
            return pdf_path


# --- 2. PDF Loading -----------------------------------------------------------

def blob_reader(pdf_path: str) -> list:
    loader = PDFReader()
    raw_pages = loader.load_data(pdf_path)
    return [{"page_num": i + 1, "text": doc.text} for i, doc in enumerate(raw_pages)]


_DOC_TYPES = ["Resume", "Contract", "Fees Worksheet", "ID", "PaySlip", "W2", "Other"]

# Keyword rules applied directly to page text — much more reliable than parsing LLM output
_DOC_TYPE_RULES = {
    "ID":             ["driver's license", "driver license", "passport",
                       "date of birth", "photo id", "identification card"],
    "W2":             ["w-2", "wage and tax statement", "wages, tips", "form w2"],
    "Fees Worksheet": ["fees worksheet", "fee worksheet", "lender fee", "closing cost",
                       "cost sheet", "estimated settlement", "loan estimate", "closing disclosure"],
    "PaySlip":        ["pay stub", "payslip", "pay slip", "earnings statement",
                       "gross pay", "net pay", "year-to-date", "ytd earnings"],
    "Contract":       ["contract", "hereby agrees", "terms and conditions",
                       "entered into", "this agreement"],
    "Resume":         ["curriculum vitae", "career summary", "work experience",
                       "employment history", "objective statement"],
}


def _rule_based_classify(text: str) -> str:
    """Keyword-only check on first 400 chars. Returns 'Other' if no match."""
    header = text[:400].lower()
    for doc_type, keywords in _DOC_TYPE_RULES.items():
        if any(kw in header for kw in keywords):
            return doc_type
    return "Other"


def is_same_document(prev_text: str, curr_text: str, doc_type: str = None) -> bool:
    curr_type = _rule_based_classify(curr_text)

    # Page header clearly matches a different known type → new document
    if curr_type != "Other" and curr_type != doc_type:
        return False

    # No keywords in header → body/continuation text, treat as same document
    if curr_type == "Other":
        return True

    # Header matches the same type — use LLM only to distinguish two separate
    # docs of the same type (e.g. two back-to-back contracts) from a true continuation
    prompt = (
        f"Both pages are '{doc_type}' documents.\n\n"
        f"End of previous page:\n{prev_text[-300:]}\n\n"
        f"Beginning of next page:\n{curr_text[:300]}\n\n"
        "Is the next page a continuation of the SAME document, or the start of a NEW separate document?\n"
        "Answer ONLY Yes (same document) or No (new document)."
    )
    response = str(llm.complete(prompt)).strip().lower()
    return response.startswith("yes")


def classify_doc_type(text: str) -> str:
    lower = text.lower()
    for doc_type, keywords in _DOC_TYPE_RULES.items():
        if any(kw in lower for kw in keywords):
            return doc_type
    return _llm_classify(text)


def _llm_classify(text: str) -> str:
    prompt = (
        "What type of document does this text belong to?\n"
        "Reply with ONE of these exact types only:\n"
        "Resume | Contract | Fees Worksheet | ID | PaySlip | W2 | Other\n\n"
        f"Text:\n{text[:400]}\n\nType:"
    )
    raw = str(llm.complete(prompt)).strip().split("\n")[0]
    for known in _DOC_TYPES:
        if known.lower() in raw.lower():
            return known
    return "Other"


def group_logical_docs(metadata_store: list, source_file: str) -> list:
    logical_docs = []
    current_doc = {"text": "", "doc_type": None, "page_start": 0, "source_file": source_file}

    for page in metadata_store:
        if page["page_in_doc"] == 0 and current_doc["text"]:
            current_doc["page_end"] = page["page"] - 1
            logical_docs.append(current_doc)
            current_doc = {"text": "", "doc_type": None, "page_start": page["page"], "source_file": source_file}

        current_doc["text"] += "\n\n" + page["text"]
        current_doc["doc_type"] = page["doc_type"]

    if metadata_store:
        current_doc["page_end"] = metadata_store[-1]["page"]
    logical_docs.append(current_doc)
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
        if not page["text"].strip():
            continue

        if prev_text is None:
            current_doc_type = classify_doc_type(page["text"])
            page_in_doc = 0
        elif is_same_document(prev_text, page["text"], current_doc_type):
            page_in_doc += 1
        else:
            current_doc_type = classify_doc_type(page["text"])
            page_in_doc = 0

        print(f"  Page {page['page_num']}/{total_pages} | {current_doc_type:<18} | page_in_doc: {page_in_doc}")

        pdf_metadata_store.append({
            "page":        page["page_num"],
            "text":        page["text"],
            "doc_type":    current_doc_type,
            "page_in_doc": page_in_doc,
            "file_id":     file_id,
            "file_name":   file_name,
        })
        prev_text = page["text"]

    # Summary table
    print(f"\n{'page':>5}  {'doc_type':<18} {'page_in_doc'}")
    print("-" * 40)
    for m in pdf_metadata_store:
        print(f"  {m['page']:>3}  {m['doc_type']:<18} {m['page_in_doc']}")

    logical_docs = group_logical_docs(pdf_metadata_store, file_name)
    print(f"\n  {len(logical_docs)} logical document(s) identified.\n")

    documents = []
    for doc in logical_docs:
        metadata = {
            "file_id":    file_id,
            "file_name":  file_name,
            "doc_type":   doc["doc_type"],
            "page_start": doc["page_start"],
            "page_end":   doc["page_end"],
        }
        documents.append(Document(text=doc["text"], metadata=metadata))

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
    """Predict which doc type a query is targeting."""
    available = list({m["doc_type"] for m in pdf_metadata_store})
    prompt = (
        f'User query: "{query}"\n\n'
        f"Available document types: {available}\n\n"
        f"Which type is this query about? Reply with one of: {' | '.join(_DOC_TYPES)}\n"
        "Type only, no explanation."
    )
    raw = str(llm.complete(prompt)).strip()
    for known in _DOC_TYPES:
        if known.lower() in raw.lower():
            return known
    return "Other"


def retrieve_files_by_doc_type(doc_type: str) -> list:
    return [m for m in pdf_metadata_store if m["doc_type"] == doc_type]


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
