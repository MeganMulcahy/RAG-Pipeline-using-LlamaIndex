# Document RAG Pipeline

A RAG (Retrieval-Augmented Generation) pipeline for querying PDF documents using natural language.

## What It Does

Upload a PDF and ask plain-English questions. The system extracts text, indexes it, and returns concise, grounded answers.

## Tech Stack

| Component | Tool |
|---|---|
| **LLM** | Google Gemini (via `google.genai`) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **RAG Framework** | LlamaIndex |
| **PDF Extraction** | PyMuPDF (`fitz`), `pymupdf4llm`, `unstructured`, Tesseract OCR |
| **Vector Search** | LlamaIndex `VectorStoreIndex` |
| **Keyword Search** | BM25 Retriever |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (SentenceTransformer) |
| **Retrieval Strategy** | Hybrid (vector + BM25) with reranking |

## How It Works

1. **PDF ingestion** — two-pass extract: fitz classifies each page (`native_text` / `image_dominant` / `table_heavy`), then routes to batched `pymupdf4llm`, parallel `unstructured`, or OCR
2. **Segmentation** — heuristic boundaries group pages into logical segments (no domain-specific doc labels)
3. **Indexing** — segments are chunked, embedded, and stored in a vector index
4. **Hybrid retrieval** — queries hit both semantic (vector) and keyword (BM25) search, results are merged and deduplicated
5. **Reranking** — a cross-encoder reranker scores and re-orders the top chunks
6. **Generation** — Gemini synthesizes a single concise answer from the retrieved context

## Setup

1. Install dependencies:
   ```bash
   python lib.py
   ```

2. Create a `.env` file:
   ```
   GEMINI_API_KEY=your_key_here
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   MAX_EXTRACT_WORKERS=4
   QUERY_REWRITE=true
   DATA_DIR=./data/rag
   ```

3. Install Tesseract OCR (required for scanned/image-heavy pages):
   - Windows (recommended): `winget install UB-Mannheim.TesseractOCR`
   - Or download from: https://github.com/UB-Mannheim/tesseract/wiki
   - If not on PATH, set in `.env`:
     ```
     TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
     ```

4. Run:
   ```bash
   python rag_pipeline.py
   ```
   Enter a PDF path when prompted, then ask questions. Type `exit` to quit.

### Inspect ingest output (sample pages)

To see what the pipeline actually extracts (1 text page, 3 table pages, 3 scanned/OCR pages):

```bash
python test_ingest_sample.py "your.pdf" --full -o ingest_sample_report.txt
```

Requires `TESSERACT_CMD` in `.env` for `image_dominant` pages.

## Architecture Progress Tracker

| Date | Status | Details |
|---|---|---|
| 2026-05-28 | Pipeline v1 | PDF extraction (PyMuPDF + Tesseract OCR fallback); classification + boundary detection + logical document grouping; chunking (`SentenceSplitter`); hybrid retrieval (Vector + BM25); reranking (cross-encoder `ms-marco-MiniLM-L-6-v2`); generation (Gemini via LlamaIndex); optional query routing |
| 2026-05-28 | Ingest refactor (current) | Two-pass ingest + heuristic segmentation + `page_manifest`; chunk dedup; query rewrite (typos); full-index hybrid retrieval + rerank + Gemini (`google.genai`) |
| 2026-05-28 | Phase 2 (partial) | `storage.py`: local blob store (`./data/rag/blobs`), SQLite metadata (jobs, versions, chunk lineage); optional MinIO + Redis via env; sync ingest wired in CLI |
| Planned | Step 4 | Phase 2 completion: Postgres metadata, background ingest worker, MinIO in prod |
| Planned | Step 5 | Retrieval/API scale-up: dedicated vector DB, sparse service, REST auth, multi-tenant knowledge bases |
