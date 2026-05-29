# AI Mortgage Document Intelligence Automation System

A RAG (Retrieval-Augmented Generation) pipeline for querying PDF documents using natural language.

## What It Does

Upload a PDF and ask plain-English questions. The system extracts text, indexes it, and returns concise, grounded answers.

## Tech Stack

| Component | Tool |
|---|---|
| **LLM** | Google Gemini (via `llama-index-llms-google-genai`)|
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **RAG Framework** | LlamaIndex |
| **PDF Extraction** | PyMuPDF (`fitz`) |
| **Vector Search** | LlamaIndex `VectorStoreIndex` |
| **Keyword Search** | BM25 Retriever |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (SentenceTransformer) |
| **Retrieval Strategy** | Hybrid (vector + BM25) with reranking |

## How It Works

1. **PDF ingestion** — PyMuPDF extracts text page by page
2. **Indexing** — pages are embedded and stored in a vector index
3. **Hybrid retrieval** — queries hit both semantic (vector) and keyword (BM25) search, results are merged and deduplicated
4. **Reranking** — a cross-encoder reranker scores and re-orders the top chunks
5. **Generation** — Gemini synthesizes a single concise answer from the retrieved context

## Setup

1. Install dependencies:
   ```bash
   python lib.py
   ```

2. Create a `.env` file:
   ```
   GEMINI_API_KEY=your_key_here
   QUERY_DOC_TYPE_ROUTING=false
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

## Architecture Progress Tracker

| Date | Status | Details |
|---|---|---|
| 2026-05-28 | Current Pipeline | PDF extraction (PyMuPDF + Tesseract OCR fallback); heuristic segmentation + logical document grouping; chunking (`SentenceSplitter`); hybrid retrieval (Vector + BM25); reranking (cross-encoder `ms-marco-MiniLM-L-6-v2`); generation (Gemini via LlamaIndex) |
| Planned | Step 1 | Multi-format support beyond PDF (Word, Excel, slides, web pages) |
| Planned | Step 2 | Document layout analysis before chunking (separate headers, tables, figures) |
| Planned | Step 3 | Table recognition/extraction as structured data |
| Planned | Step 4 | Storage/serving layer: object storage + metadata DB + async ingestion queue |
| Planned | Step 5 | Retrieval/API scale-up: dedicated vector DB, sparse service, REST auth, multi-tenant knowledge bases |
