# Document RAG Pipeline

A RAG (Retrieval-Augmented Generation) pipeline for querying documents using natural language.

## What It Does

Upload a document and ask plain-English questions. The system extracts text, indexes it, and returns concise, grounded answers.

**Supported uploads:** PDF (two-pass page pipeline), plus Word, Excel, PowerPoint, HTML, plain text, CSV, and images via `unstructured` — most common office formats work out of the box.

## Tech Stack

| Component | Tool |
|---|---|
| **LLM** | Google Gemini (via `google.genai`) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **RAG Framework** | LlamaIndex |
| **PDF Extraction** | PyMuPDF (`fitz`), `pymupdf4llm`, `unstructured`, Tesseract OCR |
| **Other formats** | `unstructured.partition.auto` (docx, xlsx, pptx, html, txt, images, …) |
| **Vector Search** | LlamaIndex `VectorStoreIndex` |
| **Keyword Search** | BM25 Retriever |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (SentenceTransformer) |
| **Retrieval Strategy** | Hybrid (vector + BM25) with reranking |

## How It Works

1. **Ingestion**
   - **PDF** — two-pass extract: fitz classifies each page (`native_text` / `image_dominant` / `table_heavy`), then routes to batched `pymupdf4llm`, parallel `unstructured`, or OCR
   - **Other files** — single-pass `unstructured` extract, then one logical segment
2. **Segmentation** — heuristic boundaries group PDF pages into logical segments (no domain-specific doc labels)
3. **Indexing** — segments are chunked, embedded, and stored in a vector index for the session
4. **Hybrid retrieval** — queries hit both semantic (vector) and keyword (BM25) search, results are merged and deduplicated
5. **Reranking** — a cross-encoder reranker scores and re-orders the top chunks
6. **Generation** — Gemini synthesizes a single concise answer from the retrieved context

### Retrieval & caching

| Step | What happens |
|------|----------------|
| Query rewrite | Gemini normalizes typos before search (`QUERY_REWRITE=true`) |
| Hybrid retrieve | Vector top-K + BM25 top-K, merge, dedupe overlapping chunks |
| Rerank + answer | Cross-encoder, then Gemini on retrieved chunks only |

| Cache / dedup layer | What it does |
|---------------------|----------------|
| Chunk dedup at index | Same segment + similar text indexed once (`dedupe_nodes`) |
| Hybrid merge dedup | Vector + BM25 hits collapsed by page/segment key |
| Source dedup | Citation lines not repeated in CLI output |
| Session index | Embeddings + BM25 stay in memory until you exit the CLI |

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

3. Install Tesseract OCR (required for scanned PDF pages and some images):
   - Windows (recommended): `winget install UB-Mannheim.TesseractOCR`
   - Or download from: https://github.com/UB-Mannheim/tesseract/wiki
   - If not on PATH, set in `.env`:
     ```
     TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
     ```

4. Run (CLI):
   ```bash
   python rag_pipeline.py
   ```
   Enter a file path when prompted (or press Enter to load Chroma), then ask questions.

   Or run the **Streamlit UI**:
   ```bash
   streamlit run app.py
   ```

### Inspect ingest output (PDF sample pages)

To see what the pipeline actually extracts for PDFs (1 text page, 3 table pages, 3 scanned/OCR pages):

```bash
python "tests/test ingestion/test_ingest_sample.py" "your.pdf" --full -o ingest_sample_report.txt
```

Requires `TESSERACT_CMD` in `.env` for `image_dominant` pages.

## Architecture Progress Tracker

| Date | Status | Details |
|---|---|---|
| 2026-05-28 | Pipeline v1 | PDF extraction (PyMuPDF + Tesseract OCR fallback); classification + boundary detection + logical document grouping; chunking (`SentenceSplitter`); hybrid retrieval (Vector + BM25); reranking (cross-encoder `ms-marco-MiniLM-L-6-v2`); generation (Gemini via LlamaIndex); optional query routing |
| 2026-05-28 | Ingest refactor | Two-pass ingest + heuristic segmentation + `page_manifest`; chunk dedup; query rewrite (typos); full-index hybrid retrieval + rerank + Gemini (`google.genai`) |
| 2026-05-29 | Phase 2 | Multi-format ingest (`load_file`); retrieval dedup (chunk, hybrid merge, citations); ingest sample tests under `tests/test ingestion/` |
| 2026-05-29 | Multi-format ingest | PDF two-pass pipeline unchanged; docx, xlsx, pptx, html, txt, images via `unstructured` |
| Planned | Step 4 | Persistent local index (Chroma); skip re-ingest on matching file hash; filename versioning when content changes |
| Planned | Step 5 | FastAPI backend + admin/chat web UI; API-key auth; Redis rate limits and query-answer cache |
| Planned | Step 6 | Background ingest worker; Postgres metadata optional; production object storage |
