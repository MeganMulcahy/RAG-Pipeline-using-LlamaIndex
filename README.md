# AI Mortgage Document Intelligence Automation System

A RAG (Retrieval-Augmented Generation) pipeline for querying mortgage PDF documents using natural language.

## What It Does

Upload a mortgage PDF and ask plain-English questions. The system extracts text, indexes it, and returns concise, grounded answers.

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
   GOOGLE_API_KEY=your_key_here
   GOOGLE_MODEL=your_model_here
   ```

3. Run:
   ```bash
   python rag_pipeline.py
   ```
   Enter a PDF path when prompted, then ask questions. Type `exit` to quit.
