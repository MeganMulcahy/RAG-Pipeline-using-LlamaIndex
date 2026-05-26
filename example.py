import os
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib

# LlamaIndex imports for enhanced document processing
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

# Configure Gemini (replace with your API key)
GEMINI_API_KEY = "AIzaSyA8IjR2fmLYMEOSj60nNxpTgL9o_Pqgmug"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-3.1-flash-lite")

# Initialize embedding models (both for compatibility)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llama_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

@dataclass
class PageInfo:
    """Stores information about a single page"""
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0

@dataclass
class LogicalDocument:
    """Represents a logical document within a PDF"""
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    chunks: List[Dict] = None

@dataclass
class ChunkMetadata:
    """Rich metadata for each chunk"""
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    embedding: Optional[np.ndarray] = None

def classify_document_type(text: str, max_length: int = 1500) -> str:
    """
    Classify the document type based on its content.
    Uses LLM to intelligently identify document category.
    """
    # Truncate text if too long to avoid token limits
    text_sample = text[:max_length] if len(text) > max_length else text

    prompt = f"""
    Analyze this document and classify it into ONE of these categories:
    - Resume: CV, professional profile, work history
    - Contract: Legal agreement, terms and conditions, service agreement
    - Mortgage Contract: Home loan agreement, mortgage terms, property financing
    - Invoice: Bill, payment request, financial statement
    - Pay Slip: Salary statement, wage slip, earnings statement
    - Lender Fee Sheet: Loan fees, lender charges, closing costs
    - Land Deed: Property deed, title document, ownership certificate
    - Bank Statement: Account statement, transaction history
    - Tax Document: W2, 1099, tax return, tax form
    - Insurance: Insurance policy, coverage document
    - Report: Analysis, research document, findings
    - Letter: Correspondence, memo, communication
    - Form: Application, questionnaire, data entry form
    - ID Document: Driver's license, passport, identification
    - Medical: Medical report, prescription, health record
    - Other: Doesn't fit other categories

    Document sample:
    {text_sample}

    Respond with ONLY the category name, nothing else.
    """

    try:
        response = gemini_model.generate_content(prompt)
        doc_type = response.text.strip()

        # Normalize the response
        valid_types = [
            'Resume', 'Contract', 'Mortgage Contract', 'Invoice', 'Pay Slip',
            'Lender Fee Sheet', 'Land Deed', 'Bank Statement', 'Tax Document',
            'Insurance', 'Report', 'Letter', 'Form', 'ID Document',
            'Medical', 'Other'
        ]

        # Find best match (case-insensitive)
        for valid_type in valid_types:
            if doc_type.lower() == valid_type.lower():
                return valid_type

        return 'Other'
    except Exception as e:
        print(f"Classification error: {e}")
        return 'Other'

def detect_document_boundary(prev_text: str, curr_text: str,
                            current_doc_type: str = None) -> bool:
    """
    Detect if two consecutive pages belong to the same document.
    Returns True if they're from the same document.
    """
    # Quick heuristic checks first
    if not prev_text or not curr_text:
        return False

    # Sample the texts for LLM analysis
    prev_sample = prev_text[-500:] if len(prev_text) > 500 else prev_text
    curr_sample = curr_text[:500] if len(curr_text) > 500 else curr_text

    prompt = f"""
    Determine if these two pages are from the SAME document.

    Current document type: {current_doc_type or 'Unknown'}

    End of Previous Page:
    ...{prev_sample}

    Start of Current Page:
    {curr_sample}...

    Consider:
    - Continuity of content
    - Formatting consistency
    - Topic coherence
    - Page numbers or headers

    Answer ONLY 'Yes' if same document or 'No' if different document.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip().lower().startswith('yes')
    except Exception as e:
        print(f"Boundary detection error: {e}")
        # Default to keeping pages together if uncertain
        return True
    
def extract_and_analyze_pdf(pdf_file) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    """
    Extract text from PDF and perform intelligent document analysis.
    Returns both page-level info and logical document groupings.
    Supports various file types including scanned PDFs with OCR.
    """
    print("📖 Starting PDF extraction and analysis...")

    # Extract text from each page
    if isinstance(pdf_file, dict) and "content" in pdf_file:
        doc = fitz.open(stream=pdf_file["content"], filetype="pdf")
    elif hasattr(pdf_file, "read"):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_file)

    pages_info = []
    for i, page in enumerate(doc):
        text = page.get_text()

        # If no text found, try OCR (for scanned documents)
        if not text.strip():
            print(f"  Page {i}: No text found, attempting OCR...")
            try:
                # Convert page to image and perform OCR
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

        pages_info.append(PageInfo(page_num=i, text=text))

    doc.close()

    if not pages_info:
        raise ValueError("No text could be extracted from PDF")

    print(f"✅ Extracted {len(pages_info)} pages")

    # Perform document classification and boundary detection
    print("🧠 Analyzing document structure...")
    logical_docs = []
    current_doc_type = None
    current_doc_pages = []
    doc_counter = 0

    for i, page_info in enumerate(pages_info):
        if i == 0:
            # First page - classify document type
            current_doc_type = classify_document_type(page_info.text)
            page_info.doc_type = current_doc_type
            page_info.page_in_doc = 0
            current_doc_pages = [page_info]
            print(f"  Page {i}: New document detected - {current_doc_type}")
        else:
            # Check if this page continues the previous document
            prev_text = pages_info[i-1].text
            is_same = detect_document_boundary(prev_text, page_info.text, current_doc_type)

            if is_same:
                # Continue current document
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = len(current_doc_pages)
                current_doc_pages.append(page_info)
            else:
                # New document detected - save previous and start new
                logical_doc = LogicalDocument(
                    doc_id=f"doc_{doc_counter}",
                    doc_type=current_doc_type,
                    page_start=current_doc_pages[0].page_num,
                    page_end=current_doc_pages[-1].page_num,
                    text="\n\n".join([p.text for p in current_doc_pages])
                )
                logical_docs.append(logical_doc)
                doc_counter += 1

                # Start new document
                current_doc_type = classify_document_type(page_info.text)
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = 0
                current_doc_pages = [page_info]
                print(f"  Page {i}: New document detected - {current_doc_type}")

    # Don't forget the last document
    if current_doc_pages:
        logical_doc = LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=current_doc_type,
            page_start=current_doc_pages[0].page_num,
            page_end=current_doc_pages[-1].page_num,
            text="\n\n".join([p.text for p in current_doc_pages])
        )
        logical_docs.append(logical_doc)

    print(f"✅ Identified {len(logical_docs)} logical documents")
    for ld in logical_docs:
        print(f"   - {ld.doc_type}: Pages {ld.page_start}-{ld.page_end}")

    return pages_info, logical_docs

def chunk_document_with_metadata(logical_doc: LogicalDocument,
                                chunk_size: int = 500,
                                overlap: int = 100) -> List[ChunkMetadata]:
    """
    Chunk a logical document while preserving rich metadata.
    Uses sliding window with overlap for better context.
    """
    chunks_metadata = []
    words = logical_doc.text.split()

    if len(words) <= chunk_size:
        # Document is small enough to be a single chunk
        chunk_meta = ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_0",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=0,
            page_start=logical_doc.page_start,
            page_end=logical_doc.page_end,
            text=logical_doc.text
        )
        chunks_metadata.append(chunk_meta)
    else:
        # Create overlapping chunks
        stride = chunk_size - overlap
        for i, start_idx in enumerate(range(0, len(words), stride)):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])

            # Calculate which pages this chunk spans
            # (simplified - in production, track more precisely)
            chunk_position = start_idx / len(words)
            page_range = logical_doc.page_end - logical_doc.page_start
            relative_page = int(chunk_position * page_range)
            chunk_page_start = logical_doc.page_start + relative_page
            chunk_page_end = min(chunk_page_start + 1, logical_doc.page_end)

            chunk_meta = ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=i,
                page_start=chunk_page_start,
                page_end=chunk_page_end,
                text=chunk_text
            )
            chunks_metadata.append(chunk_meta)

            if end_idx >= len(words):
                break

    return chunks_metadata

def chunk_with_llama_index(logical_doc: LogicalDocument,
                           chunk_size: int = 500,
                           chunk_overlap: int = 100) -> List[Document]:
    """
    Alternative: Use LlamaIndex's advanced chunking with metadata.
    """
    # Create LlamaIndex document with metadata
    doc = Document(
        text=logical_doc.text,
        metadata={
            "doc_id": logical_doc.doc_id,
            "doc_type": logical_doc.doc_type,
            "page_start": logical_doc.page_start,
            "page_end": logical_doc.page_end,
            "source": f"{logical_doc.doc_type}_document"
        }
    )

    # Use LlamaIndex's sentence splitter for better chunking
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        separator=" ",
    )

    # Create nodes (chunks) from document
    nodes = splitter.get_nodes_from_documents([doc])

    # Convert to our ChunkMetadata format for consistency
    chunks_metadata = []
    for i, node in enumerate(nodes):
        chunk_meta = ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=i,
            page_start=node.metadata.get("page_start", logical_doc.page_start),
            page_end=node.metadata.get("page_end", logical_doc.page_end),
            text=node.text
        )
        chunks_metadata.append(chunk_meta)

    return chunks_metadata

def process_all_documents(logical_docs: List[LogicalDocument],
                         use_llama_index: bool = False) -> List[ChunkMetadata]:
    """
    Process all logical documents into chunks with metadata.
    Can use either custom or LlamaIndex chunking.
    """
    all_chunks = []

    for logical_doc in logical_docs:
        if use_llama_index:
            chunks = chunk_with_llama_index(logical_doc)
        else:
            chunks = chunk_document_with_metadata(logical_doc)

        logical_doc.chunks = chunks  # Store reference
        all_chunks.extend(chunks)
        print(f"📄 {logical_doc.doc_type}: Created {len(chunks)} chunks")

    return all_chunks

def predict_query_document_type(query: str) -> Tuple[str, float]:
    """
    Predict which document type is most likely to contain the answer.
    Returns predicted type and confidence score.
    """
    prompt = f"""
    Analyze this query and predict which document type would most likely contain the answer.

    Query: "{query}"

    Choose the MOST LIKELY type from:
    - Resume: Career, experience, education, skills, employment history
    - Contract: Terms, agreements, obligations, parties, legal terms
    - Mortgage Contract: Home loan, property financing, mortgage terms, interest rates
    - Invoice: Payments, amounts due, billing, charges, invoiced items
    - Pay Slip: Salary, wages, deductions, earnings, pay period
    - Lender Fee Sheet: Loan fees, closing costs, origination fees, lender charges
    - Land Deed: Property ownership, deed information, property description, title
    - Bank Statement: Account balance, transactions, deposits, withdrawals
    - Tax Document: Tax information, W2, 1099, tax returns, tax amounts
    - Insurance: Coverage, policy details, premiums, claims
    - Report: Analysis, findings, conclusions, research data
    - Letter: Communications, requests, notifications, correspondence
    - Form: Applications, submitted data, form fields
    - ID Document: Personal identification, ID numbers, identity verification
    - Medical: Health information, medical conditions, prescriptions
    - Other: General or unclear

    Respond in JSON format:
    {{"type": "DocumentType", "confidence": 0.85}}

    Confidence should be between 0.0 and 1.0
    """

    try:
        response = gemini_model.generate_content(prompt)
        result = json.loads(response.text.strip())
        return result.get("type", "Other"), result.get("confidence", 0.5)
    except Exception as e:
        print(f"Query routing error: {e}")
        return "Other", 0.0

class IntelligentRetriever:
    """
    Advanced retrieval system with metadata filtering and query routing.
    """

    def __init__(self):
        self.index = None
        self.chunks_metadata = []
        self.doc_type_indices = {}  # Separate indices per doc type

    def build_indices(self, chunks_metadata: List[ChunkMetadata]):
        """
        Build FAISS indices with document type segregation.
        """
        print("🔨 Building vector indices...")
        self.chunks_metadata = chunks_metadata

        # Create embeddings for all chunks
        texts = [chunk.text for chunk in chunks_metadata]
        embeddings = embed_model.encode(texts, show_progress_bar=True)

        # Store embeddings in metadata
        for i, chunk in enumerate(chunks_metadata):
            chunk.embedding = embeddings[i]

        # Build main index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        # Build separate indices for each document type
        doc_types = set(chunk.doc_type for chunk in chunks_metadata)
        for doc_type in doc_types:
            type_indices = [i for i, chunk in enumerate(chunks_metadata)
                          if chunk.doc_type == doc_type]
            if type_indices:
                type_embeddings = embeddings[type_indices]
                type_index = faiss.IndexFlatL2(dim)
                type_index.add(type_embeddings)
                self.doc_type_indices[doc_type] = {
                    'index': type_index,
                    'mapping': type_indices  # Maps back to original chunks
                }

        print(f"✅ Indexed {len(chunks_metadata)} chunks across {len(doc_types)} document types")

    def retrieve(self, query: str, k: int = 4,
                filter_doc_type: Optional[str] = None,
                auto_route: bool = True) -> List[Tuple[ChunkMetadata, float]]:
        """
        Retrieve relevant chunks with optional filtering and routing.
        Returns chunks with relevance scores.
        """
        query_embedding = embed_model.encode([query])

        # Determine which index to search
        if filter_doc_type and filter_doc_type in self.doc_type_indices:
            # Use filtered index
            type_data = self.doc_type_indices[filter_doc_type]
            D, I = type_data['index'].search(query_embedding, k)
            # Map back to original chunks
            chunk_indices = [type_data['mapping'][i] for i in I[0]]
            distances = D[0]
        elif auto_route:
            # Predict best document type
            predicted_type, confidence = predict_query_document_type(query)
            print(f"🎯 Query routed to: {predicted_type} (confidence: {confidence:.2f})")

            if confidence > 0.7 and predicted_type in self.doc_type_indices:
                # High confidence - use specific index
                type_data = self.doc_type_indices[predicted_type]
                D, I = type_data['index'].search(query_embedding, k)
                chunk_indices = [type_data['mapping'][i] for i in I[0]]
                distances = D[0]
            else:
                # Low confidence - search all
                D, I = self.index.search(query_embedding, k)
                chunk_indices = I[0]
                distances = D[0]
        else:
            # Search all chunks
            D, I = self.index.search(query_embedding, k)
            chunk_indices = I[0]
            distances = D[0]

        # Convert distances to similarity scores (inverse)
        max_dist = max(distances) if len(distances) > 0 else 1.0
        scores = [(max_dist - d) / max_dist for d in distances]

        results = [(self.chunks_metadata[i], scores[idx])
                  for idx, i in enumerate(chunk_indices)]

        return results
    
def generate_answer_with_sources(query: str,
                                retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> Dict:
    """
    Generate answer with detailed source attribution.
    """
    if not retrieved_chunks:
        return {
            'answer': "I couldn't find relevant information to answer your question.",
            'sources': [],
            'confidence': 0.0
        }

    # Prepare context from retrieved chunks
    context_parts = []
    sources = []

    for chunk_meta, score in retrieved_chunks:
        context_parts.append(f"[From {chunk_meta.doc_type}, Pages {chunk_meta.page_start}-{chunk_meta.page_end}]")
        context_parts.append(chunk_meta.text)
        context_parts.append("")

        sources.append({
            'doc_type': chunk_meta.doc_type,
            'pages': f"{chunk_meta.page_start}-{chunk_meta.page_end}",
            'relevance': f"{score:.2%}",
            'preview': chunk_meta.text[:100] + "..."
        })

    context = "\n".join(context_parts)

    # Generate answer
    prompt = f"""
    You are a helpful AI assistant. Use the provided context to answer the question.
    Be specific and cite which document type and pages support your answer.

    Context:
    {context}

    Question: {query}

    Instructions:
    1. Answer based ONLY on the provided context
    2. Mention which document type(s) contain the information
    3. Be concise but complete
    4. If the context doesn't contain enough information, say so

    Answer:
    """

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()

        # Calculate overall confidence based on retrieval scores
        avg_score = sum(s for _, s in retrieved_chunks) / len(retrieved_chunks)

        return {
            'answer': answer,
            'sources': sources,
            'confidence': avg_score,
            'chunks_used': len(retrieved_chunks)
        }
    except Exception as e:
        print(f"Answer generation error: {e}")
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': sources,
            'confidence': 0.0
        }
    
class EnhancedDocumentStore:
    """
    Manages the complete document processing and retrieval pipeline.
    """

    def __init__(self):
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []
        self.retriever = IntelligentRetriever()
        self.is_ready = False
        self.processing_stats = {}
        self.filename = None

    def process_pdf(self, pdf_file, filename: str = "document.pdf"):
        """
        Complete PDF processing pipeline.
        """
        self.filename = filename
        self.is_ready = False
        start_time = datetime.now()

        try:
            # Extract and analyze PDF
            self.pages_info, self.logical_docs = extract_and_analyze_pdf(pdf_file)

            # Chunk documents with metadata
            self.chunks_metadata = process_all_documents(self.logical_docs)

            # Build retrieval indices
            self.retriever.build_indices(self.chunks_metadata)

            # Calculate processing statistics
            process_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats = {
                'filename': filename,
                'total_pages': len(self.pages_info),
                'documents_found': len(self.logical_docs),
                'total_chunks': len(self.chunks_metadata),
                'document_types': list(set(doc.doc_type for doc in self.logical_docs)),
                'processing_time': f"{process_time:.1f}s"
            }

            self.is_ready = True
            return True, self.processing_stats

        except Exception as e:
            return False, {'error': str(e)}

    def query(self, question: str, filter_type: Optional[str] = None,
             auto_route: bool = True, k: int = 4) -> Dict:
        """
        Query the document store.
        """
        if not self.is_ready:
            return {
                'answer': "Please upload and process a PDF first.",
                'sources': [],
                'confidence': 0.0
            }

        # Retrieve relevant chunks
        retrieved = self.retriever.retrieve(
            question, k=k,
            filter_doc_type=filter_type,
            auto_route=auto_route
        )

        # Generate answer with sources
        result = generate_answer_with_sources(question, retrieved)
        result['filter_used'] = filter_type or ('auto' if auto_route else 'none')

        return result

    def get_document_structure(self) -> List[Dict]:
        """
        Get the document structure for UI display.
        """
        if not self.logical_docs:
            return []

        structure = []
        for doc in self.logical_docs:
            structure.append({
                'id': doc.doc_id,
                'type': doc.doc_type,
                'pages': f"{doc.page_start + 1}-{doc.page_end + 1}",  # 1-indexed for UI
                'chunks': len(doc.chunks) if doc.chunks else 0,
                'preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            })

        return structure
    
def main():
    print("\n=== Enhanced Document Q&A ===\n")

    pdf_path = input("Enter path to PDF file: ").strip().strip('"')
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    doc_store = EnhancedDocumentStore()
    success, stats = doc_store.process_pdf(pdf_path, filename=os.path.basename(pdf_path))

    if not success:
        print(f"Error processing PDF: {stats.get('error', 'unknown error')}")
        return

    print(f"\nProcessed: {stats['filename']}")
    print(f"  Pages:     {stats['total_pages']}")
    print(f"  Documents: {stats['documents_found']}")
    print(f"  Chunks:    {stats['total_chunks']}")
    print(f"  Types:     {', '.join(stats['document_types'])}")
    print(f"  Time:      {stats['processing_time']}")

    print("\nDocument structure:")
    for doc in doc_store.get_document_structure():
        print(f"  [{doc['type']}]  pages {doc['pages']}  —  {doc['chunks']} chunks")

    print('\nReady. Type your question or "exit" to quit.\n')

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = doc_store.query(query, auto_route=True, k=4)

        print(f"\nAnswer:\n{result['answer']}\n")

        if result["sources"]:
            print("Sources:")
            for src in result["sources"]:
                print(f"  {src['doc_type']}  pages {src['pages']}  relevance {src['relevance']}")

        print(f"Confidence: {result['confidence']:.1%}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()