import os
import fitz  # PyMuPDF
import nest_asyncio
from llama_index.core import Document
from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from rag_models import initialize_models

# Module-level object placeholders. Actual initialization happens inside runtime setup.
llm = None
embed_model = None


def initialize_runtime():
    global llm, embed_model
    nest_asyncio.apply()
    llm, embed_model = initialize_models()


# File upload
def file_upload():
    # If running in notebook-like environments with files.upload available,
    # use it. Otherwise, ask for a local PDF path.
    if 'files' in globals() and hasattr(files, 'upload'):
        print("Please select a PDF file to upload:")
        uploaded = files.upload()

        for filename, filedata in uploaded.items():
            if filename.lower().endswith('.pdf'):
                with open(filename, 'wb') as f:
                    f.write(filedata)
                return filename
            else:
                print(f"File {filename} is not a PDF. Please upload a PDF file.")

        raise ValueError("No PDF file was uploaded.")

    while True:
        pdf_path = input("Enter the local path to a PDF file: ").strip()
        if not pdf_path:
            print("PDF path is required. Please try again.")
            continue
        if not os.path.isfile(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            continue
        if not pdf_path.lower().endswith('.pdf'):
            print("The specified file is not a PDF. Please enter a .pdf file.")
            continue
        return pdf_path


# Extract text
def extract_text_pdf(pdf_name):
    doc = fitz.open(pdf_name)

    # Extract text from all pages
    text = "\n".join([page.get_text() for page in doc])

    # Print some stats
    print(f"Number of pages: {len(doc)}")
    print(f"Extracted {len(text.split())} words from the PDF.")

    doc.close()
    print(text[:500])  # Print first 500
    return

# Convert pdf to LlammaIndex file
def load_pdf_with_pymupdf(pdf_path: str) -> List[Document]:
    doc = fitz.open(pdf_path)

    # Extract text from each page
    documents = []
    for i, page in enumerate(doc):
      text = page.get_text()

      # Skip empty pages
      if not text.strip():
        continue

      # Create Document object with metadata
      documents.append(
        Document(
          text=text,
          metadata={
          "file_name": os.path.basename(pdf_path),
          "page_number": i + 1,
          "total_pages": len(doc)
          }
        )
      )

    doc.close()

    # Print stats
    print(f"Processed {pdf_path}:")
    print(f"Extracted {len(documents)} pages with content")

    return documents


def process_and_index_pdf(pdf_path):
    #Process a PDF and create both vector and keyword indices.
    # Load documents
    documents = load_pdf_with_pymupdf(pdf_path)

    # Create vector index
    vector_index = VectorStoreIndex.from_documents(documents)

    print(f"Indexed {len(documents)} document chunks")

    return vector_index

# Function to create a query engine that uses query expansion
def create_query_expansion_engine(index):
    """Create a query engine that uses query expansion."""
    # First create multiple retrievers (base retriever)
    base_retriever = index.as_retriever(similarity_top_k=2)

    # Create a query fusion retriever
    fusion_retriever = QueryFusionRetriever(
        retrievers=[base_retriever],
        llm=llm,
        similarity_top_k=2,
        num_queries=3,  # Generate 3 queries per original query
        mode="reciprocal_rerank"  # Use reciprocal rank fusion
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact"
    )

    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors
    )

    return query_engine


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever, top_k=2):
        """Initialize with vector and keyword retrievers."""
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle, **kwargs) -> List[NodeWithScore]:
        """Retrieve from both retrievers and combine results."""
        # Get results from both retrievers
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        keyword_nodes = self.keyword_retriever.retrieve(query_bundle)

        # Combine all nodes
        all_nodes = list(vector_nodes) + list(keyword_nodes)

        # Remove duplicates (by node_id)
        unique_nodes = {}
        for node in all_nodes:
            if node.node_id not in unique_nodes:
                unique_nodes[node.node_id] = node

        # Sort by score (higher is better)
        sorted_nodes = sorted(
            unique_nodes.values(),
            key=lambda x: x.score if hasattr(x, 'score') else 0.0,
            reverse=True
        )
        return sorted_nodes[:self.top_k]  # Return top results
    
def build_rag_pipeline(index):
    """Build a simple but effective RAG pipeline with hybrid retrieval and reranking."""

    # Get all nodes from the index's docstore
    nodes = list(index.docstore.docs.values())

    # Determine safe top_k value (number of nodes to retrieve)
    # Must be at least 1 and no more than the number of available nodes
    num_nodes = len(nodes)
    safe_top_k = min(2, max(1, num_nodes))

    print(f"Index contains {num_nodes} nodes, using top_k={safe_top_k}")

    # Step 1: Create a hybrid retriever combining vector and keyword search
    # First, get the vector retriever (for semantic understanding)
    vector_retriever = index.as_retriever(
        similarity_top_k=safe_top_k  # Retrieve top 3 most similar chunks
    )

    # Next, create a BM25 retriever (for keyword matching)
    # Get all nodes from the index's docstore
    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=safe_top_k  # Retrieve top 3 most similar chunks
    )

    # Create our hybrid retriever instance
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=bm25_retriever,
        top_k=safe_top_k
    )

    # Step 2: Create a reranker to prioritize the most relevant chunks
    node_postprocessors = [] # Initialize as empty list
    if num_nodes > 1:
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=min(2, num_nodes)  # Keep only top results after reranking
        )
        node_postprocessors = [reranker]

    # Step 3: Build the query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        node_postprocessors=node_postprocessors
    )

    return query_engine


def main():
    initialize_runtime()
    pdf_path = file_upload()
    index = process_and_index_pdf(pdf_path)

    # The system prompt is now defined within initialize_models(), no need to pass it here
    rag_engine = build_rag_pipeline(index)

    print('\nChat with the RAG engine. Type exit to quit.\n')
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break

        response = rag_engine.query(user_query)
        print(response)
        print('\n---------------------- \n')


if __name__ == "__main__":
    main()