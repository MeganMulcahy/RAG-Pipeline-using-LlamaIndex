import os
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import nest_asyncio
from google.colab import files
from llama_index.core import Document
from typing import List
from llama_index.llms.google_genai import GoogleGenAI # Corrected import
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from IPython.display import clear_output
from getpass import getpass

nest_asyncio.apply()

# Set up Google API key for Gemini and file upload
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = getpass("Enter your Google API key: ")
    print("Note: do not store API keys directly in the code.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Use an environment variable for the model if you want to change it without editing code.
MODEL_NAME = os.getenv("GOOGLE_MODEL")

# Initialize Gemini LLM with max_output_tokens to control response length and token usage
llm = GoogleGenAI(
    model=MODEL_NAME,
    max_tokens=500,
    system_prompt="""
    You are a strict extraction engine.

    Rules:
    - Output exactly ONE sentence.
    - Maximum 15 words.
    - No explanations.
    - No markdown.
    - No bullet points.
    - No introductions or conclusions.
    - No extra context.
    - Answer directly using retrieved information only.
    """
)
Settings.llm = llm

# Initialize embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
Settings.embed_model = embed_model

# File upload
def file_upload():
    print("Please select a PDF file to upload:")
    uploaded = files.upload()

    pdf_path = None # Initialize pdf_path to None

    for filename in uploaded.keys():
      if filename.endswith('.pdf'):
        pdf_path = filename

        with open(pdf_path, 'wb') as f:
          f.write(uploaded[filename])
        # Once a PDF is found and saved, we can break and return its path
        break # Exit the loop after processing the first PDF
      else:
        print(f"File {filename} is not a PDF. Please upload a PDF file.")

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

# Example usage:
pdf_path = file_upload()
index = process_and_index_pdf(pdf_path)

# The system prompt is now defined within build_rag_pipeline, no need to pass it here
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