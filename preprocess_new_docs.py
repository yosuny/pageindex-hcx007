import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comparison.modules.vector_rag import VectorRAG
from comparison.modules.pageindex_rag import PageIndexRAG

def preprocess_new_documents():
    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc_dir = os.path.join(base_dir, "comparison", "data", "documents")
    
    # Initialize Systems
    print("Initializing RAG systems...")
    vector_rag = VectorRAG(chunking_strategy="semantic")
    pageindex_rag = PageIndexRAG(thinking_effort="low") # Use simple mode for build
    
    # List all PDFs
    documents = [f for f in os.listdir(doc_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(documents)} documents in directory.")
    
    new_docs = []
    
    # Check which ones are new (based on PageIndex cache existence)
    for doc in documents:
        pdf_path = os.path.join(doc_dir, doc)
        cache_path = pageindex_rag._get_cache_path(pdf_path)
        
        if not os.path.exists(cache_path):
            new_docs.append(doc)
            
    if not new_docs:
        print("✅ No new documents found. All documents are already indexed/cached.")
        return

    print(f"🆕 Found {len(new_docs)} new documents: {new_docs}")
    
    for doc in tqdm(new_docs, desc="Processing New Documents"):
        pdf_path = os.path.join(doc_dir, doc)
        
        try:
            # 1. Vector Ingestion
            print(f"\n[Vector RAG] Ingesting {doc}...")
            vector_rag.ingest_document(pdf_path)
            
            # 2. PageIndex Building
            print(f"[PageIndex RAG] Building Tree for {doc}...")
            # thinking_effort is managed inside the class instance or method default
            pageindex_rag.build_tree(pdf_path, force_rebuild=True) 
            
            print(f"✅ Successfully processed {doc}")
            
        except Exception as e:
            print(f"❌ Error processing {doc}: {e}")

if __name__ == "__main__":
    preprocess_new_documents()
