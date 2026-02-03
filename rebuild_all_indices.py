import os
import sys
import glob
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comparison.modules.vector_rag import VectorRAG
from comparison.modules.pageindex_rag import PageIndexRAG

def rebuild_all():
    print("="*60)
    print("🚀 Rebuilding All Indices (Vector & PageIndex)")
    print("="*60)
    
    # 1. Initialize
    print("\n[1] Initializing Systems...")
    vector_rag = VectorRAG(chunking_strategy="semantic")
    pageindex_rag = PageIndexRAG(thinking_effort="low")  # Use low effort for faster building
    
    # Documents Directory
    doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison", "data", "documents")
    pdf_files = sorted(glob.glob(os.path.join(doc_dir, "*.pdf")))
    
    if not pdf_files:
        print("❌ No PDF files found in comparison/data/documents!")
        return

    print(f"Found {len(pdf_files)} documents.")
    for pdf in pdf_files:
        print(f" - {os.path.basename(pdf)}")

    # 2. Rebuild Vector Store
    print("\n" + "-"*60)
    print("📦 [Vector Store] Resetting and Re-indexing...")
    print("-"*60)
    
    try:
        vector_rag.vector_store.reset_collection()
        print("✅ Collection reset.")
        
        for i, pdf_path in enumerate(pdf_files):
            print(f"[{i+1}/{len(pdf_files)}] Ingesting {os.path.basename(pdf_path)}...")
            vector_rag.ingest_document(pdf_path)
            
        print("✅ Vector Store Rebuild Complete.")
        
    except Exception as e:
        print(f"❌ Vector Store Error: {e}")
        import traceback
        traceback.print_exc()

    # 3. Rebuild PageIndex
    print("\n" + "-"*60)
    print("🌲 [PageIndex] Rebuilding Tree Cache...")
    print("-"*60)
    
    for i, pdf_path in enumerate(pdf_files):
        print(f"[{i+1}/{len(pdf_files)}] Building Tree for {os.path.basename(pdf_path)}...")
        try:
            pageindex_rag.build_tree(pdf_path, force_rebuild=True)
        except Exception as e:
            print(f"❌ PageIndex Error for {os.path.basename(pdf_path)}: {e}")

    print("\n" + "="*60)
    print("🎉 All Indices Rebuilt Successfully!")
    print("="*60)

if __name__ == "__main__":
    rebuild_all()
