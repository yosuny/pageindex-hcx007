import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.modules.vector_rag import VectorRAG

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_pipeline.py <pdf_path> <query>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    
    print(f"Testing Vector RAG Pipeline with Semantic Chunking...")
    rag = VectorRAG(chunking_strategy="semantic")
    
    # Ingest
    print(f"\n[1] Chunking & Indexing {pdf_path}...")
    rag.ingest_document(pdf_path)
    
    # Search
    print(f"\n[2] Searching for: {query}")
    results = rag.search(query, top_k=3)
    for i, res in enumerate(results):
        print(f"[{i+1}] Score: {res['score']:.4f}")
        print(f"Text: {res['text'][:100]}...")
        print(f"Metadata: {res['metadata']}")
        
    # Answer
    print(f"\n[3] Generating Answer...")
    answer = rag.answer(query)
    print("\n=== Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
