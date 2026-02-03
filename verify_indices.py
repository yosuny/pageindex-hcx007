
import os
import sys
from qdrant_client import QdrantClient

# Setup
BASE_DIR = os.getcwd()
sys.path.insert(0, BASE_DIR)
from comparison.config import settings
from comparison.modules.pageindex_router import PageIndexRouter

def verify_indices():
    print("🔍 Verifying RAG Indices...\n")
    
    # 1. Check PageIndex Cache (Disk)
    print("[1] Checking PageIndex Cache Files:")
    cache_dir = "comparison/data/cache/pageindex_trees"
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        tree_files = [f for f in files if f.endswith('_tree.json')]
        print(f"   Found {len(tree_files)} tree cache files.")
        for f in tree_files:
            print(f"   - {f}")
    else:
        print("   ❌ Cache directory not found!")

    # 2. Check Vector DB (Qdrant)
    print("\n[2] Checking Vector DB (Qdrant):")
    try:
        client = QdrantClient(path=settings.QDRANT_PATH)
        collection_name = "vector_rag"
        
        # Get count
        count_result = client.count(collection_name=collection_name)
        print(f"   Total vectors: {count_result.count}")
        
        # Get unique sources
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True
        )
        
        sources = set()
        for p in points:
            if 'source' in p.payload:
                sources.add(p.payload['source'])
        
        print(f"   Found {len(sources)} unique documents in Vector DB:")
        print(f"   Debug - Actual Sources: {list(sources)}")
        expected_docs = [
            "00_AI_기본법.pdf",
            "01_AI_투명성_가이드라인.pdf",
            "02_AI_안전성_가이드라인.pdf",
            "03_고영향_AI_판단_가이드라인.pdf",
            "04_고영향_AI_책무_가이드라인.pdf",
            "05_AI_영향평가_가이드라인.pdf",
            "06_산업기술보호법.pdf",
            "07_국가핵심기술_클라우드_보안.pdf"
        ]
        
        missing = []
        for doc in expected_docs:
            if doc in sources:
                print(f"   ✅ Found: {doc}")
            else:
                print(f"   ❌ Missing: {doc}")
                missing.append(doc)
                
        if not missing:
            print("\n   ✅ Vector DB Verification Passed!")
        else:
            print(f"\n   ❌ Vector DB Verification Failed! Missing {len(missing)} documents.")
            
    except Exception as e:
        print(f"   ❌ Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    verify_indices()
