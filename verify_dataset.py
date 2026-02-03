import json
import os
import glob

def verify_dataset_integrity():
    print("🔍 Verifying Evaluation Dataset Integrity...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'comparison/data/eval_questions_v2.json')
    doc_dir = os.path.join(base_dir, 'comparison/data/documents')
    
    # 1. Load Real Documents
    real_docs = set()
    print(f"\n[1] Scanning Document Directory: {doc_dir}")
    if os.path.exists(doc_dir):
        files = os.listdir(doc_dir)
        for f in files:
            if f.lower().endswith('.pdf'):
                real_docs.add(f)
                # Normalization check
                import unicodedata
                nfc_name = unicodedata.normalize('NFC', f)
                real_docs.add(nfc_name)
                
        print(f"   Found {len(files)} files ({len(real_docs)} valid PDF names tracking).")
    else:
        print("   ❌ Document directory not found!")
        return

    # 2. Load JSON Data
    print(f"\n[2] Loading Dataset: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   Loaded {len(data)} questions.")
    except Exception as e:
        print(f"   ❌ Failed to load JSON: {e}")
        return

    # 3. Check Consistency
    print("\n[3] Checking References...")
    missing_docs = set()
    valid_count = 0
    
    for item in data:
        source = item.get('source_doc')
        if not source:
            print(f"   ⚠️ Question ID {item.get('id')} has no source_doc")
            continue
            
        import unicodedata
        norm_source = unicodedata.normalize('NFC', source)
        
        if source in real_docs or norm_source in real_docs:
            valid_count += 1
        else:
            missing_docs.add(source)
            print(f"   ❌ Missing File: {source} (ID: {item.get('id')})")

    print("\n" + "="*50)
    if not missing_docs:
        print(f"✅ PASSED: All {valid_count} questions reference existing files.")
    else:
        print(f"❌ FAILED: Found {len(missing_docs)} missing document references.")
        print("Missing Files:")
        for m in missing_docs:
            print(f" - {m}")
    print("="*50)

if __name__ == "__main__":
    verify_dataset_integrity()
