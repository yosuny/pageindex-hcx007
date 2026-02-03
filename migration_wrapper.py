
import os
import json
import shutil
import subprocess
import sys

# Setup
base_dir = os.getcwd()
LOG_FILE = os.path.join(base_dir, "migration_result.log")
doc_dir = os.path.join(base_dir, 'comparison/data/documents')
json_path = os.path.join(base_dir, 'comparison/data/eval_questions_v2.json')
router_path = os.path.join(base_dir, 'comparison/modules/pageindex_router.py')

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    # Also print to stdout just in case
    print(msg)

# Clear log
with open(LOG_FILE, "w") as f:
    f.write(f"Migration started at {base_dir}\n")

# Exact mapping from list_dir output
mapping = {
    "인공지능_발전과_신뢰_기반_조성_등에_관한_기본법(법률)(제21311호)(20260122).pdf": "00_AI_기본법.pdf",
    "1._260126_인공지능_투명성_확보_가이드라인.pdf": "01_AI_투명성_가이드라인.pdf",
    "2._260122_인공지능_안전성_확보_가이드라인.pdf": "02_AI_안전성_가이드라인.pdf",
    "3._260122_고영향_인공지능_판단_가이드라인.pdf": "03_고영향_AI_판단_가이드라인.pdf",
    "4._260122_고영향_인공지능_사업자_책무_가이드라인-1.pdf": "04_고영향_AI_책무_가이드라인.pdf",
    "5._260122_인공지능_영향평가_가이드라인.pdf": "05_AI_영향평가_가이드라인.pdf",
    "6._산업기술보호법.pdf": "06_산업기술보호법.pdf",
    "국가핵심기술_클라우드_컴퓨팅_서비스_이용을_위한_보안관리_안내서_배포용.pdf": "07_국가핵심기술_클라우드_보안.pdf"
}

# 1. Rename Files
log("\n[STEP 1] Renaming Files")
try:
    if os.path.exists(doc_dir):
        files = os.listdir(doc_dir)
        log(f"Found {len(files)} files in documents dir")
        
        for old, new in mapping.items():
            old_p = os.path.join(doc_dir, old)
            new_p = os.path.join(doc_dir, new)
            
            if os.path.exists(old_p):
                os.rename(old_p, new_p)
                log(f"Renamed: {old} -> {new}")
            elif os.path.exists(new_p):
                log(f"Verified: {new} already exists")
            else:
                # Try normalization check
                import unicodedata
                found = False
                normalized_old = unicodedata.normalize('NFC', old)
                for f in files:
                    if unicodedata.normalize('NFC', f) == normalized_old:
                         real_old_p = os.path.join(doc_dir, f)
                         try:
                             os.rename(real_old_p, new_p)
                             log(f"Renamed (Normalized): {f} -> {new}")
                             found = True
                         except Exception as e:
                             log(f"Error renaming {f}: {e}")
                         break
                if not found:
                    log(f"Not Found: {old}")
    else:
        log(f"Error: {doc_dir} does not exist")
except Exception as e:
    log(f"Step 1 failed: {e}")

# 2. Update JSON
log("\n[STEP 2] Updating JSON")
try:
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        count = 0
        for item in data:
            src = item.get('source_doc', '')
            if src in mapping:
                item['source_doc'] = mapping[src]
                count += 1
                
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log(f"Updated {count} items in JSON")
    else:
        log(f"Error: {json_path} does not exist")
except Exception as e:
    log(f"Step 2 failed: {e}")

# 3. Patch Code
log("\n[STEP 3] Patching Code")
try:
    if os.path.exists(router_path):
        with open(router_path, 'r', encoding='utf-8') as f:
            code = f.read()
        patch_count = 0
        original_code = code
        for old, new in mapping.items():
            if old in code:
                code = code.replace(old, new)
                patch_count += 1
        
        if code != original_code:
            with open(router_path, 'w', encoding='utf-8') as f:
                f.write(code)
            log(f"Patched {patch_count} occurrences in router")
        else:
            log("No changes needed in router")
    else:
        log(f"Error: {router_path} does not exist")
except Exception as e:
    log(f"Step 3 failed: {e}")

# 4. Clean Data
log("\n[STEP 4] Cleaning Indices")
dirs_to_clean = [
    'comparison/qdrant_storage',
    'comparison/data/cache/pageindex_trees'
]
for d in dirs_to_clean:
    p = os.path.join(base_dir, d)
    if os.path.exists(p):
        try:
            shutil.rmtree(p)
            log(f"Deleted {p}")
        except Exception as e:
            log(f"Failed to delete {p}: {e}")
    else:
        log(f"Already clean: {p}")

# 5. Rebuild
log("\n[STEP 5] Rebuilding Indices")
try:
    # Use venv python if available
    venv_python = os.path.join(base_dir, "venv/bin/python3")
    if os.path.exists(venv_python):
        python_exe = venv_python
        log("Using venv python")
    else:
        python_exe = sys.executable
        log(f"Using system python: {python_exe}")
        
    script = os.path.join(base_dir, "preprocess_new_docs.py")
    
    # Run with subprocess
    log(f"Executing {script}...")
    result = subprocess.run(
        [python_exe, "-u", script],
        capture_output=True,
        text=True,
        cwd=base_dir
    )
    
    log(f"Process return code: {result.returncode}")
    log("--- STDOUT ---")
    log(result.stdout)
    log("--- STDERR ---")
    log(result.stderr)
    
except Exception as e:
    log(f"Step 5 failed: {e}")

log("\n[DONE] All steps completed.")
