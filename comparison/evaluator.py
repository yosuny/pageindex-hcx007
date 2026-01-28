import os
import sys
import json
import time
import argparse
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.modules.vector_rag import VectorRAG
from comparison.modules.pageindex_rag import PageIndexRAG
from comparison.modules.pageindex_router import PageIndexRouter
from comparison.modules.ncloud_llm import NCloudLLM
from comparison.config import settings

class Evaluator:
    def __init__(self):
        print("Initializing Systems for Evaluation...")
        self.vector_rag = VectorRAG(chunking_strategy="semantic")
        self.pageindex_rag = PageIndexRAG(thinking_effort="medium")
        self.pageindex_router = PageIndexRouter(thinking_effort="medium")
        
        # Judge LLM (HCX-007)
        self.judge_llm = NCloudLLM(
            api_key=settings.NCLOUD_API_KEY,
            api_url=settings.NCLOUD_API_URL,
            thinking_effort="medium" # Use medium for judging logic
        )
        
        # Cache for PageIndex (for router filtering)
        # We need to know which files are "available" in the PageIndex cache to simulate the UI logic
        self.pageindex_cache_dir = self.pageindex_rag.cache_dir
        self.available_pageindex_docs = []
        if os.path.exists(self.pageindex_cache_dir):
            for f in os.listdir(self.pageindex_cache_dir):
                if f.endswith("_tree.json"):
                    # Extract original filename approximation or just use the cache naming convention
                    # Actually, the UI uses the document list to filter. 
                    # For evaluation, we assume all documents in the dataset are available.
                    pass

        # Load all PDF files names from the document directory to simulate available docs
        self.doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")
        self.all_docs = [f for f in os.listdir(self.doc_dir) if f.lower().endswith('.pdf')]
        print(f"Loaded {len(self.all_docs)} documents for evaluation context.")

    def run_judge(self, question: str, ground_truth: str, answer: str) -> Dict:
        from pageindex.utils import extract_json
        import re
        
        """Run LLM-as-a-Judge to score the answer."""
        if "검색된 문서에서 해당 정보를 찾을 수 없습니다" in answer or "관련 정보를 찾을 수 없습니다" in answer:
            return {"score": 1, "reason": "Model failed to find answer."}

        prompt = f"""You are an impartial judge evaluating the quality of an AI generated answer.
Compare the AI Answer with the Ground Truth.

[Question]
{question}

[Ground Truth]
{ground_truth}

[AI Answer]
{answer}

Evaluate the AI Answer on a scale of 1 to 5:
1: Completely incorrect or irrelevant.
2: Mostly incorrect, misses key points.
3: Partially correct, but misses some details or contains minor errors.
4: Mostly correct, captures key meaning.
5: Perfect match in meaning and details.

Return ONLY valid JSON:
{{ "score": 3, "reason": "Explanation..." }}"""

        try:
            response = self.judge_llm.generate(
                [{"role": "user", "content": prompt}], 
                thinking_effort="medium"
            )
            
            # 1. Try standard JSON extraction
            result = extract_json(response)
            if result and "score" in result:
                return result
                
            # 2. Fallback: Regex for score
            score_match = re.search(r'"score":\s*(\d)', response)
            if not score_match:
                score_match = re.search(r'score:\s*(\d)', response, re.IGNORECASE)
                
            reason_match = re.search(r'"reason":\s*"(.*?)"', response)
            
            if score_match:
                score = int(score_match.group(1))
                reason = reason_match.group(1) if reason_match else "Parsed via regex"
                return {"score": score, "reason": reason}

            return {"score": 0, "reason": f"Parsing failed. Response: {response[:50]}..."}
            
        except Exception as e:
            return {"score": 0, "reason": f"Judge Error: {str(e)}"}

    def evaluate(self, limit: int = None):
        # Load questions
        q_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eval_questions.json")
        with open(q_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        if limit:
            questions = questions[:limit]
            
        results = []
        
        print(f"Starting evaluation of {len(questions)} questions...")
        
        for idx, q_item in enumerate(tqdm(questions)):
            qid = q_item["id"]
            question = q_item["question"]
            gt = q_item["ground_truth"]
            target_source = q_item["source_doc"]
            
            # 1. Vector RAG
            start_v = time.time()
            try:
                # Direct answer generation (includes search)
                # Note: VectorRAG.answer() handles search internally.
                # However, to check retrieval accuracy, we ideally want the retrieved docs too.
                # But VectorRAG.answer() returns string. 
                # Let's inspect retrieval by calling search first separately (or just trust the answer/source string).
                # For robustness, let's call answer().
                
                # To measure retrieval accuracy properly, let's modify how we call it or just parse the answer sources?
                # Actually, VectorRAG logic puts sources in the prompt. 
                # Let's just run .search() to check retrieval metric, then .answer() for generation.
                
                # Retrieval Check
                v_docs = self.vector_rag.search(question, top_k=3)
                v_hit = any(target_source in d['metadata'].get('source', '') for d in v_docs)
                
                # Generation
                v_ans = self.vector_rag.answer(question, top_k=3, thinking_effort="medium")
                v_time = time.time() - start_v
                
            except Exception as e:
                v_ans = f"Error: {e}"
                v_hit = False
                v_time = 0

            # 2. PageIndex RAG (with Global Routing)
            start_p = time.time()
            try:
                # Global Routing
                selected_docs = self.pageindex_router.route(question, self.all_docs, top_k=2)
                
                # Retrieval Check (Did it select the right document?)
                # Note: pageindex_router returns list of filenames.
                # Check target_source match.
                # Sometimes filenames differ slightly due to normalization.
                # target_source is exact filename from JSON.
                p_router_hit = any(target_source in d for d in selected_docs)
                
                # Search & Generation
                # We need to simulate the UI loop
                all_p_results = []
                for doc in selected_docs:
                    pdf_path = os.path.join(self.doc_dir, doc)
                    # Assuming cached for speed, or it will build
                    self.pageindex_rag.build_tree(pdf_path) # Ensure tree exists
                    res = self.pageindex_rag.search(pdf_path, question, top_k=2)
                    for r in res:
                        r['source'] = doc
                    all_p_results.extend(res)
                
                # Retrieval Hit (Did the final search find relevant chunks?)
                p_search_hit = len(all_p_results) > 0 # Simple check if anything found
                
                if all_p_results:
                    # Construct Context
                    context_parts = []
                    for r in all_p_results:
                        context_parts.append(f"[[{r['source']}]] {r.get('title','')} (p.{r.get('page','?')})\n{r.get('text','')[:1000]}")
                    context = "\n\n".join(context_parts)
                    
                    # Generate
                    sys_prompt = "You are a legal expert. Answer based on context only. Cite source (doc, page)."
                    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
                    p_ans = self.pageindex_rag.llm.generate(
                        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], 
                        thinking_effort="medium"
                    )
                else:
                    p_ans = "검색된 문서에서 관련 정보를 찾을 수 없습니다."
                    
                p_time = time.time() - start_p
                
            except Exception as e:
                p_ans = f"Error: {e}"
                p_router_hit = False
                p_time = 0

            # 3. Judge
            v_eval = self.run_judge(question, gt, v_ans)
            p_eval = self.run_judge(question, gt, p_ans)
            
            # Store Result
            results.append({
                "id": qid,
                "question": question,
                "category": q_item["category"],
                # Vector Stats
                "v_time": v_time,
                "v_hit": v_hit,
                "v_score": v_eval.get("score", 0),
                "v_reason": v_eval.get("reason", ""),
                # PageIndex Stats
                "p_time": p_time,
                "p_router_hit": p_router_hit,
                "p_score": p_eval.get("score", 0),
                "p_reason": p_eval.get("reason", "")
            })
            
        # Save Report
        df = pd.DataFrame(results)
        os.makedirs("comparison/data/results", exist_ok=True)
        report_path = "comparison/data/results/evaluation_report.json"
        df.to_json(report_path, orient="records", force_ascii=False, indent=2)
        
        # Summary
        summary = f"""
### Evaluation Summary (N={len(df)})

| Metric | Vector RAG | PageIndex RAG |
| :--- | :--- | :--- |
| **Avg Score (1-5)** | {df['v_score'].mean():.2f} | {df['p_score'].mean():.2f} |
| **Avg Time (s)** | {df['v_time'].mean():.2f} | {df['p_time'].mean():.2f} |
| **Retrieval Hit Rate** | {df['v_hit'].mean()*100:.1f}% | {df['p_router_hit'].mean()*100:.1f}% (Router) |

*Retrieval Hit Rate for PageIndex measures if the Router selected the correct document.*
"""
        print(summary)
        with open("comparison/data/results/summary.md", "w") as f:
            f.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()
    
    evaluator = Evaluator()
    evaluator.evaluate(limit=args.limit)
