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
        
        # Load all PDF files from document directory
        self.doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")
        self.all_docs = [f for f in os.listdir(self.doc_dir) if f.lower().endswith('.pdf')]
        
        # Build pageindex_cached list (matching UI logic)
        from pathlib import Path
        self.pageindex_cache_dir = self.pageindex_rag.cache_dir
        self.pageindex_cached = set()
        if os.path.exists(self.pageindex_cache_dir):
            for cache_file in os.listdir(self.pageindex_cache_dir):
                if cache_file.endswith("_tree.json"):
                    # Match cache file to original PDF
                    for pdf in self.all_docs:
                        pdf_stem = Path(pdf).stem[:30]
                        if cache_file.startswith(pdf_stem):
                            self.pageindex_cached.add(pdf)
                            break
        
        print(f"Loaded {len(self.all_docs)} documents ({len(self.pageindex_cached)} PageIndex cached).")

    def run_judge(self, question: str, ground_truth: str, answer: str) -> Dict:
        """
        Run LLM-as-a-Judge to score the answer using a 2-stage process:
        1. Reasoning (Thinking Mode): Analyze the answer against ground truth.
        2. Formatting (Structured Output): Extract score and reason as JSON.
        """
        if "검색된 문서에서 해당 정보를 찾을 수 없습니다" in answer or "관련 정보를 찾을 수 없습니다" in answer:
            return {"score": 1, "reason": "Model failed to find answer."}

        # Stage 1: Reasoning (Thinking Mode)
        # We ask the model to think deeply and produce a free-form analysis.
        reasoning_prompt = f"""You are an impartial judge evaluating the quality of an AI generated answer.
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

Step-by-step Execution:
1. Analyze the core requirements of the Question and Ground Truth.
2. Check if the AI Answer covers all key facts.
3. Identify any hallucinations or wrong details.
4. Determine the final score (1-5).

Output your analysis in detail. Do NOT output JSON yet."""

        try:
            # Call LLM with Thinking Mode
            analysis_text = self.judge_llm.generate(
                [{"role": "user", "content": reasoning_prompt}], 
                thinking_effort="medium"
            )
            
            # Stage 2: Formatting (Prompt-based JSON extraction)
            # Since response_format may not be fully supported, we use explicit prompt
            format_prompt = f"""Based on the following analysis, extract the final score and reason.

[Analysis]
{analysis_text}

Output ONLY a valid JSON object with exactly this format (no other text):
{{"score": <integer 1-5>, "reason": "<brief explanation>"}}"""

            response_json_str = self.judge_llm.generate(
                [{"role": "user", "content": format_prompt}],
                thinking_effort="none"
            )
            
            # Parse the JSON output
            import re
            # Extract JSON object
            json_match = re.search(r'\{.*\}', response_json_str, re.DOTALL)
            if json_match:
                response_json_str = json_match.group(0)

            try:
                result = json.loads(response_json_str)
            except json.JSONDecodeError:
                # Fallback: try to extract score with regex
                score_match = re.search(r'"score"\s*:\s*(\d)', response_json_str)
                if score_match:
                    result = {"score": int(score_match.group(1)), "reason": "Parsed from partial response"}
                else:
                    result = {"score": 0, "reason": f"Parse error: {response_json_str[:100]}"}
                
            return result

        except Exception as e:
            print(f"Judge Error: {e}")
            return {"score": 0, "reason": f"Judge Error: {str(e)}"}

    def evaluate(self, limit: int = None):
        # Load questions
        q_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eval_questions_v2.json")
        with open(q_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        if limit:
            questions = questions[:limit]
            
        results = []
        
        # Load previous results if reuse requested
        prev_results_map = {}
        if hasattr(self, 'reuse_vector_path') and self.reuse_vector_path:
            print(f"Loading previous Vector RAG results from: {self.reuse_vector_path}")
            try:
                with open(self.reuse_vector_path, 'r', encoding='utf-8') as f:
                    prev_data = json.load(f)
                    for item in prev_data:
                        prev_results_map[item['id']] = item
            except Exception as e:
                print(f"Failed to load previous results: {e}")
        
        print(f"Starting evaluation of {len(questions)} questions...")
        
        for idx, q_item in enumerate(tqdm(questions)):
            qid = q_item["id"]
            question = q_item["question"]
            gt = q_item["ground_truth"]
            target_source = q_item["source_doc"]
            
            # 1. Vector RAG
            # If reusing, skip execution
            if prev_results_map and qid in prev_results_map:
                prev = prev_results_map[qid]
                v_time = prev.get('v_time', 0)
                v_hit = prev.get('v_hit', False)
                v_score = prev.get('v_score', 0)
                v_reason = prev.get('v_reason', "Reused from previous run")
                v_ans = "Reused from previous run" # We don't need actual answer for reporting mostly
                
                # If we need v_ans for something else, we might miss it if not saved. 
                # But for score/time report it's fine.
            else:
                start_v = time.time()
                try:
                    # Retrieval Check
                    v_docs = self.vector_rag.search(question, top_k=3)
                    v_hit = any(target_source in d['metadata'].get('source', '') for d in v_docs)
                    
                    # Generation
                    v_ans = self.vector_rag.answer(question, top_k=3, thinking_effort="medium")
                    v_time = time.time() - start_v
                    
                    # Judge for Vector
                    v_eval = self.run_judge(question, gt, v_ans)
                    v_score = v_eval.get("score", 0)
                    v_reason = v_eval.get("reason", "")
                    
                except Exception as e:
                    v_ans = f"Error: {e}"
                    v_hit = False
                    v_time = 0
                    v_score = 0
                    v_reason = f"Error: {e}"

            # 2. PageIndex RAG (with Global Routing) - Matching UI logic
            start_p = time.time()
            p_router_hit = False
            try:
                # Filter to only cached documents (matching UI)
                available_docs = [d for d in self.all_docs if d in self.pageindex_cached]
                
                if not available_docs:
                    p_ans = "PageIndex 캐시된 문서가 없습니다."
                    p_time = time.time() - start_p
                else:
                    # Global Routing on CACHED docs only
                    selected_docs = self.pageindex_router.route(question, available_docs, top_k=2)
                    
                    # Retrieval Check
                    p_router_hit = any(target_source in d for d in selected_docs)
                    
                    # Search on selected docs
                    all_p_results = []
                    for doc in selected_docs:
                        if doc in self.pageindex_cached:
                            pdf_path = os.path.join(self.doc_dir, doc)
                            self.pageindex_rag.build_tree(pdf_path)
                            res = self.pageindex_rag.search(pdf_path, question, top_k=2)
                            for r in res:
                                r['source'] = doc
                            all_p_results.extend(res)
                    
                    all_p_results = all_p_results[:5]  # Limit like UI
                    
                    if all_p_results:
                        # Construct Context (matching UI format)
                        context_parts = []
                        for r in all_p_results:
                            source = r.get('source', 'Unknown') if isinstance(r, dict) else 'Unknown'
                            title = r.get('title', '') if isinstance(r, dict) else ''
                            page = r.get('page', '?') if isinstance(r, dict) else '?'
                            raw_text = r.get('text', '') if isinstance(r, dict) else str(r)
                            text = str(raw_text)[:1500] if raw_text else ''
                            context_parts.append(f"[[{source}]] {title} (p.{page})\n{text}")
                        context = "\n\n".join(context_parts)
                        
                        # Generate with Korean prompts (matching UI)
                        sys_prompt = """당신은 법률 문서 분석 전문가입니다.
1. 반드시 아래 [검색된 섹션]의 내용만 사용하여 답변하세요.
2. 각 정보의 끝에 반드시 출처를 명시하세요. 형식: `(문서명, p.페이지번호)`
3. 문서명은 파일명 그대로(확장자 포함) 정확하게 기재하세요.
4. 검색된 섹션에서 답을 찾을 수 없으면 "검색된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요."""

                        user_prompt = f"""[검색된 섹션]
{context}

[질문]
{question}

[답변]"""
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
                p_time = time.time() - start_p  # Record actual time even on error

            # 3. Judge
            # v_eval is already computed or loaded in step 1
            
            # Evaluate PageIndex
            p_eval = self.run_judge(question, gt, p_ans)
            
            # Store Result
            results.append({
                "id": qid,
                "question": question,
                "category": q_item["category"],
                # Vector Stats
                "v_time": v_time,
                "v_hit": v_hit,
                "v_score": v_score,
                "v_reason": v_reason,
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
    parser.add_argument("--reuse-vector-from", type=str, default=None, help="Path to previous JSON result to reuse Vector RAG stats")
    args = parser.parse_args()
    
    evaluator = Evaluator()
    if args.reuse_vector_from:
        evaluator.reuse_vector_path = args.reuse_vector_from
        
    evaluator.evaluate(limit=args.limit)
    evaluator.evaluate(limit=args.limit)
