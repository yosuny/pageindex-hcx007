import os
import sys
import json
import random
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comparison.modules.ncloud_llm import NCloudLLM
from comparison.modules.document_loader import DocumentLoader
from comparison.config import settings

def generate_questions():
    print("Initializing Question Generator...")
    llm = NCloudLLM(
        api_key=settings.NCLOUD_API_KEY,
        api_url=settings.NCLOUD_API_URL,
        thinking_effort="medium"
    )

    doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison", "data", "documents")
    
    # Target Documents for Question Generation
    targets = [
        # 1. Industrial Technology Protection Act (New)
        {
            "filename": "6._산업기술보호법.pdf",
            "category": "산업기술보호법",
            "count": 10,
            "prompt_type": "fact_and_reasoning"
        },
        # 2. AI Basic Act (New/Review)
        {
            "filename": "인공지능_발전과_신뢰_기반_조성_등에_관한_기본법(법률)(제21311호)(20260122).pdf",
            "category": "AI 기본법",
            "count": 10,
            "prompt_type": "fact_and_reasoning"
        },
        # 3. Comparative / Complex (Mixed)
        {
            "filename": "mixed", # conceptual
            "category": "종합/비교",
            "count": 10,
            "prompt_type": "comparison"
        }
    ]

    new_questions = []
    
    # Helper to load text
    def load_doc_text(filename):
        path = os.path.join(doc_dir, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return ""
        loader = DocumentLoader(path)
        docs = loader.load()
        # Concat first 20 pages or essential parts to fit context
        full_text = "\n".join([d["text"] for d in docs[:30]]) 
        return full_text

    current_id_start = 21 # Start after existing 20

    for target in targets:
        filename = target["filename"]
        count = target["count"]
        cat = target["category"]
        
        print(f"\nGenerating {count} questions for {cat}...")
        
        if filename == "mixed":
            # For comparison, we need to load excerpts from multiple files
            # Simplified: Just provide context about "General AI Guidelines vs Industrial Tech Protection"
            # Or ask LLM to rely on its training/context if we feed two docs.
            # To be safe, let's load logic from core docs.
            context = "Focus on comparing 'AI Basic Act' and 'Industrial Technology Protection Act'."
            # Load snippet from both
            c1 = load_doc_text("6._산업기술보호법.pdf")[:15000]
            c2 = load_doc_text("인공지능_발전과_신뢰_기반_조성_등에_관한_기본법(법률)(제21311호)(20260122).pdf")[:15000]
            context = f"Document A (Sanup-Tech): {c1}\n\nDocument B (AI-Basic): {c2}"
            
        else:
            context = load_doc_text(filename)
            if not context:
                continue

        # Dynamic Prompt
        system_prompt = """You are an expert examiner creating a high-difficulty exam for Legal AI.
Create questions that test the AI's ability to:
1. Find specific details (penalties, dates, numbers).
2. Understand definitions of terms.
3. Compare rules between different sections or documents.
4. Perform multi-step reasoning.

Output Format: JSON Array
[
  {
    "question": "...",
    "ground_truth": "...",
    "source_doc": "exact_filename_here"
  }
]"""
        
        user_prompt = f"""[Context]
{context[:50000]}... (truncated)

[Task]
Generate {count} pairs of high-quality Question and Ground Truth based on the text above.
Category: {cat}
Source Document: {filename if filename != 'mixed' else 'Multiple'}

Requirements:
- Questions must be in Korean.
- Ground Truth must be detailed and cite relevant articles if possible.
- Include some "What is the difference between..." questions if appropriate.
- Include at least one question where the answer depends on a condition (e.g., "If condition X, then Y").

Return valid JSON array of {count} items."""

        try:
            # Generate
            response = llm.generate(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                thinking_effort="medium"
            )
            
            # Parse
            # Simple cleanup
            clean_res = response.replace("```json", "").replace("```", "").strip()
            # Extract JSON list
            start = clean_res.find("[")
            end = clean_res.rfind("]")
            if start != -1 and end != -1:
                chunk_qs = json.loads(clean_res[start:end+1])
                
                # Post-process
                for q in chunk_qs:
                    q["id"] = current_id_start
                    q["category"] = cat
                    if filename != "mixed":
                        q["source_doc"] = filename
                    else:
                        q["source_doc"] = "Multiple (AI Basic Act, Industrial Tech Act)" # Simplified
                    
                    current_id_start += 1
                    new_questions.append(q)
            else:
                print(f"Failed to parse JSON for {cat}")
                print(clean_res[:200])
                
        except Exception as e:
            print(f"Error generating for {cat}: {e}")

    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison", "data", "eval_questions_expanded.json")
    
    # Load existing to merge? Or just save new ones? User wants expansion.
    # Let's create a NEW file first.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_questions, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Generated {len(new_questions)} new questions. Saved to {output_path}")

if __name__ == "__main__":
    generate_questions()
