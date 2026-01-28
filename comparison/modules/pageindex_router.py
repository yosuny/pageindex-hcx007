"""
PageIndex Router Module

This module is responsible for selecting the most relevant documents
globally before performing detailed PageIndex search.
"""
import os
import sys
import json
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from comparison.modules.ncloud_llm import NCloudLLM
from comparison.config import settings

class PageIndexRouter:
    def __init__(self, thinking_effort: str = "medium"):
        self.llm = NCloudLLM(
            api_key=settings.NCLOUD_API_KEY,
            api_url=settings.NCLOUD_API_URL,
            thinking_effort=thinking_effort
        )

    def route(self, query: str, documents: List[str], top_k: int = 2) -> List[str]:
        """
        Select the most relevant documents for the query.
        
        Args:
            query: User's question
            documents: List of available filenames
            top_k: Number of documents to select
            
        Returns:
            List of selected filenames
        """
        # Formulate prompt
        doc_list_str = "\n".join([f"- {doc}" for doc in documents])
        
        system_prompt = f"""You are a Document Router. 
Identify the most relevant documents for the user's query from the list below.
Return ONLY valid JSON array of strings."""

        user_prompt = f"""[Available Documents]
{doc_list_str}

[User Query]
{query}

[Task]
Select {top_k} documents that are most likely to contain the answer.
If the query is general, select the most comprehensive ones.
If the query mentions a specific guideline (e.g., Transparency), select that file.

Return format: ["exact_filename_1.pdf", "exact_filename_2.pdf"]"""

        try:
            # Generate response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.generate(messages, thinking_effort="medium")
            
            # Robust JSON parsing
            clean_response = response.replace("```json", "").replace("```", "").strip()
            # If multiple lines, join them
            clean_response = "".join(clean_response.splitlines())
            
            # Find array brackets
            start = clean_response.find("[")
            end = clean_response.rfind("]")
            
            if start != -1 and end != -1:
                clean_response = clean_response[start:end+1]
                selected_docs = json.loads(clean_response)
            else:
                # Basic split if no JSON structure
                if "," in clean_response:
                    selected_docs = [d.strip().strip('"').strip("'") for d in clean_response.split(",")]
                else:
                    selected_docs = [clean_response.strip().strip('"').strip("'")]

            # Validate filenames
            valid_docs = []
            for doc in selected_docs:
                # Find best match in original document list
                # Simple exact match first
                if doc in documents:
                    valid_docs.append(doc)
                    continue
                
                # Check for substring match (if LLM returned partial name)
                for original_doc in documents:
                    if doc in original_doc or original_doc in doc:
                        valid_docs.append(original_doc)
                        break
            
            # Remove duplicates
            valid_docs = list(set(valid_docs))
            
            if not valid_docs:
                raise ValueError("No valid documents found after parsing")
                print("⚠️ Router returned invalid or empty list. Fallback to keyword match.")
                # Simple keyword match fallback
                keywords = query.split()
                scores = []
                for doc in documents:
                    score = sum(1 for k in keywords if k in doc)
                    scores.append((doc, score))
                # Sort by score desc
                scores.sort(key=lambda x: x[1], reverse=True)
                valid_docs = [x[0] for x in scores[:top_k]]
                
            return valid_docs[:top_k]
            
        except Exception as e:
            print(f"❌ Router Error: {e}")
            # Fallback on error: return top K documents simply
            return documents[:top_k]

# Test
if __name__ == "__main__":
    router = PageIndexRouter()
    docs = [
        "1._260126_인공지능_투명성_확보_가이드라인.pdf",
        "2._260122_인공지능_안전성_확보_가이드라인.pdf",
        "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법.pdf"
    ]
    q = "투명성 가이드라인의 주요 내용은?"
    print(f"Query: {q}")
    print(f"Docs: {docs}")
    print(f"Selected: {router.route(q, docs)}")
