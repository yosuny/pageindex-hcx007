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

from comparison.modules.pageindex_rag import PageIndexRAG

class PageIndexRouter:
    def __init__(self, thinking_effort: str = "medium"):
        self.llm = NCloudLLM(
            api_key=settings.NCLOUD_API_KEY,
            api_url=settings.NCLOUD_API_URL,
            thinking_effort=thinking_effort
        )
        # Initialize helper to access cache paths
        self.helper_rag = PageIndexRAG(thinking_effort="none")

    def _get_document_summary(self, filename: str) -> str:
        """Read cached tree and extract a brief summary."""
        # Find the full path for the filename logic (simplified simulation)
        # In real app, we need absolute paths. For now, assume data/documents relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(base_dir, "data", "documents", filename)
        
        # Check cache
        cache_path = self.helper_rag._get_cache_path(pdf_path)
        
        summary_text = ""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    tree = json.load(f)
                
                # Extract top-level nodes titles and summaries
                items = []
                for node in tree[:5]: # Top 5 nodes
                    title = node.get("title", "")
                    summ = node.get("summary", "")[:100]
                    items.append(f"- {title}: {summ}")
                
                if items:
                    summary_text = "\n  ".join(items)
            except:
                pass
                
        return summary_text if summary_text else "(No summary available)"

    def route(self, query: str, documents: List[str], top_k: int = 2) -> List[str]:
        """
        Select relevant documents using summaries.
        """
        # Prepare document list with summaries
        doc_info_parts = []
        for doc in documents:
            summary = self._get_document_summary(doc)
            doc_info_parts.append(f"Document: {doc}\nSummary:\n  {summary}\n")
            
        doc_list_str = "\n".join(doc_info_parts)
        
        system_prompt = f"""You are a Document Router. 
Identify the most relevant documents for the user's query strategies.
Return ONLY valid JSON array of strings."""

        user_prompt = f"""[Available Documents]
{doc_list_str}

[User Query]
{query}

[Task]
Select {top_k} documents that are most likely to contain the answer.
Consider both filenames and provided summaries.

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
                doc = doc.strip()
                # Exact match
                if doc in documents:
                    valid_docs.append(doc)
                    continue
                
                # Partial match
                for original_doc in documents:
                    if doc in original_doc or original_doc in doc:
                        valid_docs.append(original_doc)
                        break
            
            # Remove duplicates
            valid_docs = list(set(valid_docs))
            
            if not valid_docs:
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
        "01_AI_투명성_가이드라인.pdf",
        "02_AI_안전성_가이드라인.pdf",
        "00_AI_기본법.pdf"
    ]
    q = "투명성 가이드라인의 주요 내용은?"
    print(f"Query: {q}")
    # Note: Summary retrieval will fail in this standalone test unless paths are correct relative to execution
    # But logic is sound.
    print(f"Selected: {router.route(q, docs)}")
