"""
PageIndex RAG Wrapper with NCloud HCX-007 Support

This module wraps the PageIndex vectorless RAG system to use NCloud's
HCX-007 model instead of OpenAI GPT-4o.
"""
import os
import sys
import json
import hashlib
import fitz  # PyMuPDF
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from comparison.modules.ncloud_llm import NCloudLLM
from comparison.config import settings

# PageIndex imports (only what we need)
from pageindex.utils import (
    get_page_tokens, 
    get_text_of_pdf_pages_with_labels,
    extract_json,
    JsonLogger
)


class PageIndexRAG:
    """
    PageIndex (Vectorless RAG) wrapper using NCloud HCX-007.
    
    Instead of embedding-based retrieval, PageIndex constructs a hierarchical
    tree structure of the document and uses LLM reasoning to navigate the tree.
    """
    
    def __init__(self, 
                 cache_dir: str = None,
                 thinking_effort: str = "medium"):
        """
        Initialize PageIndexRAG with NCloud LLM.
        
        Args:
            cache_dir: Directory to cache tree structures
            thinking_effort: HCX-007 thinking effort level (none, low, medium, high)
        """
        self.llm = NCloudLLM(
            api_key=settings.NCLOUD_API_KEY,
            api_url=settings.NCLOUD_API_URL,
            thinking_effort=thinking_effort
        )
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", "data", "cache", "pageindex_trees"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.trees = {}  # Store loaded trees by document path
        
    def _get_cache_path(self, pdf_path: str) -> str:
        """Get cache file path for a PDF's tree structure."""
        pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        pdf_name = Path(pdf_path).stem[:30]  # Truncate long names
        return os.path.join(self.cache_dir, f"{pdf_name}_{pdf_hash}_tree.json")
    
    def _add_node_ids(self, toc: List, prefix: str = "") -> None:
        """Add hierarchical node IDs to tree structure in-place."""
        if isinstance(toc, list):
            for i, node in enumerate(toc):
                node_id = f"{prefix}{i+1}" if prefix else str(i+1)
                if isinstance(node, dict):
                    node["node_id"] = node_id
                    if "children" in node and node["children"]:
                        self._add_node_ids(node["children"], f"{node_id}.")

    def _extract_pages_with_fitz(self, pdf_path: str) -> List[tuple]:
        """Extract text from PDF using PyMuPDF (fitz) to avoid encoding issues."""
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text()
            # Use character count as proxy for token count to maintain compatibility
            pages.append((text, len(text)))
        return pages

    def _get_text_from_pages(self, page_list: List[tuple], start_idx: int, end_idx: int) -> str:
        """Get text from a range of pages (0-indexed inclusive)."""
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(page_list) - 1, end_idx)
        
        texts = []
        for i in range(start_idx, end_idx + 1):
            text = page_list[i][0]
            if text.strip():
                texts.append(f"--- Page {i+1} ---\n{text}")
        return "\n\n".join(texts)
        
    def _llm_call_build(self, prompt: str, system_prompt: str = None) -> str:
        """LLM call for tree building - uses HIGH thinking for quality."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.generate(messages, thinking_effort="high")
    
    def _llm_call_search(self, prompt: str, system_prompt: str = None) -> str:
        """LLM call for search - uses MEDIUM thinking for balance."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.generate(messages, thinking_effort="high")
    
    def _llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """Default LLM call - delegates to build call for backward compatibility."""
        return self._llm_call_build(prompt, system_prompt)
    
    def build_tree(self, pdf_path: str, force_rebuild: bool = False) -> Dict:
        """
        Build PageIndex tree structure for a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            force_rebuild: If True, rebuild even if cached
            
        Returns:
            Tree structure dictionary
        """
        cache_path = self._get_cache_path(pdf_path)
        
        # Check cache
        if not force_rebuild and os.path.exists(cache_path):
            print(f"Loading cached tree from {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                tree = json.load(f)
            self.trees[pdf_path] = tree
            return tree
            
        print(f"Building tree for {pdf_path}...")
        
        # Step 1: Extract pages with tokens
        page_list = self._extract_pages_with_fitz(pdf_path)  # Returns [(text, char_count), ...]
        print(f"Extracted {len(page_list)} pages")
        
        # Step 2: Generate initial TOC structure
        # We use the first few pages to understand document structure
        sample_pages = min(30, len(page_list))
        sample_text = self._get_text_from_pages(page_list, 0, sample_pages - 1)
        
        toc_prompt = f"""Analyze the document structure from the text below and extract Key Chapters/Sections.
Look for a Table of Contents (TOC) pattern like "01장 ... 5p" or "1. Introduction ... 1".
**Also inspect Legal/Statute headers like "제1장", "제1절", "제1조"(Article) if no standard TOC exists.**
Ignore dotted lines (.......). Combine multi-line titles.

Document sample (first {sample_pages} pages):
{sample_text}

Return ONLY valid JSON like:
[{{ "title": "제1장 총칙", "page": 1, "children": [...] }}]

Important:
- Handle hierarchical structure if visible.
- For laws, extract Chapters(장) and key Articles(조).
- Ensure page numbers are integers.
- Do NOT wrap in markdown code blocks. Return raw JSON only."""

        system_prompt = """You are a document structure analyzer. 
Extract the hierarchical structure of the document.
Be precise with page numbers. Return valid JSON only."""

        response = self._llm_call_build(toc_prompt, system_prompt)
        
        # Parse response
        try:
            toc = extract_json(response)
            if not toc:
                toc = json.loads(response)
        except:
            # Fallback: create simple structure
            toc = [{"title": "Document", "page": 1}]
            
        # Step 3: Add node IDs
        self._add_node_ids(toc)
        
        # Step 4: Generate summaries for each section (skip for now to speed up)
        # toc = self._add_summaries(toc, page_list)
        
        # Cache the result
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(toc, f, indent=2, ensure_ascii=False)
        print(f"Tree cached to {cache_path}")
        
        self.trees[pdf_path] = toc
        return toc
    
    def _add_summaries(self, toc: List, page_list: List) -> List:
        """Add summaries to each node in the tree."""
        def process_node(node, depth=0):
            if isinstance(node, dict):
                start_page = node.get("page", 1) - 1  # 0-indexed
                
                # Determine end page (next sibling's page or document end)
                # For simplicity, use fixed window
                end_page = min(start_page + 3, len(page_list) - 1)
                
                if start_page < len(page_list):
                    # get_text_of_pdf_pages_with_labels uses 1-indexed start_page
                    page_text = self._get_text_from_pages(
                        page_list, start_page, end_page
                    )[:2000]  # Limit text length
                    
                    summary_prompt = f"""Summarize this section titled "{node.get('title', 'Untitled')}" in 1-2 sentences.

Content:
{page_text}

Summary:"""
                    
                    try:
                        summary = self._llm_call(summary_prompt)
                        node["summary"] = summary.strip()[:500]
                    except:
                        node["summary"] = f"Section about {node.get('title', 'this topic')}"
                        
                # Process children
                if "children" in node and node["children"]:
                    for child in node["children"]:
                        process_node(child, depth + 1)
                        
            elif isinstance(node, list):
                for item in node:
                    process_node(item, depth)
                    
        process_node(toc)
        return toc
    
    def search(self, pdf_path: str, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search the tree structure for relevant nodes.
        
        Uses LLM reasoning to navigate the tree and find relevant sections.
        
        Args:
            pdf_path: Path to PDF file
            query: User query
            top_k: Number of relevant sections to return
            
        Returns:
            List of relevant sections with their content
        """
        if pdf_path not in self.trees:
            self.build_tree(pdf_path)
            
        tree = self.trees[pdf_path]
        
        # Step 1: Ask LLM to identify relevant nodes
        tree_summary = self._tree_to_summary(tree)
        
        nav_prompt = f"""Given this document structure and a user question, identify the {top_k} most relevant sections.

Document Structure:
{tree_summary}

Question: {query}

Return the node IDs and titles of the most relevant sections as JSON:
[{{"node_id": "1.2", "title": "Relevant Section", "relevance": "why this is relevant"}}]"""

        system_prompt = """You are a document navigator. 
Analyze the structure and identify sections most likely to contain the answer.
Return valid JSON only."""

        response = self._llm_call_search(nav_prompt, system_prompt)
        
        try:
            relevant_nodes = extract_json(response)
            if not relevant_nodes:
                relevant_nodes = json.loads(response)
        except:
            relevant_nodes = [{"node_id": "1", "title": "Document"}]
            
        # Step 2: Extract content from identified nodes
        page_list = self._extract_pages_with_fitz(pdf_path)  # Returns [(text, char_count), ...]
        results = []
        
        for node_info in relevant_nodes[:top_k]:
            node = self._find_node(tree, node_info.get("node_id", "1"))
            if node:
                start_page = node.get("page", 1) - 1
                end_page = min(start_page + 2, len(page_list) - 1)
                
                # get_text_of_pdf_pages_with_labels uses 1-indexed pages
                text = self._get_text_from_pages(
                    page_list, start_page, end_page
                )
                
                results.append({
                    "node_id": node_info.get("node_id", ""),
                    "title": node.get("title", ""),
                    "summary": node.get("summary", ""),
                    "text": text[:2000],
                    "page": node.get("page", 1),
                    "relevance": node_info.get("relevance", ""),
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": node.get("page", 1)
                    }
                })
                
        return results
    
    def _tree_to_summary(self, tree: List, indent: int = 0) -> str:
        """Convert tree to readable summary."""
        lines = []
        
        def process(nodes, level=0):
            if isinstance(nodes, list):
                for node in nodes:
                    process(node, level)
            elif isinstance(nodes, dict):
                prefix = "  " * level
                node_id = nodes.get("node_id", "")
                title = nodes.get("title", "Untitled")
                page = nodes.get("page", "?")
                summary = nodes.get("summary", "")[:100]
                
                lines.append(f"{prefix}[{node_id}] {title} (p.{page})")
                if summary:
                    lines.append(f"{prefix}    → {summary}")
                    
                if "children" in nodes and nodes["children"]:
                    process(nodes["children"], level + 1)
                    
        process(tree)
        return "\n".join(lines)
    
    def _find_node(self, tree: List, node_id: str) -> Optional[Dict]:
        """Find a node by its ID."""
        def search(nodes):
            if isinstance(nodes, list):
                for node in nodes:
                    result = search(node)
                    if result:
                        return result
            elif isinstance(nodes, dict):
                if nodes.get("node_id") == node_id:
                    return nodes
                if "children" in nodes:
                    return search(nodes["children"])
            return None
            
        return search(tree)
    
    def answer(self, pdf_path: str, query: str, top_k: int = 3, 
               thinking_effort: str = "medium") -> str:
        """
        Generate an answer using PageIndex tree-based retrieval.
        
        Args:
            pdf_path: Path to PDF file
            query: User query
            top_k: Number of relevant sections to use
            thinking_effort: Thinking effort for answer generation
            
        Returns:
            Generated answer
        """
        # Retrieve relevant sections
        docs = self.search(pdf_path, query, top_k)
        
        # Construct context
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc['metadata'].get('source', 'Unknown')
            page = doc['metadata'].get('page', '?')
            title = doc.get('title', '')
            context_parts.append(
                f"[Section {i+1}] {title} (Source: {source}, Page: {page})\n{doc['text']}"
            )
            
        context = "\n\n".join(context_parts)
        
        # Generate answer - STRICT: only use retrieved context
        system_prompt = """당신은 법률 문서 분석 전문가입니다.

**중요 규칙:**
1. 반드시 아래 [검색된 섹션]에 포함된 내용만 사용하여 답변하세요.
2. 검색된 섹션에 없는 정보는 절대 사용하지 마세요.
3. 추측하거나 일반 지식을 사용하지 마세요.
4. 답변 시 반드시 출처(섹션명, 페이지)를 명시하세요.
5. 검색된 섹션에서 답을 찾을 수 없으면 "검색된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요."""

        user_prompt = f"""[검색된 섹션]
{context}

---

[질문]
{query}

[답변 규칙]
- 위 [검색된 섹션]의 내용만 사용하세요.
- 섹션에 없는 내용은 답변하지 마세요.
- 출처(Section 번호, 페이지)를 반드시 인용하세요.

[답변]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.llm.generate(messages, thinking_effort=thinking_effort)


# Simple test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pageindex_rag.py <pdf_path> <query>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    
    print("Initializing PageIndex RAG with NCloud HCX-007...")
    rag = PageIndexRAG()
    
    # Build tree
    print("\n[1] Building tree structure...")
    tree = rag.build_tree(pdf_path)
    print(f"Tree built with root nodes: {len(tree) if isinstance(tree, list) else 1}")
    
    # Search
    print(f"\n[2] Searching for: {query}")
    results = rag.search(pdf_path, query, top_k=3)
    for i, res in enumerate(results):
        print(f"[{i+1}] {res['title']} (Page {res['page']})")
        print(f"    Relevance: {res.get('relevance', 'N/A')}")
        
    # Answer
    print(f"\n[3] Generating Answer...")
    answer = rag.answer(pdf_path, query)
    print("\n=== Answer ===")
    print(answer)
