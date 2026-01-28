import os
import hashlib
import json
from typing import List, Dict
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class LocalSemanticChunker:
    """
    LangChain + HuggingFace (Local) based Semantic Chunker.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' (free, lightweight).
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = None):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile" # or "standard_deviation", "interquartile"
        )
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_path(self, text: str) -> str:
        """Generate cache filename based on text hash"""
        if not self.cache_dir:
            return None
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"chunk_cache_{text_hash}.json")

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk documents using semantic split.
        Implements State Persistence via caching.
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc["text"]
            
            # 1. Check Cache
            cache_path = self._get_cache_path(text)
            if cache_path and os.path.exists(cache_path):
                print(f"Loading chunks from cache: {cache_path}")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_chunks = json.load(f)
                    chunked_docs.extend(cached_chunks)
                continue

            # 2. Perform Semantic Chunking
            # Create LangChain Document
            lc_docs = self.text_splitter.create_documents([text])
            
            current_chunks = []
            for i, lc_doc in enumerate(lc_docs):
                chunk_data = {
                    "text": lc_doc.page_content,
                    "metadata": {
                        "page": doc["page"],
                        "source": doc["source"],
                        "chunk_index": i,
                        "chunk_method": "semantic_local"
                    }
                }
                chunked_docs.append(chunk_data)
                current_chunks.append(chunk_data)
            
            # 3. Save Cache
            if cache_path:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(current_chunks, f, ensure_ascii=False, indent=2)
                print(f"Saved chunks to cache: {cache_path}")

        return chunked_docs
