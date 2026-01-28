from typing import List, Dict
import tiktoken

class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, model_name: str = "gpt-4o"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk = tokens[i : i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk))
            
        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk documents and preserve metadata.
        Args:
            documents: List[{"text": "...", "page": 1, ...}]
        Returns:
            List[{"text": "chunk...", "metadata": {...}}]
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc["text"]
            chunks = self.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "metadata": {
                        "page": doc["page"],
                        "source": doc["source"],
                        "chunk_index": i
                    }
                })
                
        return chunked_docs
