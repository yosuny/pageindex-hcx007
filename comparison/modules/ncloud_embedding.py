import os
import json
import requests
from typing import List
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# Load env variables
load_dotenv()

class NCloudEmbeddings(Embeddings):
    """
    NCloud CLOVA Studio Embedding v2 Wrapper.
    """
    def __init__(self):
        self.api_key = os.getenv("NCLOUD_API_KEY")
        # Embedding v2 Test App URL (Always uses v1 path or specific tool path)
        self.api_url = "https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/v2"
        self.request_id = os.getenv("NCLOUD_REQUEST_ID", "kg-rag-embedding")
        
        if not self.api_key:
            raise ValueError("NCLOUD_API_KEY not found in .env")

    def _embed(self, text: str) -> List[float]:
        # Ensure api_key starts with Bearer if not already
        auth_header = self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}"
        
        # Test App Authentication: Bearer Token
        headers = {
            "Authorization": auth_header,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id,
            "Content-Type": "application/json"
        }
        
        data = {"text": text}
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                
                if response.status_code == 429:
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                
                result = response.json()
                if "result" in result and "embedding" in result["result"]:
                    return result["result"]["embedding"]
                else:
                    return []
                    
            except Exception as e:
                print(f"Error embedding text: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                else:
                    return []
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        import time
        for text in texts:
            # Test App Limit: 60 QPM -> 1 request per second.
            # We add 0.8s sleep (trying to be faster but safe).
            time.sleep(0.8)
            
            emb = self._embed(text)
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append([0.0] * 1024)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        emb = self._embed(text)
        if not emb:
             return [0.0] * 1024
        return emb
