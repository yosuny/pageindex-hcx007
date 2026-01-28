import fitz  # PyMuPDF
from typing import List, Dict

class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, str]]:
        """
        Load PDF and return text by page.
        Returns:
            List[Dict]: [{"text": "...", "page": 1, "source": "filename"}, ...]
        """
        doc = fitz.open(self.file_path)
        documents = []
        
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():  # Skip empty pages
                documents.append({
                    "text": text,
                    "page": i + 1,
                    "source": self.file_path.split("/")[-1]
                })
                
        return documents
