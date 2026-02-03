from typing import List, Dict, Optional
import os
from .document_loader import DocumentLoader
from .chunker import Chunker
from .semantic_chunker import LocalSemanticChunker
from .vector_store import VectorStore
from .ncloud_llm import NCloudLLM
from ..config import settings

class VectorRAG:
    def __init__(self, collection_name: str = "vector_rag", chunking_strategy: str = "semantic"):
        self.vector_store = VectorStore(
            collection_name=collection_name, 
            use_ncloud=True
        )
        self.llm = NCloudLLM(
            api_key=settings.NCLOUD_API_KEY,
            api_url=settings.NCLOUD_API_URL
        )
        
        self.chunking_strategy = chunking_strategy
        if chunking_strategy == "semantic":
            self.chunker = LocalSemanticChunker(cache_dir=settings.CACHE_DIR)
        else:
            self.chunker = Chunker(
                chunk_size=settings.CHUNK_SIZE, 
                chunk_overlap=settings.CHUNK_OVERLAP
            )

    def ingest_document(self, file_path: str):
        """Load, chunk and index a PDF document"""
        print(f"Ingesting {file_path}...")
        
        # 1. Load
        loader = DocumentLoader(file_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages.")
        
        # 2. Chunk
        chunks = self.chunker.chunk_documents(docs)
        print(f"Created {len(chunks)} chunks.")
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # 3. Index
        self.vector_store.add_nodes(texts, metadatas)
        print("Indexing complete.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search relevant chunks"""
        return self.vector_store.search(query, top_k)

    def answer(self, query: str, top_k: int = 3, thinking_effort: str = "medium") -> str:
        """Generate answer with retrieved context"""
        # Retrieve
        docs = self.search(query, top_k)
        
        # Construct Context with proper document names
        context_parts = []
        for i, doc in enumerate(docs):
            page = doc['metadata'].get('page', 'Unknown')
            source_path = doc['metadata'].get('source', 'Unknown')
            # Extract filename from path
            import os
            doc_name = os.path.basename(source_path) if source_path != 'Unknown' else 'Unknown'
            # Use full filename for clarity
            context_parts.append(f"[[{doc_name}]] (p.{page})\n{doc['text']}")
            
        context = "\n\n".join(context_parts)
        
        # Construct Prompt - STRICT: only use retrieved context
        system_prompt = """당신은 법률 문서 분석 전문가입니다.

**중요 규칙:**
1. 반드시 아래 [검색된 문서]에 포함된 내용만 사용하여 답변하세요.
2. 검색된 문서에 없는 정보는 절대 사용하지 마세요.
3. 추측하거나 일반 지식을 사용하지 마세요.
4. 답변 시 근거가 되는 문서를 반드시 `[[파일명]] (p.페이지)` 형식으로 명시하세요. (예: [[AI기본법.pdf]] (p.12))
5. 여러 문서를 참고했다면 모두 표기하세요.
6. 검색된 문서에서 답을 찾을 수 없으면 "검색된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요."""

        user_prompt = f"""[검색된 문서]
{context}

---

[질문]
{query}

[답변 규칙]
- 위 [검색된 문서]의 내용만 사용하세요.
- 각 문장의 끝에 반드시 출처를 표기하세요.
- 출처 표기 형식: 문장 끝에 `[[실제파일명.pdf]] (p.페이지번호)` 를 추가.
- **주의**: '파일명'이라는 단어를 쓰지 말고, 위 [검색된 문서]에 적힌 실제 파일 이름(예: 6._산업기술보호법.pdf)을 그대로 사용하세요.

[답변]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate Answer
        response = self.llm.generate(messages, thinking_effort=thinking_effort)
        return response
