from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Any
import uuid
import numpy as np
from .ncloud_embedding import NCloudEmbeddings

class VectorStore:
    def __init__(self, collection_name: str = "kg_nodes", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", use_ncloud: bool = False, client: QdrantClient = None):
        # Use disk storage for persistence
        if client:
            self.client = client
        else:
            from ..config.settings import QDRANT_PATH
            self.client = QdrantClient(path=QDRANT_PATH) 
        self.collection_name = collection_name
        self.use_ncloud = use_ncloud
        
        if self.use_ncloud:
            self.model = NCloudEmbeddings()
            self.vector_size = 1024
        else:
            self.model = SentenceTransformer(embedding_model)
            self.vector_size = 384
        
        # Initialize collection
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            
    def reset_collection(self):
        """Deletes and recreates the collection to ensure a clean state."""
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        print(f"Collection '{self.collection_name}' reset.")

    def count(self) -> int:
        """Returns the number of vectors in the collection."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except:
            return 0

    def _embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if self.use_ncloud:
            if isinstance(texts, str):
                return [self.model.embed_query(texts)]
            return self.model.embed_documents(texts)
        else:
            # SentenceTransformer returns ndarray
            embeddings = self.model.encode(texts)
            if isinstance(texts, str):
                return [embeddings.tolist()]
            return embeddings.tolist()

    def add_nodes(self, nodes: List[str], metadatas: List[Dict] = None):
        """
        Embeds and indexes graph nodes (or document chunks).
        """
        if not nodes:
            return
            
        embeddings = self._embed(nodes)
        
        if metadatas is None:
            metadatas = [{} for _ in nodes]
            
        points = []
        for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
            payload = {"text": node}
            payload.update(metadatas[i])
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Indexed {len(nodes)} nodes into VectorStore.")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Searches for relevant nodes using vector similarity.
        """
        query_vector = self._embed(query)[0]
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        return [{"text": res.payload.get("text", ""), "score": res.score, "metadata": res.payload} for res in results]
