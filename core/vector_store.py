"""
In-memory FAISS vector store. Stores chunks and allows similarity search.
"""
import faiss
import numpy as np
from typing import List
from core.embedder import embed_texts, embed_query


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: List[str] = []
        self.dim = 384  # all-MiniLM-L6-v2 output dim

    def add_chunks(self, chunks: List[str]):
        embeddings = embed_texts(chunks)
        embeddings = np.array(embeddings).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        if self.index is None or len(self.chunks) == 0:
            return []
        q_emb = embed_query(query).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results

    def reset(self):
        self.index = None
        self.chunks = []

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0
