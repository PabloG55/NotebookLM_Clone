import os
import chromadb
import uuid
from typing import List
from core.embedder import embed_texts, embed_query

class VectorStore:
    def __init__(self, db_dir: str):
        """
        Initializes a persistent ChromaDB client for a specific notebook.
        db_dir: The path generated from core.storage_manager.get_chroma_db_dir()
        """
        self.db_dir = db_dir
        self.client = chromadb.PersistentClient(path=self.db_dir)
        # Get or create the collection for this specific notebook
        self.collection = self.client.get_or_create_collection(
            name="notebook_chunks",
            metadata={"hnsw:space": "l2"} # L2 distance to match the old FAISS behavior
        )

    def add_chunks(self, chunks: List[str], source_filename: str = "Unknown Source"):
        if not chunks:
            return

        # Generate embeddings using our existing sentence-transformer logic
        embeddings = embed_texts(chunks).tolist()
        
        # Generate unique IDs for chroma insertion
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Provide metadata tracking the original file name so we can cite its chunks
        metadatas = [{"source": source_filename} for _ in chunks]
        
        # Add to the chroma collection
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, top_k: int = 5, include_metadata: bool = False):
        """
        Returns a list of chunks if include_metadata is False
        If include_metadata is True, returns a list of dicts: [{"text": <chunk>, "metadata": <meta>}]
        """
        if self.collection.count() == 0:
            return []
            
        # Embed the query string
        q_emb = embed_query(query).tolist()
        
        # Query chroma
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas"] if include_metadata else ["documents"]
        )
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return []
            
        docs = results["documents"][0]
        
        if include_metadata and results.get("metadatas") and results["metadatas"][0]:
            metas = results["metadatas"][0]
            return [{"text": docs[i], "source": metas[i].get("source", "Unknown")} for i in range(len(docs))]
            
        return docs

    def is_ready(self) -> bool:
        return self.collection.count() > 0
