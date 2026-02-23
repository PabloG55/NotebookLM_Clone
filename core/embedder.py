"""
Generates embeddings using sentence-transformers (runs locally, no API cost).
"""
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    return model.encode([query], convert_to_numpy=True)[0]
