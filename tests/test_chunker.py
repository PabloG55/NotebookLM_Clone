"""Basic tests for chunker module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.chunker import chunk_text


def test_chunk_basic():
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_overlap():
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # With overlap, words from end of chunk 1 should appear in start of chunk 2
    assert len(chunks) >= 2


def test_chunk_short_text():
    text = "This is a very short document."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


if __name__ == "__main__":
    test_chunk_basic()
    test_chunk_overlap()
    test_chunk_short_text()
    print("All chunker tests passed!")
