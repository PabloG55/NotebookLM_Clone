"""Basic tests for ingestion module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ingestion import load_txt


def test_load_txt_utf8():
    sample = b"Hello, world! This is a test document."
    result = load_txt(sample)
    assert "Hello" in result
    assert len(result) > 0


def test_load_txt_empty():
    result = load_txt(b"   ")
    assert result == ""


if __name__ == "__main__":
    test_load_txt_utf8()
    test_load_txt_empty()
    print("All ingestion tests passed!")
