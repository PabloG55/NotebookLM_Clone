"""
RAG-based chat: retrieves relevant chunks from vector store, then answers with Groq.
Maintains conversation history.
"""
from core.vector_store import VectorStore
from core.groq_client import groq_stream
from typing import List


SYSTEM_PROMPT = """You are ThinkBook AI, an intelligent assistant that answers questions 
based on the documents the user has uploaded. 

Rules:
- Always ground your answers in the provided context chunks.
- If the answer isn't in the context, say so honestly.
- Be conversational, clear, and helpful.
- You can reference previous conversation turns.
- Format responses with markdown when helpful.
- Be thorough but concise."""


def build_rag_messages(
    query: str,
    vector_store: VectorStore,
    history: List[dict],
    top_k: int = 6,
) -> List[dict]:
    # Retrieve relevant chunks with metadata
    results = vector_store.search(query, top_k=top_k, include_metadata=True)
    
    context_blocks = []
    if results:
        # results is a list of dicts: [{"text": chunk, "source": filename}, ...]
        for i, res in enumerate(results):
            text = res.get("text", "")
            source = res.get("source", "Unknown")
            context_blocks.append(f"[Source {i+1}: {source}]\n{text}")
            
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last 10 turns to stay within context)
    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add the current user message with retrieved context
    user_message = f"""Context from documents:
---
{context}
---

User question: {query}"""

    messages.append({"role": "user", "content": user_message})
    return messages


def stream_chat_response(query: str, vector_store: VectorStore, history: List[dict]):
    """
    Generator that yields response tokens one by one.
    """
    messages = build_rag_messages(query, vector_store, history)
    yield from groq_stream(messages, temperature=0.6, max_tokens=2048)
