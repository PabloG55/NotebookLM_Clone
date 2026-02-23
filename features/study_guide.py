"""
Generates a structured study guide with key concepts, definitions, and flashcards.
"""
from core.groq_client import groq_chat


def generate_study_guide(text: str) -> str:
    words = text.split()
    if len(words) > 10000:
        text = " ".join(words[:10000])

    system_prompt = """You are an expert educator. Generate a comprehensive study guide 
from the provided document. Structure it as follows using markdown:

## 📚 Study Guide

### 🎯 Key Concepts
List the 5-8 most important concepts with a 2-3 sentence explanation each.

### 📖 Definitions & Terminology
List important terms with their definitions as a definition list.

### 💡 Key Facts & Data Points
Bullet points of crucial facts, statistics, or data from the document.

### 🔗 Relationships & Connections
Explain how the main concepts relate to each other.

### 🃏 Flashcards
Generate 8-10 flashcards in this exact format:

**Q:** Question here
**A:** Answer here

---

**Q:** Next question
**A:** Next answer

### ✅ Summary
A 3-4 sentence overall summary of the most important takeaways.

Use markdown formatting throughout. Be comprehensive and educational."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Document:\n\n{text}"},
    ]
    return groq_chat(messages, temperature=0.5, max_tokens=3000)
