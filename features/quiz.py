"""
Generates a multiple-choice quiz from document content.
Handles answer checking and scoring.
"""
import json
import re
from core.groq_client import groq_chat


def generate_quiz(text: str, num_questions: int = 5) -> list:
    """
    Returns a list of question dicts:
    {
      "question": str,
      "options": {"A": str, "B": str, "C": str, "D": str},
      "answer": "A"/"B"/"C"/"D",
      "explanation": str
    }
    """
    words = text.split()
    if len(words) > 10000:
        text = " ".join(words[:10000])

    system_prompt = f"""You are a quiz master. Based on the document content, generate exactly {num_questions} 
multiple-choice questions that test genuine comprehension.

Return ONLY a valid JSON array. No extra text, no markdown code blocks, no preamble. Just raw JSON.

Format:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    }},
    "answer": "A",
    "explanation": "Brief explanation of why A is correct."
  }}
]

Rules:
- Questions must be based strictly on the document content
- Make distractors plausible but clearly wrong
- Vary difficulty (easy, medium, hard)
- Keep questions clear and unambiguous
- Return ONLY the JSON array, nothing else"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Document:\n\n{text}"},
    ]

    raw = groq_chat(messages, temperature=0.4, max_tokens=3000)

    # Extract JSON from response
    try:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)
    except json.JSONDecodeError:
        return [
            {
                "question": "Could not parse quiz. Please try regenerating.",
                "options": {"A": "Regenerate", "B": "-", "C": "-", "D": "-"},
                "answer": "A",
                "explanation": "Quiz generation encountered a parsing error. Try again.",
            }
        ]


def check_answer(question_dict: dict, user_answer: str) -> tuple:
    """
    Returns (is_correct: bool, explanation: str)
    """
    correct = question_dict["answer"]
    is_correct = user_answer.upper() == correct.upper()
    return is_correct, question_dict.get("explanation", "")
