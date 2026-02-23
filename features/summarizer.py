"""
Generates brief or descriptive summaries of the full document text.
"""
from core.groq_client import groq_chat


def summarize(text: str, mode: str = "brief") -> str:
    """
    mode: 'brief' (3-5 sentences) or 'descriptive' (detailed, structured)
    """
    # Truncate text to ~12000 words to fit context window
    words = text.split()
    if len(words) > 12000:
        text = " ".join(words[:12000]) + "\n\n[... document truncated for summarization ...]"

    if mode == "brief":
        instruction = (
            "You are an expert summarizer. Read the following document and provide a "
            "concise summary in 4-6 sentences. Capture the main topic, key points, and "
            "conclusion. Be direct and clear."
        )
    else:
        instruction = (
            "You are an expert analyst. Read the following document and provide a detailed, "
            "structured summary. Include:\n"
            "- **Overview**: What the document is about\n"
            "- **Key Themes**: Main topics covered\n"
            "- **Important Details**: Critical facts, data, arguments\n"
            "- **Conclusions**: Key takeaways\n"
            "Use markdown formatting. Be thorough and comprehensive."
        )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Document:\n\n{text}"},
    ]
    return groq_chat(messages, temperature=0.4, max_tokens=2048)
