"""
Groq client with automatic dynamic model selection.
Always selects the best lightweight available model
directly from Groq's model list.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")
    return Groq(api_key=api_key)


def get_dynamic_model():
    """
    Fetch models from Groq API and select best lightweight one.
    Avoid large 70B+ models automatically.
    """
    client = get_groq_client()
    models = client.models.list().data

    model_ids = [m.id for m in models]

    # Sort models to prefer smaller ones automatically
    # Avoid 70B and very large models
    safe_models = [
        m for m in model_ids
        if "70b" not in m.lower()
        and "120b" not in m.lower()
        and "preview" not in m.lower()
    ]

    # Prefer instant / instruct / smaller context models
    preferred_keywords = ["8b", "instant", "instruct"]

    for keyword in preferred_keywords:
        for m in safe_models:
            if keyword in m.lower():
                return m

    # fallback to first safe model
    if safe_models:
        return safe_models[0]

    # last fallback
    return model_ids[0]


def groq_chat(messages, temperature=0.3, max_tokens=800):
    client = get_groq_client()
    model = get_dynamic_model()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠ Model error: {e}"


def groq_stream(messages, temperature=0.3, max_tokens=800):
    client = get_groq_client()
    model = get_dynamic_model()

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"⚠ Model error: {e}"