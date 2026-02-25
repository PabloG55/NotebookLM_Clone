"""
Groq client with automatic model fallback.
If the primary model hits a rate limit, it automatically tries the next one.
Shows friendly retry messages instead of crashing.
"""
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODELS = [
    "llama-3.3-70b-versatile",
]

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Please add it to your .env file or HuggingFace Space Secrets."
        )
    return Groq(api_key=api_key)


def _is_rate_limit_error(e) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit_exceeded" in msg


def _extract_retry_time(e) -> str:
    """
    Pull the human-readable retry time from the Groq error message.
    e.g. 'Please try again in 1h56m35.808s' → '1h 56m 35s'
    """
    raw = str(e)
    # Match patterns like 1h56m35.808s / 45m12s / 30s
    match = re.search(
        r'try again in\s+((?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?)',
        raw, re.IGNORECASE
    )
    if match:
        hours   = match.group(2)
        minutes = match.group(3)
        seconds = match.group(4)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds:
            parts.append(f"{int(float(seconds))}s")
        return " ".join(parts) if parts else "a few minutes"
    return "a few minutes"


def _friendly_rate_limit_message(errors: list) -> str:
    """Build a friendly message from all the rate limit errors collected."""
    # Try to find any retry time hint from the errors
    retry_time = "a few minutes"
    for e in errors:
        t = _extract_retry_time(e)
        if t != "a few minutes":
            retry_time = t
            break

    return (
        f"⏳ **All AI models are currently at their rate limit.**\n\n"
        f"Please try again in **{retry_time}**.\n\n"
        f"> 💡 Tip: Groq's free tier resets daily. "
        f"You can also upgrade at [console.groq.com/settings/billing](https://console.groq.com/settings/billing) for more tokens."
    )


def groq_chat(
    messages: list,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    Send messages to Groq. Auto-falls back through MODELS list if rate limited.
    Returns a friendly message string instead of crashing when all models exhausted.
    """
    client = get_groq_client()
    models_to_try = MODELS if model is None else ([model] + [m for m in MODELS if m != model])

    errors = []
    for attempt_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if _is_rate_limit_error(e):
                print(f"[ThinkBook] Rate limit on {attempt_model}, trying next model...")
                errors.append(e)
                continue
            else:
                raise e

    # All models exhausted — return friendly message instead of crashing
    return _friendly_rate_limit_message(errors)


def groq_stream(
    messages: list,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """
    Stream tokens from Groq. Auto-falls back through MODELS if rate limited.
    If all models exhausted, yields a single friendly message string.
    """
    client = get_groq_client()
    models_to_try = MODELS if model is None else ([model] + [m for m in MODELS if m != model])

    errors = []
    for attempt_model in models_to_try:
        try:
            stream = client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return  # success

        except Exception as e:
            if _is_rate_limit_error(e):
                print(f"[ThinkBook] Rate limit on {attempt_model}, trying next model...")
                errors.append(e)
                continue
            else:
                raise e

    # All models exhausted — yield friendly message
    yield _friendly_rate_limit_message(errors)