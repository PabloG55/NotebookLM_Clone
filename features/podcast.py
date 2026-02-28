"""
Generates a 2-person podcast script from document content.
Alex (female voice) = curious host
Dr. Sam (male voice) = expert guest

Audio uses Hugging Face TTS (facebook/mms-tts-eng).
Each speaker line is rendered separately and stitched together with pydub.
"""

import re
from core.groq_client import groq_chat
import io
import asyncio
import tempfile
import os
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer


# Load HF TTS model once (global)
TTS_MODEL_NAME = "facebook/mms-tts-eng"
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
tts_model = VitsModel.from_pretrained(TTS_MODEL_NAME)


def generate_podcast_script(text: str, num_exchanges: int = 12) -> str:
    words = text.split()
    if len(words) > 10000:
        text = " ".join(words[:10000])

    system_prompt = f"""You are a podcast scriptwriter. Write an engaging, natural podcast 
conversation between two hosts based on the document content below.

SPEAKERS:
- Alex: Enthusiastic, curious female host. She is the everyday person — asks smart questions, 
  reacts naturally (Oh wow, Wait seriously, That is fascinating), uses casual language, 
  and makes the topic accessible. She opens and closes the show.
- Dr. Sam: Warm but authoritative male expert. Gives clear insightful explanations with 
  real-world analogies. Occasionally says things like Great question or Exactly right.

CRITICAL FORMAT — output ONLY lines in this exact format, nothing else at all:
Alex: [her dialogue]
Dr. Sam: [his dialogue]
Alex: [her dialogue]

STRUCTURE TO FOLLOW:
1. Alex opens: introduces herself by name, the show called ThinkBook Podcast, and introduces Dr. Sam by name and his credentials
2. Dr. Sam greets the listeners warmly and says something personal about why this topic excites him
3. They dive into the content naturally — Alex asks, Dr. Sam explains, they riff off each other
4. Natural reactions throughout: No way, So what you are saying is, That reminds me of
5. Alex closes: summarizes 2 or 3 key takeaways and says goodbye to listeners

RULES:
- Generate at least {num_exchanges} back-and-forth exchanges after the intro
- NO stage directions, NO brackets, NO asterisks, NO music cues, NO episode numbers
- Every single line MUST start with exactly Alex: or Dr. Sam: and nothing else
- Sound like real humans talking, not a script being read
- Use contractions, filler words like right, exactly, yeah
- NEVER have a speaker say their own name in their own line"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Create the podcast episode based on this document:\n\n{text}",
        },
    ]
    return groq_chat(messages, temperature=0.88, max_tokens=4096)


def parse_podcast_script(script: str) -> list:
    lines = []
    for line in script.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r'^dr\.?\s*sam\s*:', line, re.IGNORECASE):
            content = re.sub(r'^dr\.?\s*sam\s*:\s*', '', line, flags=re.IGNORECASE).strip()
            if content:
                lines.append(("Dr. Sam", content))

        elif re.match(r'^alex\s*:', line, re.IGNORECASE):
            content = re.sub(r'^alex\s*:\s*', '', line, flags=re.IGNORECASE).strip()
            if content:
                lines.append(("Alex", content))

    return lines


def _synthesize_line_local(text: str, output_path: str):
    """Generate TTS audio locally using Hugging Face VITS model."""
    inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = tts_model(**inputs).waveform

    waveform = output.squeeze().cpu().numpy()

    # Save WAV temporarily
    sf.write(output_path, waveform, 16000)


async def _build_audio_async(script_lines: list) -> bytes:
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    pause_same   = AudioSegment.silent(duration=350)
    pause_switch = AudioSegment.silent(duration=650)

    prev_speaker = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (speaker, line) in enumerate(script_lines):
            out_path = os.path.join(tmpdir, f"line_{i}.wav")

            _synthesize_line_local(line, out_path)

            segment = AudioSegment.from_wav(out_path)

            if prev_speaker is not None:
                combined += pause_switch if prev_speaker != speaker else pause_same

            combined += segment
            prev_speaker = speaker

    buf = io.BytesIO()
    combined.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)
    return buf.read()


async def generate_podcast_audio(script_lines: list) -> bytes:
    """
    Public entry point: converts parsed script into a dual-voice stitched MP3.
    """
    return await _build_audio_async(script_lines)
