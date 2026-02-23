"""
Generates a 2-person podcast script from document content.
Alex (female voice) = curious host
Dr. Sam (male voice) = expert guest

Audio uses edge-tts for two distinct, natural Microsoft neural voices.
Each speaker line is rendered separately and stitched together with pydub.
"""
from core.groq_client import groq_chat
import io
import asyncio
import tempfile
import os


# Microsoft Neural Voice assignments
VOICE_ALEX = "en-US-JennyNeural"   # Warm, friendly female — host
VOICE_SAM  = "en-US-GuyNeural"     # Clear, authoritative male — expert


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
    """
    Parse script into list of (speaker, line) tuples.
    Returns: [("Alex", "dialogue..."), ("Dr. Sam", "dialogue..."), ...]
    """
    lines = []
    for line in script.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Dr. Sam:"):
            content = line[8:].strip()
            if content:
                lines.append(("Dr. Sam", content))
        elif line.startswith("Alex:"):
            content = line[5:].strip()
            if content:
                lines.append(("Alex", content))
    return lines


async def _synthesize_line(text: str, voice: str, output_path: str):
    """Async: synthesize one line of dialogue to an MP3 file using edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def _build_audio_async(script_lines: list) -> bytes:
    """
    Render each line with the correct speaker voice, stitch into one MP3.
    Speaker changes get a longer pause for natural conversation feel.
    """
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    pause_same   = AudioSegment.silent(duration=350)  # 0.35s same speaker
    pause_switch = AudioSegment.silent(duration=650)  # 0.65s speaker change

    prev_speaker = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (speaker, line) in enumerate(script_lines):
            voice = VOICE_ALEX if speaker == "Alex" else VOICE_SAM
            out_path = os.path.join(tmpdir, f"line_{i}.mp3")

            await _synthesize_line(line, voice, out_path)
            segment = AudioSegment.from_mp3(out_path)

            if prev_speaker is not None:
                combined += pause_switch if prev_speaker != speaker else pause_same

            combined += segment
            prev_speaker = speaker

    buf = io.BytesIO()
    combined.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)
    return buf.read()


def generate_podcast_audio(script_lines: list) -> bytes:
    """
    Public entry point: converts parsed script into a dual-voice stitched MP3.
    Alex = JennyNeural (female), Dr. Sam = GuyNeural (male).
    Returns MP3 bytes.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop.run_until_complete(_build_audio_async(script_lines))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_build_audio_async(script_lines))
        finally:
            loop.close()