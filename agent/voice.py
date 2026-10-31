"""Voice mode: the coach speaks questions and explanations; you answer out loud.

Speech-to-text (your answer): Groq Whisper (free, uses GROQ_API_KEY) → Gemini audio understanding (GEMINI_API_KEY)
Text-to-speech (the coach):   Microsoft Edge neural voices via edge-tts (free, no key) → Google gTTS (free, no key)
Audio is cached in data/audio/ so replays are instant.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import re

import requests

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "audio")
VOICE = os.getenv("COACH_VOICE", "en-US-AriaNeural")       # e.g. en-US-GuyNeural, en-IN-NeerjaNeural, en-GB-SoniaNeural


# ------------------------------------------------------------------ text to speech
def clean_for_speech(text: str) -> str:
    t = re.sub(r"```.*?```", " (see the code on screen) ", text or "", flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"[*_#>|]", "", t)
    t = re.sub(r"\bO\(([^)]*)\)", r"O of \1", t)            # O(n log n) → "O of n log n"
    t = t.replace("≥", " at least ").replace("≤", " at most ").replace("→", " to ")
    return re.sub(r"\s+", " ", t).strip()


def _edge_tts(text: str, voice: str) -> bytes | None:
    try:
        import edge_tts

        async def run():
            buf = io.BytesIO()
            async for chunk in edge_tts.Communicate(text, voice, rate="+5%").stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()
        data = asyncio.run(run())
        return data or None
    except Exception:
        return None


def _gtts(text: str) -> bytes | None:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", tld="com").write_to_fp(buf)
        return buf.getvalue() or None
    except Exception:
        return None


def speak(text: str, voice: str | None = None) -> bytes | None:
    """MP3 bytes for the text (cached), or None if no TTS service is reachable."""
    text = clean_for_speech(text)
    if not text:
        return None
    voice = voice or VOICE
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:20]
    path = os.path.join(AUDIO_DIR, f"{key}.mp3")
    if os.path.exists(path):
        return open(path, "rb").read()
    data = _edge_tts(text, voice) or _gtts(text)
    if data:
        os.makedirs(AUDIO_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return data


def turn_script(last: dict | None, question: dict | None, label: str) -> str:
    """What the coach says this turn: feedback on your last answer (teaching the concept if you
    missed it), then the next question."""
    parts = []
    if last and last.get("eval"):
        ev, score = last["eval"], last.get("score")
        if score is not None and score < 7:
            parts.append(f"You scored {score:g} out of 10. {ev.get('feedback', '')}")
            teach = ev.get("explain_concept") or ev.get("model_answer")
            if teach:
                parts.append(f"Let me explain the concept. {teach}")
        elif score is not None:
            parts.append(f"Good answer, {score:g} out of 10. {ev.get('feedback', '')}")
    if question:
        parts.append(f"{label}. {question['question']}")
    return " ".join(parts)


# ------------------------------------------------------------------ speech to text
def _groq_whisper(audio: bytes, mime: str) -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                          headers={"Authorization": f"Bearer {key}"},
                          files={"file": ("answer.wav", audio, mime)},
                          data={"model": os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
                                "language": "en", "response_format": "json"}, timeout=90)
        r.raise_for_status()
        return (r.json().get("text") or "").strip() or None
    except Exception:
        return None


def _gemini_transcribe(audio: bytes, mime: str) -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    try:
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          params={"key": key}, timeout=120,
                          json={"contents": [{"parts": [
                              {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio).decode()}},
                              {"text": "Transcribe this spoken interview answer verbatim in English. "
                                       "Return only the transcript, with no commentary."}]}]})
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return " ".join(p.get("text", "") for p in parts).strip() or None
    except Exception:
        return None


def transcribe(audio: bytes, mime: str = "audio/wav") -> tuple[str | None, str]:
    """(transcript, engine)."""
    for name, fn in (("groq-whisper", _groq_whisper), ("gemini", _gemini_transcribe)):
        text = fn(audio, mime)
        if text:
            return text, name
    return None, "none"
