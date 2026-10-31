"""
Free-first, multi-provider LLM layer with automatic fallback.

Every provider below exposes an OpenAI-compatible /chat/completions endpoint,
so one small HTTP client covers all of them, with no vendor SDKs needed.

Order of attempts is set by LLM_PROVIDERS in .env (default: gemini,groq,cohere,openrouter,ollama).
Providers without a key are skipped. If a call fails (rate limit, outage, bad
JSON) the next provider is tried, so a demo never dies on a single 429.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass


@dataclass
class Provider:
    name: str
    base_url: str
    key_env: str | None
    default_model: str
    model_env: str
    extra_headers: dict = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.key_env) if self.key_env else "ollama"

    @property
    def model(self) -> str:
        return os.getenv(self.model_env, self.default_model)

    def configured(self) -> bool:
        if self.name == "ollama":
            return os.getenv("OLLAMA_ENABLED", "0") == "1"
        return bool(self.api_key)


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "gemini-flash-latest",
        "GEMINI_MODEL",
    ),
    "groq": Provider(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "openai/gpt-oss-120b",
        "GROQ_MODEL",
    ),
    "cohere": Provider(
        "cohere",
        "https://api.cohere.ai/compatibility/v1",
        "COHERE_API_KEY",
        "command-a-03-2025",
        "COHERE_MODEL",
    ),
    "openrouter": Provider(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free",
        "OPENROUTER_MODEL",
        {"X-Title": "SJSU Agentic Coworker"},
    ),
    "ollama": Provider(
        "ollama",
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        None,
        "llama3.1",
        "OLLAMA_MODEL",
    ),
}


class LLMError(RuntimeError):
    pass


class LLM:
    """Tiny client: .chat() for text, .chat_json() for structured output."""

    def __init__(self, order: list[str] | None = None, mock: Callable[[str, str], str] | None = None):
        raw = order or os.getenv("LLM_PROVIDERS", "gemini,groq,cohere,openrouter,ollama").split(",")
        self.order = [p.strip().lower() for p in raw if p.strip()]
        self.mock = mock
        self.last_provider: str | None = None
        self.calls: list[dict] = []  # lightweight telemetry for the evaluation tab

    # ------------------------------------------------------------------ status
    def available(self) -> list[str]:
        if self.mock:
            return ["mock"]
        return [p for p in self.order if p in PROVIDERS and PROVIDERS[p].configured()]

    def status(self) -> str:
        av = self.available()
        return "No LLM key configured. Add one to .env" if not av else " → ".join(av)

    # -------------------------------------------------------------------- core
    def chat(self, prompt: str, system: str = "You are a helpful expert assistant.",
             temperature: float = 0.4, max_tokens: int = 1800) -> str:
        if self.mock:
            self.last_provider = "mock"
            return self.mock(system, prompt)

        errors = []
        for name in self.available():
            p = PROVIDERS[name]
            use_effort = name in ("gemini", "groq")   # new Flash / gpt-oss models "think" before answering
            for attempt in range(3):  # retries: rate limit, retired model, unsupported option
                t0 = time.time()
                try:
                    body = {"model": p.model,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": prompt}],
                            "temperature": temperature,
                            # thinking tokens count toward the limit, so leave generous room
                            "max_tokens": max_tokens * 4 if name in ("gemini", "groq") else max_tokens}
                    if use_effort:
                        body["reasoning_effort"] = "low"
                    r = requests.post(f"{p.base_url}/chat/completions",
                                      headers={"Authorization": f"Bearer {p.api_key}",
                                               "Content-Type": "application/json", **p.extra_headers},
                                      json=body, timeout=120)
                    if r.status_code == 400 and use_effort and "reasoning" in r.text.lower():
                        use_effort = False          # provider/model doesn't accept the option
                        continue
                    if r.status_code == 429 and attempt < 2:
                        time.sleep(5)
                        continue
                    if r.status_code == 404 and name == "gemini" and attempt < 2:
                        newer = discover_gemini_model(p.api_key, exclude=p.model)
                        if newer:  # model retired/renamed: switch to the newest available Flash model
                            os.environ["GEMINI_MODEL"] = newer
                            continue
                    r.raise_for_status()
                    text = extract_text(r.json())
                    if not text.strip():
                        raise LLMError(f"empty reply (finish_reason={r.json()['choices'][0].get('finish_reason')})")
                    self.last_provider = f"{name}:{p.model}"
                    self.calls.append({"provider": name, "ok": True, "secs": round(time.time() - t0, 2)})
                    return text
                except Exception as e:  # noqa: BLE001
                    detail = getattr(getattr(e, "response", None), "text", "")[:200]
                    errors.append(f"{name}: {e} {detail}")
                    self.calls.append({"provider": name, "ok": False, "secs": round(time.time() - t0, 2)})
                    break
        if not self.available():
            raise LLMError("No LLM provider configured. Copy .env.example to .env and add a free GEMINI_API_KEY (or GROQ/COHERE/OPENROUTER).")
        raise LLMError("All providers failed:\n" + "\n".join(errors))

    def chat_json(self, prompt: str, system: str = "You return only valid JSON.",
                  temperature: float = 0.2, max_tokens: int = 2200, retries: int = 1) -> Any:
        """Ask for JSON and parse it robustly (strips ```json fences, trailing prose)."""
        sys = system + " Respond with ONLY valid JSON, with no markdown fences and no commentary."
        last = ""
        for _ in range(retries + 1):
            last = self.chat(prompt, sys, temperature, max_tokens)
            parsed = parse_json(last)
            if parsed is not None:
                return parsed
            prompt = prompt + "\n\nYour previous reply was not valid JSON. Return ONLY the JSON."
        raise LLMError(f"Could not parse JSON from model output: {last[:300]}")


def parse_json(text: str) -> Any | None:
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except Exception:
        pass
    # grab the first {...} or [...] block
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = t.find(open_c), t.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(t[s : e + 1])
            except Exception:
                continue
    return None


def discover_gemini_model(api_key: str, exclude: str = "") -> str | None:
    """List the models this key can use and pick the newest general-purpose Flash model."""
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"key": api_key, "pageSize": 200}, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/", 1)[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception:
        return None
    bad = ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "exp", "preview", "8b", "native")
    flash = [n for n in names if "flash" in n and not any(b in n for b in bad) and n != exclude]
    if not flash:
        flash = [n for n in names if "flash" in n and n != exclude]
    def version(n: str):
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (float(m.group(1)) if m else 0.0, "latest" in n, -len(n))
    return max(flash, key=version) if flash else None


def extract_text(data: dict) -> str:
    """Pull the answer text out of a chat-completions reply, whatever shape the provider uses."""
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):  # some providers return content parts
        c = "".join(part.get("text", "") for part in c if isinstance(part, dict))
    return c or ""
