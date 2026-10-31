"""Check your API keys:  python check_llm.py
Lists the Gemini models your key can use and sends one test message to every configured provider."""
import os

import requests

from agent.llm import LLM, PROVIDERS, discover_gemini_model

key = os.getenv("GEMINI_API_KEY")
if key:
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key, "pageSize": 200}, timeout=30)
    if r.ok:
        flash = sorted(m["name"].split("/")[-1] for m in r.json().get("models", []) if "flash" in m["name"])
        print("Gemini Flash models available to your key:\n  " + "\n  ".join(flash))
        print("Auto-pick would be:", discover_gemini_model(key))
    else:
        print("Gemini key problem:", r.status_code, r.text[:300])

for name, p in PROVIDERS.items():
    if not p.configured():
        print(f"\n[{name}] skipped (no key)")
        continue
    try:
        reply = LLM(order=[name]).chat("Reply with exactly: OK", max_tokens=50)
        print(f"\n[{name}] model={p.model} -> WORKS: {reply.strip()[:60]}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[{name}] model={p.model} -> FAILED\n   {str(e)[:500]}")

try:
    print("\nJSON test:", LLM().chat_json('Return {"status": "ok", "n": 3}'))
except Exception as e:  # noqa: BLE001
    print("\nJSON test FAILED:", str(e)[:400])
