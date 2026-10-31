"""Coding Lab: run your solution against test cases in a subprocess, then get an interviewer-style
code review. The agent can also generate NEW problems and self-verifies them before showing them."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import yaml

from . import memory, skills
from .llm import LLM, LLMError

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "coding_problems.yaml")
GENERATED = os.path.join(os.path.dirname(__file__), "..", "data", "generated_problems.json")

HARNESS = r'''
import json, sys, copy, time, traceback
spec = json.load(open(sys.argv[1]))
ns = {}
try:
    exec(open(sys.argv[2]).read(), ns)
    fn = ns[spec["fn"]]
except Exception as e:
    print("@@RESULT@@" + json.dumps({"compile_error": f"{type(e).__name__}: {e}"})); sys.exit(0)
def norm(x, unordered):
    if unordered and isinstance(x, list):
        return sorted(json.dumps(i, sort_keys=True) for i in x)
    return json.loads(json.dumps(x))
out = []
for t in spec["tests"]:
    t0 = time.perf_counter()
    try:
        got = fn(*copy.deepcopy(t["args"]))
        ok = norm(got, spec["unordered"]) == norm(t["expected"], spec["unordered"])
        out.append({"ok": ok, "got": repr(got)[:200], "ms": round((time.perf_counter()-t0)*1000, 2)})
    except Exception as e:
        out.append({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
print("@@RESULT@@" + json.dumps({"tests": out}))
'''


def problems() -> list[dict]:
    with open(CONFIG) as f:
        base = yaml.safe_load(f)["problems"]
    gen = json.load(open(GENERATED)) if os.path.exists(GENERATED) else []
    return base + gen


def fn_name(signature: str) -> str:
    m = re.search(r"def\s+(\w+)\s*\(", signature)
    return m.group(1) if m else "solution"


def run_tests(problem: dict, code: str, timeout: int = 8) -> dict:
    with tempfile.TemporaryDirectory() as d:
        spec, src, harness = (os.path.join(d, n) for n in ("spec.json", "solution.py", "harness.py"))
        json.dump({"fn": fn_name(problem["signature"]), "tests": problem["tests"],
                   "unordered": problem.get("compare") == "unordered"}, open(spec, "w"))
        open(src, "w").write(code)
        open(harness, "w").write(HARNESS)
        try:
            p = subprocess.run([sys.executable, "-I", harness, spec, src], capture_output=True, text=True,
                               timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return {"error": f"Timed out after {timeout}s (infinite loop or too slow?)", "passed": 0, "total": len(problem["tests"])}
        line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
        if not line:
            return {"error": (p.stderr or p.stdout)[-800:], "passed": 0, "total": len(problem["tests"])}
        res = json.loads(line[len("@@RESULT@@"):])
        if "compile_error" in res:
            return {"error": res["compile_error"], "passed": 0, "total": len(problem["tests"])}
        res["passed"] = sum(t["ok"] for t in res["tests"])
        res["total"] = len(res["tests"])
        return res


def review(llm: LLM, problem: dict, code: str, result: dict, style: str) -> dict:
    prompt = (f"TASK: CODE_REVIEW\nYou are a {'FAANG bar-raiser' if style == 'faang_strict' else 'senior'} interviewer reviewing a coding-round solution.\n"
              f"PROBLEM: {problem['title']}: {problem['prompt']}\nSIGNATURE: {problem['signature']}\n"
              f"TEST RESULTS: passed {result.get('passed')}/{result.get('total')} {result.get('error', '')}\nCODE:\n```python\n{code}\n```\n"
              "Return JSON: {\"time_complexity\": str, \"space_complexity\": str, \"optimal_complexity\": str, "
              "\"is_optimal\": bool, \"code_quality\": 0-10, \"bugs\": [str], \"edge_cases_missed\": [str], "
              "\"feedback\": \"3-5 direct sentences\", \"interviewer_followups\": [2 follow-up questions an interviewer would ask]}")
    try:
        return llm.chat_json(prompt, temperature=0.1)
    except LLMError as e:
        return {"feedback": f"Review unavailable: {e}", "code_quality": None}


def grade_and_record(problem: dict, code: str, result: dict, rev: dict) -> float:
    rate = result.get("passed", 0) / max(1, result.get("total", 1))
    q = rev.get("code_quality")
    score = round(7 * rate + (0.3 * float(q) if isinstance(q, (int, float)) else 3 * rate), 1)
    if rev.get("is_optimal") is False:
        score = round(score * 0.85, 1)        # strict: brute force passes tests but loses points
    memory.execute("INSERT INTO coding_attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (memory.new_id(), memory.now(), problem["id"], problem["title"], problem["difficulty"],
                    result.get("passed", 0), result.get("total", 0), json.dumps(rev), score, code))
    skills.update_mastery(problem.get("skill", "dsa"), score, problem["difficulty"], "coding")
    return score


def generate_problem(llm: LLM, subtopic: str, difficulty: str, flavor: str) -> dict:
    """Generate a fresh problem + reference solution, then VERIFY the tests by running the reference."""
    prompt = (f"TASK: GEN_PROBLEM\nCreate an original {difficulty} Python coding-interview problem on '{subtopic}', "
              f"flavored for a {flavor} internship candidate (use a realistic domain scenario). "
              "Return JSON: {\"id\": snake_case, \"title\": str, \"prompt\": str, \"signature\": \"def name(...) -> type:\", "
              "\"reference_solution\": \"full python code defining the function\", "
              "\"tests\": [{\"args\": [positional args as JSON], \"expected\": JSON}] (6-8 tests incl. edge cases)}")
    for _ in range(2):
        p = llm.chat_json(prompt, temperature=0.8, max_tokens=2500)
        p.update({"difficulty": difficulty, "subtopic": subtopic, "skill": "dsa", "generated": True})
        check = run_tests(p, p.get("reference_solution", ""))
        if "tests" in check:
            p["tests"] = [t for t, r in zip(p["tests"], check["tests"]) if r["ok"]]  # drop tests the reference fails
        if len(p.get("tests", [])) >= 3:
            p.pop("reference_solution", None)
            existing = json.load(open(GENERATED)) if os.path.exists(GENERATED) else []
            os.makedirs(os.path.dirname(GENERATED), exist_ok=True)
            json.dump(existing + [p], open(GENERATED, "w"), indent=1)
            return p
    raise LLMError("Generated problem failed self-verification twice; try again.")
