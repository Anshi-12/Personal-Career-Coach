"""Grader calibration:  python -m eval.run_calibration
Measures how close the LLM grader is to human-labelled score ranges and how stable it is."""
import json
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import memory, skills                       # noqa: E402
from agent.interview import build_interview_graph       # noqa: E402
from agent.llm import LLM                               # noqa: E402

HERE = os.path.dirname(__file__)


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def grade_once(llm, profile, taxonomy, item):
    """Run a 1-question session where the 'question' is forced to the calibration item."""
    graph = build_interview_graph(llm, {**profile, "max_followups_per_question": 0}, taxonomy,
                                  interrupt_before=["wrap_up", "ask_question", "follow_up"])
    target = next(k for k, t in profile["targets"].items() if item["skill"] in t["skills"]) if any(
        item["skill"] in t["skills"] for t in profile["targets"].values()) else next(iter(profile["targets"]))
    sid = memory.new_id()
    cfg = {"configurable": {"thread_id": sid}}
    q = {"skill": item["skill"], "subtopic": "calibration", "difficulty": item["difficulty"], "question": item["question"],
         "expected_points": item["expected_points"], "red_flags": [], "is_followup": False}
    # start directly at evaluation: inject state, then resume with the reference answer
    graph.update_state(cfg, {"session_id": sid, "target": target, "n": 1, "queue": [], "results": [],
                             "followups_used": 0, "current": q, "last": None, "jd": None, "focus": []}, as_node="ask_question")
    from langgraph.types import Command
    graph.invoke(None, cfg)                                  # runs to the answer interrupt
    graph.invoke(Command(resume=item["answer"]), cfg)        # grades, then pauses before wrap-up
    st = graph.get_state(cfg).values
    return st["results"][0]["score"]


def main(llm=None):
    real_db = memory.DB_PATH
    memory.DB_PATH = os.path.join(tempfile.mkdtemp(), "calib.db")   # never touch your real progress
    try:
        return _main(llm)
    finally:
        memory.DB_PATH = real_db


def _main(llm=None):
    profile = skills.load_yaml("profile.yaml")
    taxonomy = skills.load_yaml("skills.yaml")
    memory.ensure_mastery(profile, taxonomy)
    llm = llm or LLM()
    items = yaml.safe_load(open(os.path.join(HERE, "calibration.yaml")))["items"]
    rows = []
    for it in items:
        s1 = grade_once(llm, profile, taxonomy, it)
        s2 = grade_once(llm, profile, taxonomy, it)
        lo, hi = it["expected_range"]
        mid = (lo + hi) / 2
        avg = (s1 + s2) / 2
        rows.append({"id": it["id"], "expected": f"{lo}-{hi}", "score_1": s1, "score_2": s2,
                     "in_range": lo <= avg <= hi, "abs_err": round(max(0, lo - avg, avg - hi), 2), "mid": mid, "avg": avg})
        print(f"{it['id']:<20} expected {lo:>2}-{hi:<2}  got {s1:>4} / {s2:<4}  {'OK' if lo <= avg <= hi else 'MISS'}")
    res = {
        "n_items": len(rows),
        "within_range": round(sum(r["in_range"] for r in rows) / len(rows), 3),
        "mae_outside_range": round(sum(r["abs_err"] for r in rows) / len(rows), 3),
        "spearman_vs_human": round(spearman([r["mid"] for r in rows], [r["avg"] for r in rows]), 3),
        "regrade_drift": round(sum(abs(r["score_1"] - r["score_2"]) for r in rows) / len(rows), 3),
        "provider": llm.last_provider, "rows": rows,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    json.dump(res, open(os.path.join(HERE, "calibration_results.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
