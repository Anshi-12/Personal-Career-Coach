"""Evaluation of the coach: grader accuracy, question quality, opportunity precision, user progress."""
from __future__ import annotations

import json
import os

import pandas as pd

from . import memory

CALIB = os.path.join(os.path.dirname(__file__), "..", "eval", "calibration_results.json")


def mastery_history() -> pd.DataFrame:
    h = pd.DataFrame(memory.rows("SELECT skill, value, ts FROM mastery_history ORDER BY ts"))
    if h.empty:
        return h
    h["ts"] = pd.to_datetime(h["ts"])
    return h.pivot_table(index="ts", columns="skill", values="value", aggfunc="last").ffill()


def readiness_history(profile: dict) -> pd.DataFrame:
    """Replay mastery history to reconstruct role readiness over time."""
    hist = memory.rows("SELECT skill, value, ts FROM mastery_history ORDER BY ts")
    cur, out = {}, []
    for r in hist:
        cur[r["skill"]] = r["value"]
        row = {"ts": r["ts"]}
        for k, t in profile["targets"].items():
            if all(s in cur for s in t["skills"]):
                row[t["label"]] = sum(w * cur[s] for s, w in t["skills"].items()) / sum(t["skills"].values())
        out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").groupby(level=0).last().ffill()


def session_trend() -> pd.DataFrame:
    s = pd.DataFrame(memory.rows("SELECT ts, avg_score, target FROM sessions ORDER BY ts"))
    if s.empty:
        return s
    s["ts"] = pd.to_datetime(s["ts"])
    return s.set_index("ts")[["avg_score"]]


def metrics() -> dict:
    ans = memory.rows("SELECT * FROM answers WHERE score>=0")
    rated = [a for a in ans if a["q_rating"] is not None]
    # did follow-up probing lead to a better answer on the same topic?
    fu_gain = []
    for i, a in enumerate(ans):
        if a["is_followup"] and i > 0 and ans[i - 1]["session_id"] == a["session_id"]:
            fu_gain.append(a["score"] - ans[i - 1]["score"])
    opp = {r["status"]: r["n"] for r in memory.rows("SELECT status, COUNT(*) n FROM opportunities GROUP BY status")}
    judged = opp.get("saved", 0) + opp.get("applied", 0) + opp.get("dismissed", 0)
    sessions = memory.rows("SELECT avg_score FROM sessions ORDER BY ts")
    half = len(sessions) // 2
    first = sum(s["avg_score"] for s in sessions[:half]) / half if half else None
    last = sum(s["avg_score"] for s in sessions[half:]) / (len(sessions) - half) if len(sessions) > 1 else None
    coding = memory.rows("SELECT passed, total, score FROM coding_attempts")
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else None
    return {
        "answers": len(ans), "sessions": len(sessions),
        "avg_score": round(sum(a["score"] for a in ans) / len(ans), 2) if ans else None,
        "score_trend": round(last - first, 2) if first is not None and last is not None else None,
        "question_relevance": round(sum(a["q_rating"] for a in rated) / len(rated), 2) if rated else None,
        "questions_rated": len(rated),
        "followup_gain": round(sum(fu_gain) / len(fu_gain), 2) if fu_gain else None,
        "opportunity_precision": round((opp.get("saved", 0) + opp.get("applied", 0)) / judged, 2) if judged else None,
        "opportunities_judged": judged,
        "coding_pass_rate": round(sum(c["passed"] for c in coding) / max(1, sum(c["total"] for c in coding)), 2) if coding else None,
        "calibration": {k: v for k, v in (calib or {}).items() if k != "rows"} or None,
    }
