"""Skill model: mastery updates, adaptive difficulty and role readiness."""
from __future__ import annotations

import os

import yaml

from . import memory

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")

# A perfect answer to an easy question proves less than a perfect answer to a hard one.
DIFFICULTY_CAP = {"easy": 0.70, "medium": 0.85, "hard": 1.00}
BANDS = [(0.85, "Strong: interview-ready for top-tier loops"), (0.70, "Interview-ready"),
         (0.50, "Developing: close specific gaps"), (0.0, "Not ready yet: build fundamentals")]


def load_yaml(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def update_mastery(skill: str, score_10: float, difficulty: str, source: str = "interview") -> tuple[float, float]:
    """Exponential moving average toward the evidence; early answers move it more."""
    m = memory.get_mastery()[skill]
    evidence = (score_10 / 10.0) * DIFFICULTY_CAP.get(difficulty, 0.85)
    # failing an easy question is strong negative evidence; acing it is weak positive evidence
    if difficulty == "easy" and score_10 >= 7:
        evidence = max(evidence, m["value"])  # don't drag a strong candidate down for acing an easy one
    alpha = max(0.15, 1.0 / (m["n"] + 3))
    new = round(min(1.0, max(0.0, m["value"] + alpha * (evidence - m["value"]))), 4)
    memory.set_mastery(skill, new, source)
    return m["value"], new


def pick_difficulty(mastery: float, style: str) -> str:
    bump = 0.1 if style == "faang_strict" else (-0.1 if style == "supportive" else 0.0)
    m = mastery + bump
    return "easy" if m < 0.40 else ("medium" if m < 0.68 else "hard")


def readiness(profile: dict) -> dict[str, dict]:
    m = memory.get_mastery()
    out = {}
    for key, t in profile["targets"].items():
        total = sum(t["skills"].values())
        score = sum(w * m.get(s, {"value": 0})["value"] for s, w in t["skills"].items()) / total
        evidence = sum(m.get(s, {"n": 0})["n"] for s in t["skills"])
        band = next(label for th, label in BANDS if score >= th)
        out[key] = {"label": t["label"], "score": round(score, 3), "band": band, "evidence": evidence,
                    "confidence": "low" if evidence < 10 else ("medium" if evidence < 30 else "high")}
    return out


def priority_skills(profile: dict, target: str, focus: list[str] | None = None) -> list[tuple[str, float]]:
    """Which skills to probe next: role weight × gap, boosted for chosen focus areas."""
    m = memory.get_mastery()
    weights = dict(profile["targets"][target]["skills"])
    for f in focus or []:                         # a focus skill outside the role still gets practice
        if f in m and f not in weights:
            weights[f] = 0.15
    pri = {}
    for s, w in weights.items():
        gap = 1 - m[s]["value"]
        uncertainty = 1 / (1 + m[s]["n"] / 3)       # explore skills with little evidence
        pri[s] = w * (0.7 * gap + 0.3 * uncertainty) * (1.6 if focus and s in focus else 1.0)
    return sorted(pri.items(), key=lambda x: x[1], reverse=True)
