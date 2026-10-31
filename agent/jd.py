"""Job-description analyzer: extracts requirements, maps them to the skill taxonomy and
compares them with your profile + assessed mastery to produce a gap analysis."""
from __future__ import annotations

import json

from . import memory
from .llm import LLM


def analyze(llm: LLM, profile: dict, taxonomy: dict, jd_text: str, title: str = "", company: str = "") -> dict:
    keys = {k: v["label"] for k, v in taxonomy.items()}
    prompt = (f"TASK: JD\nJOB DESCRIPTION:\n{jd_text[:7000]}\n\nCANDIDATE: skills {', '.join(profile['skills'])}; "
              f"experience: {' | '.join(profile['experience'])}\nSKILL TAXONOMY KEYS: {json.dumps(keys)}\n\n"
              "Extract the requirements and compare them with the candidate. Return JSON: {\"title\": str, \"company\": str, "
              "\"required_skills\": [{\"name\": str, \"importance\": \"must|nice\", \"taxonomy\": one key from the taxonomy or null, "
              "\"candidate_evidence\": \"where the candidate shows it, or empty\"}], \"responsibilities\": [str], "
              "\"likely_interview_topics\": [str], \"resume_keywords_missing\": [str], \"summary\": \"2 sentences on fit\"}")
    a = llm.chat_json(prompt, max_tokens=2500)
    a["title"] = title or a.get("title", "Untitled role")
    a["company"] = company or a.get("company", "")
    mastery = memory.get_mastery()
    mapped, gaps = [], []
    for r in a.get("required_skills", []):
        t = r.get("taxonomy")
        if t in taxonomy:
            mapped.append(t)
            r["mastery"] = round(mastery[t]["value"], 2)
        if not r.get("candidate_evidence") or (t in mastery and mastery[t]["value"] < 0.6):
            gaps.append(r)
    a["mapped_skills"] = sorted(set(mapped), key=mapped.count, reverse=True)
    a["gaps"] = gaps
    a["required_skills_flat"] = [r["name"] for r in a.get("required_skills", [])]
    must = [r for r in a.get("required_skills", []) if r.get("importance") == "must"]
    covered = [r for r in must if r.get("candidate_evidence")]
    a["must_have_coverage"] = round(len(covered) / len(must), 2) if must else None
    a["id"] = memory.new_id()
    memory.execute("INSERT INTO jd_analyses VALUES(?,?,?,?,?,?)",
                   (a["id"], memory.now(), a["title"], a["company"], jd_text[:10000], json.dumps(a)))
    return a


def saved() -> list[dict]:
    return [json.loads(r["analysis_json"]) for r in memory.rows("SELECT analysis_json FROM jd_analyses ORDER BY ts DESC")]
