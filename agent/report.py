"""Personalized Career Readiness Report: Markdown for the app, .docx for submission
(Times New Roman, 16/14/12 pt hierarchy)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from . import jd as jd_mod
from . import memory, skills
from .llm import LLM, LLMError

OUT = os.path.join(os.path.dirname(__file__), "..", "sample_output")


def gather(profile: dict, taxonomy: dict) -> dict:
    mastery = memory.get_mastery()
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    delta = {}
    for s in mastery:
        old = memory.rows("SELECT value FROM mastery_history WHERE skill=? AND ts<=? ORDER BY ts DESC LIMIT 1", (s, week_ago))
        first = memory.rows("SELECT value FROM mastery_history WHERE skill=? ORDER BY ts LIMIT 1", (s,))
        base = (old or first or [{"value": mastery[s]["value"]}])[0]["value"]
        delta[s] = round(mastery[s]["value"] - base, 3)
    ans = memory.rows("SELECT * FROM answers WHERE score>=0 ORDER BY ts")
    sessions = memory.rows("SELECT * FROM sessions ORDER BY ts")
    coding = memory.rows("SELECT * FROM coding_attempts ORDER BY ts")
    ready = skills.readiness(profile)
    # importance of a skill across both targets
    imp = {s: sum(t["skills"].get(s, 0) for t in profile["targets"].values()) for s in taxonomy}
    ranked = sorted(mastery, key=lambda s: mastery[s]["value"], reverse=True)
    strengths = [s for s in ranked if mastery[s]["value"] >= 0.6][:3]
    gaps = sorted([s for s in mastery if imp.get(s, 0) > 0], key=lambda s: (mastery[s]["value"] - 0.15 * imp[s]))[:3]
    opps = memory.rows("SELECT * FROM opportunities WHERE status!='dismissed' ORDER BY fit DESC LIMIT 6")
    pipeline = {r["status"]: r["n"] for r in memory.rows("SELECT status, COUNT(*) n FROM opportunities GROUP BY status")}
    return {"mastery": mastery, "delta": delta, "answers": ans, "sessions": sessions, "coding": coding,
            "readiness": ready, "strengths": strengths, "gaps": gaps, "jds": jd_mod.saved()[:3],
            "opps": opps, "pipeline": pipeline}


def plan(llm: LLM, profile: dict, taxonomy: dict, d: dict) -> list[dict]:
    weak = [{"skill": taxonomy[s]["label"], "mastery": round(d["mastery"][s]["value"], 2),
             "recent_gaps": [g for a in d["answers"][-30:] if a["skill"] == s for g in json.loads(a["eval_json"]).get("gaps", [])][:6]}
            for s in d["gaps"]]
    jd_gaps = [g.get("name") for j in d["jds"] for g in j.get("gaps", [])][:10]
    prompt = (f"TASK: PLAN\nCandidate: {profile['name']} targeting {', '.join(t['label'] for t in profile['targets'].values())} "
              f"for {profile['target_season']}.\nWeakest areas with observed gaps: {json.dumps(weak)}\nGaps vs saved job descriptions: {jd_gaps}\n"
              "Create a realistic 14-day preparation plan for a full-time graduate student (60-90 min/day). Mix coding practice, "
              "concept review, mock-interview sessions with this coach and one portfolio action. Return JSON: "
              "{\"plan\": [{\"days\": \"Day 1-2\", \"focus\": str, \"tasks\": [str], \"done_when\": \"measurable outcome\"}]}")
    try:
        return llm.chat_json(prompt, temperature=0.4).get("plan", [])
    except LLMError:
        return [{"days": f"Day {2*i+1}-{2*i+2}", "focus": taxonomy[s]["label"],
                 "tasks": [f"Study: {taxonomy[s]['resources'][0]['title']}", "Run a 4-question coach session on this skill"],
                 "done_when": "Session average ≥ 7/10"} for i, s in enumerate(d["gaps"])]


def build(llm: LLM, profile: dict, taxonomy: dict) -> tuple[str, dict]:
    d = gather(profile, taxonomy)
    d["plan"] = plan(llm, profile, taxonomy, d)
    lab = lambda s: taxonomy[s]["label"]  # noqa: E731
    L = [f"# Career Readiness Report: {profile['name']}",
         f"*{profile['program']} · generated {datetime.now():%B %d, %Y} · target: {profile['target_season']} internships*", "",
         "## 1. Readiness snapshot"]
    for r in d["readiness"].values():
        L.append(f"- **{r['label']}:** {r['score']*100:.0f}/100 · {r['band']} (evidence: {r['evidence']} graded answers, confidence {r['confidence']})")
    n_ans, n_sess = len(d["answers"]), len(d["sessions"])
    avg = sum(a["score"] for a in d["answers"]) / n_ans if n_ans else 0
    L += ["", f"Assessed through **{n_sess} mock-interview sessions**, **{n_ans} graded answers** (average {avg:.1f}/10) "
              f"and **{len(d['coding'])} coding-lab attempts**.", "", "## 2. Technical assessment by skill",
          "| Skill | Mastery | 7-day change | Graded answers |", "|---|---|---|---|"]
    for s, m in sorted(d["mastery"].items(), key=lambda x: -x[1]["value"]):
        ch = d["delta"][s]
        L.append(f"| {lab(s)} | {m['value']*100:.0f}% | {'+' if ch >= 0 else ''}{ch*100:.0f} pts | {m['n']} |")
    if d["sessions"]:
        L += ["", "**Session trend (avg score /10):** " + " → ".join(f"{s['avg_score']:.1f}" for s in d["sessions"][-8:])]
    if d["answers"]:
        best = sorted(d["answers"], key=lambda a: -a["score"])[:2]
        worst = sorted(d["answers"], key=lambda a: a["score"])[:2]
        L += ["", "**Strongest answers**"]
        L += [f"- ({a['score']:.0f}/10, {lab(a['skill'])}) {a['question'][:140]}" for a in best]
        L += ["", "**Weakest answers and what was missing**"]
        for a in worst:
            ev = json.loads(a["eval_json"])
            L.append(f"- ({a['score']:.0f}/10, {lab(a['skill'])}) {a['question'][:140]}. *Missing:* {'; '.join(ev.get('missing_points', [])[:3]) or ev.get('feedback', '')[:160]}")
    if d["coding"]:
        p = sum(c["passed"] for c in d["coding"]); t = sum(c["total"] for c in d["coding"])
        L += ["", f"**Coding lab:** {len(d['coding'])} attempts, {p}/{t} tests passed ({p/max(1,t)*100:.0f}%), "
                  f"avg score {sum(c['score'] for c in d['coding'])/len(d['coding']):.1f}/10."]
    L += ["", "## 3. Strengths"]
    L += [f"- **{lab(s)}** ({d['mastery'][s]['value']*100:.0f}%)" for s in d["strengths"]] or ["- Not enough evidence yet. Complete 2–3 sessions."]
    L += ["", "## 4. Knowledge gaps and recommended resources"]
    for s in d["gaps"]:
        recent = [g for a in d["answers"][-40:] if a["skill"] == s for g in json.loads(a["eval_json"]).get("gaps", [])][:3]
        L.append(f"**{lab(s)}** ({d['mastery'][s]['value']*100:.0f}%)" + (f": observed gaps: {'; '.join(recent)}" if recent else ""))
        L += [f"- [{r['title']}]({r['url']})" for r in taxonomy[s]["resources"][:3]]
        L.append("")
    if d["jds"]:
        L += ["## 5. Alignment with target job descriptions"]
        for j in d["jds"]:
            cov = j.get("must_have_coverage")
            L.append(f"- **{j['title']}** ({j.get('company', '')}): must-have coverage {cov*100:.0f}%" if cov is not None else f"- **{j['title']}**")
            if j.get("gaps"):
                L.append(f"  - Gaps: {', '.join(g['name'] for g in j['gaps'][:6])}")
            if j.get("resume_keywords_missing"):
                L.append(f"  - Resume keywords to add (only if true): {', '.join(j['resume_keywords_missing'][:6])}")
        L.append("")
    L += ["## 6. 14-day preparation plan"]
    for p in d["plan"]:
        L.append(f"**{p.get('days')}: {p.get('focus')}**")
        L += [f"- {t}" for t in p.get("tasks", [])]
        L.append(f"- *Done when:* {p.get('done_when', '')}")
    L += ["", "## 7. Opportunity pipeline",
          "Status: " + (", ".join(f"{k}: {v}" for k, v in d["pipeline"].items()) or "no opportunities scanned yet")]
    for o in d["opps"]:
        L.append(f"- **{o['title']}**, {o['company']} ({o['location'] or 'n/a'}): fit {o['fit']:.0f}/100 · {o['status']} · {o['url']}")
    md = "\n".join(L)
    rid = memory.new_id()
    memory.execute("INSERT INTO reports VALUES(?,?,?,?)", (rid, memory.now(), md, json.dumps(d["readiness"])))
    return md, d


def to_docx(md: str, path: str) -> str:
    """Convert the report Markdown into a Word file: Times New Roman, 16pt H1/H2, 14pt subheads, 12pt body."""
    import re

    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    def runs(p, text, size=12, bold=False):
        for i, part in enumerate(re.split(r"(\*\*[^*]+\*\*)", re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text))):
            if not part:
                continue
            b = part.startswith("**") and part.endswith("**")
            r = p.add_run(part.strip("*") if b else part.replace("*", ""))
            r.font.name, r.font.size, r.bold = "Times New Roman", Pt(size), bold or b
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    table_rows = []

    def flush_table():
        if not table_rows:
            return
        t = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        t.style = "Table Grid"
        for i, row in enumerate(table_rows):
            for j, cell in enumerate(row):
                c = t.cell(i, j)
                c.text = ""
                runs(c.paragraphs[0], cell, 12, bold=(i == 0))
        table_rows.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                table_rows.append(cells)
            continue
        flush_table()
        if line.startswith("# "):
            runs(doc.add_paragraph(), line[2:], 16, True)
        elif line.startswith("## "):
            runs(doc.add_paragraph(), line[3:], 16, True)
        elif line.startswith("**") and line.rstrip().endswith("**") and line.count("**") == 2:
            runs(doc.add_paragraph(), line, 14, True)
        elif line.lstrip().startswith("- "):
            runs(doc.add_paragraph(style="List Bullet"), line.lstrip()[2:])
        elif line.strip():
            runs(doc.add_paragraph(), line)
    flush_table()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc.save(path)
    return path
