"""
Interactive mock-interview agent (LangGraph + human-in-the-loop interrupts).

plan_session → ask_question → await_answer ⏸ → evaluate_answer ─┬─ weak answer & follow-ups left → follow_up → await_answer ⏸
                                                               ├─ more questions → ask_question
                                                               └─ done → wrap_up → END
"""
from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from . import memory, skills
from .llm import LLM, LLMError

PERSONAS = {
    "faang_strict": ("You are a bar-raiser interviewer at a top-tier tech company (FAANG level). You are rigorous, precise "
                     "and direct. You never give credit for vague, buzzword-heavy or partially correct answers. You probe "
                     "for depth, complexity, trade-offs, edge cases and real ownership."),
    "tough_fair": "You are a demanding but fair senior interviewer. You push on weak spots and explain what was missing.",
    "supportive": "You are an encouraging interview coach. You are honest about gaps but frame feedback constructively.",
}

RUBRIC = {
    "default": ("correctness 0-4 (technically right, no errors), depth 0-3 (mechanisms, not just definitions), "
                "communication 0-2 (structured, concise, precise terms), tradeoffs 0-1 (limitations, alternatives, edge cases)"),
    "behavioral": ("correctness 0-4 (complete STAR: clear situation, task, YOUR actions, result; answers the actual question), "
                   "depth 0-3 (specific details, real ownership, 'I' not 'we'), communication 0-2 (structured, under ~2 minutes spoken), "
                   "tradeoffs 0-1 (measurable impact + reflection on what you'd do differently)"),
}

SKIP = {"", "skip", "pass", "idk", "i don't know", "i dont know", "no idea"}


class S(TypedDict, total=False):
    session_id: str
    target: str
    focus: list
    jd: dict | None
    n: int
    queue: list
    current: dict
    followups_used: int
    pending_answer: str
    results: list
    last: dict | None
    summary: dict
    transcript_md: str


def candidate_summary(profile: dict) -> str:
    return (f"Candidate: {profile['name']}, {profile['program']}. Targeting {profile['target_season']} internships.\n"
            f"Experience:\n- " + "\n- ".join(profile["experience"]) + f"\nSkills: {', '.join(profile['skills'])}")


def build_interview_graph(llm: LLM, profile: dict, taxonomy: dict, checkpointer=None, interrupt_before=None):
    style = profile.get("coach_style", "faang_strict")
    persona = PERSONAS.get(style, PERSONAS["tough_fair"])

    # ------------------------------------------------------------ nodes
    def plan_session(state: S) -> S:
        focus = list(state.get("focus") or [])
        if state.get("jd"):
            focus += [s for s in state["jd"].get("mapped_skills", []) if s in taxonomy]
        pri = dict(skills.priority_skills(profile, state["target"], focus))
        mastery = memory.get_mastery()
        counts, used_sub, queue = {s: 0 for s in pri}, {}, []
        for _ in range(state["n"]):
            s = max(pri, key=lambda k: pri[k] / (1 + counts[k]))
            counts[s] += 1
            asked = memory.subtopic_counts(s)
            for sub in used_sub.get(s, []):
                asked[sub] = asked.get(sub, 0) + 100
            subs = taxonomy[s]["subtopics"]
            low = min(asked.get(x, 0) for x in subs)
            sub = random.choice([x for x in subs if asked.get(x, 0) == low])
            used_sub.setdefault(s, []).append(sub)
            queue.append({"skill": s, "subtopic": sub, "difficulty": skills.pick_difficulty(mastery[s]["value"], style)})
        # warm up with the easiest item, end with behavioral if present (like a real loop)
        queue.sort(key=lambda q: (q["skill"] == "behavioral", {"easy": 0, "medium": 1, "hard": 2}[q["difficulty"]]))
        return {"queue": queue, "results": [], "followups_used": 0, "last": None}

    def ask_question(state: S) -> S:
        item, rest = state["queue"][0], state["queue"][1:]
        jd = state.get("jd")
        jd_ctx = (f"\nThe candidate is preparing for this specific role: {jd.get('title')} at {jd.get('company')}. "
                  f"Key requirements: {', '.join(jd.get('required_skills_flat', [])[:12])}. Ask as this company's interviewer would.") if jd else ""
        kind = {"dsa": "an algorithmic problem to solve verbally: ask for approach, data structures, time/space complexity and edge cases "
                       "(no full code needed)",
                "behavioral": "a behavioral question. Where natural, anchor it in one of the candidate's real experiences listed above",
                "ml_system_design": "a design scenario (design an ML system end to end); tie it to healthcare/anomaly detection or security when natural"
                }.get(item["skill"], "a conceptual or scenario question; where natural, connect it to the candidate's projects (sepsis Transformer, XGBoost churn, RBAC dashboards, SOC triage)")
        prompt = (f"TASK: QUESTION\n{candidate_summary(profile)}\n\nTarget role: {profile['targets'][state['target']]['label']}.{jd_ctx}\n"
                  f"Skill: {taxonomy[item['skill']]['label']}. Subtopic: {item['subtopic']}. Difficulty: {item['difficulty']}.\n"
                  f"Write {kind}. One question only, as you would say it in the interview. It must be answerable in 2-4 minutes.\n"
                  f"Do NOT repeat or paraphrase these previously asked questions:\n- " +
                  "\n- ".join(memory.recent_questions(item["skill"])[:10] or ["(none)"]) +
                  "\nReturn JSON: {\"question\": str, \"expected_points\": [3-6 key points a strong answer covers], "
                  "\"red_flags\": [common mistakes]}")
        try:
            q = llm.chat_json(prompt, system=persona, temperature=0.7)
        except LLMError as e:
            q = {"question": f"(LLM unavailable: {e}) Explain the key ideas of {item['subtopic']}.", "expected_points": []}
        current = {**item, "question": q.get("question", ""), "expected_points": q.get("expected_points", []),
                   "red_flags": q.get("red_flags", []), "is_followup": False}
        return {"current": current, "queue": rest, "followups_used": 0}

    def await_answer(state: S) -> S:
        done = len([r for r in state["results"] if not r["q"].get("is_followup")])
        answer = interrupt({"type": "question", "question": state["current"], "last": state.get("last"),
                            "index": done + (0 if state["current"].get("is_followup") else 1),
                            "total": state["n"]})
        return {"pending_answer": str(answer or "")}

    def evaluate_answer(state: S) -> S:
        q, ans = state["current"], state["pending_answer"].strip()
        rubric = RUBRIC["behavioral" if q["skill"] == "behavioral" else "default"]
        if ans.lower() in SKIP:
            ans_for_llm = "(The candidate did not answer.)"
        else:
            ans_for_llm = ans
        prompt = (f"TASK: EVALUATE\nRole: {profile['targets'][state['target']]['label']}. Difficulty: {q['difficulty']}.\n"
                  f"QUESTION: {q['question']}\nEXPECTED POINTS: {q.get('expected_points')}\nCOMMON MISTAKES: {q.get('red_flags')}\n"
                  f"CANDIDATE ANSWER: {ans_for_llm}\n\n"
                  f"Grade with this rubric: {rubric}.\nCalibrate like a real {'FAANG ' if style == 'faang_strict' else ''}interviewer: "
                  "9-10 only for complete, precise answers that also discuss trade-offs; 7-8 solid hire; 5-6 borderline; "
                  "3-4 significant gaps; 0-2 wrong, off-topic or no answer. Judge only what is written; do not assume knowledge "
                  "the candidate did not show.\nReturn JSON: {\"scores\": {\"correctness\": int, \"depth\": int, \"communication\": int, "
                  "\"tradeoffs\": int}, \"verdict\": \"strong hire|hire|lean no hire|no hire\", \"strengths\": [str], "
                  "\"gaps\": [str], \"missing_points\": [str], \"feedback\": \"2-4 direct sentences\", "
                  "\"model_answer\": \"a concise ideal answer\", \"follow_up\": \"one probing question on the biggest gap, or empty if total >= 8\", "
                  "\"explain_concept\": \"if total < 7: a warm, spoken-style mini-lesson (80-130 words) that teaches the underlying concept "
                  "from first principles with one tiny concrete example, the way a tutor would explain it out loud (no code, no bullet points); "
                  "otherwise empty\"}")
        try:
            ev = llm.chat_json(prompt, system=persona, temperature=0.1)
            sc = ev.get("scores", {})
            parts = [min(4, max(0, float(sc.get("correctness", 0)))), min(3, max(0, float(sc.get("depth", 0)))),
                     min(2, max(0, float(sc.get("communication", 0)))), min(1, max(0, float(sc.get("tradeoffs", 0))))]
            score = 0.0 if ans.lower() in SKIP else round(sum(parts), 1)
        except (LLMError, ValueError, TypeError) as e:
            ev, score = {"feedback": f"Evaluation failed: {e}", "follow_up": ""}, None
        before = after = None
        if score is not None:
            before, after = skills.update_mastery(q["skill"], score, q["difficulty"],
                                                  "followup" if q.get("is_followup") else "interview")
        aid = memory.save_answer(state["session_id"], q, ans, score if score is not None else -1, ev)
        result = {"q": q, "answer": ans, "score": score, "eval": ev, "answer_id": aid,
                  "mastery_before": before, "mastery_after": after}
        return {"results": state["results"] + [result], "last": result}

    def route(state: S) -> str:
        last = state["last"]
        weak = last["score"] is not None and last["score"] < 7
        if weak and last["eval"].get("follow_up") and state["followups_used"] < profile.get("max_followups_per_question", 1):
            return "follow_up"
        return "ask_question" if state["queue"] else "wrap_up"

    def follow_up(state: S) -> S:
        prev, ev = state["current"], state["last"]["eval"]
        current = {**prev, "question": ev["follow_up"], "expected_points": ev.get("missing_points", []), "is_followup": True}
        return {"current": current, "followups_used": state["followups_used"] + 1}

    def wrap_up(state: S) -> S:
        res = [r for r in state["results"] if r["score"] is not None]
        avg = round(sum(r["score"] for r in res) / len(res), 2) if res else 0.0
        digest = "\n".join(f"- [{r['q']['skill']}/{r['q']['difficulty']}{'/follow-up' if r['q'].get('is_followup') else ''}] "
                           f"score {r['score']}/10. Q: {r['q']['question'][:160]} | gaps: {r['eval'].get('gaps')}" for r in res)
        try:
            summary = llm.chat_json(
                f"TASK: SUMMARY\nRole: {profile['targets'][state['target']]['label']}\nSession results:\n{digest}\n\n"
                "Write the interviewer's debrief. Return JSON: {\"overall_verdict\": str, \"strengths\": [str], "
                "\"gaps\": [str], \"next_steps\": [3 concrete actions for this week]}", system=persona)
        except LLMError:
            summary = {"overall_verdict": f"Average {avg}/10", "strengths": [], "gaps": [], "next_steps": []}
        summary["avg_score"] = avg
        md = transcript_markdown(profile, state, res, summary)
        memory.save_session(state["session_id"], state["target"], (state.get("jd") or {}).get("id"),
                            len(res), avg, summary, md)
        out_dir = os.path.join(os.path.dirname(__file__), "..", "sample_output")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"session_{datetime.now():%Y%m%d_%H%M}_{state['session_id'][:6]}.md"), "w") as f:
            f.write(md)
        return {"summary": summary, "transcript_md": md}

    g = StateGraph(S)
    for name, fn in [("plan_session", plan_session), ("ask_question", ask_question), ("await_answer", await_answer),
                     ("evaluate_answer", evaluate_answer), ("follow_up", follow_up), ("wrap_up", wrap_up)]:
        g.add_node(name, fn)
    g.set_entry_point("plan_session")
    g.add_edge("plan_session", "ask_question")
    g.add_edge("ask_question", "await_answer")
    g.add_edge("await_answer", "evaluate_answer")
    g.add_conditional_edges("evaluate_answer", route, {"follow_up": "follow_up", "ask_question": "ask_question", "wrap_up": "wrap_up"})
    g.add_edge("follow_up", "await_answer")
    g.add_edge("wrap_up", END)
    return g.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=interrupt_before)


def transcript_markdown(profile: dict, state: S, res: list, summary: dict) -> str:
    L = [f"# Mock Interview Transcript: {profile['targets'][state['target']]['label']}",
         f"*{datetime.now():%B %d, %Y %H:%M} · candidate: {profile['name']} · coach style: {profile.get('coach_style')} · "
         f"session {state['session_id']}*", ""]
    if state.get("jd"):
        L += [f"**Tailored to:** {state['jd'].get('title')} at {state['jd'].get('company')}", ""]
    for i, r in enumerate(res, 1):
        q, ev = r["q"], r["eval"]
        tag = "Follow-up" if q.get("is_followup") else f"Q{sum(1 for x in res[:i] if not x['q'].get('is_followup'))}"
        L += [f"## {tag} · {q['skill']} · {q.get('subtopic')} · {q['difficulty']}",
              f"**Coach:** {q['question']}", "", f"**Me:** {r['answer'] or '(no answer)'}", "",
              f"**Score:** {r['score']}/10 · **Verdict:** {ev.get('verdict', '')} · rubric {ev.get('scores', {})}",
              f"**Feedback:** {ev.get('feedback', '')}"]
        if ev.get("missing_points"):
            L.append("**Missing:** " + "; ".join(ev["missing_points"]))
        if ev.get("model_answer"):
            L.append(f"**Model answer:** {ev['model_answer']}")
        if ev.get("explain_concept"):
            L.append(f"**Concept explained:** {ev['explain_concept']}")
        if r.get("mastery_before") is not None:
            L.append(f"*Mastery {q['skill']}: {r['mastery_before']:.2f} → {r['mastery_after']:.2f}*")
        L.append("")
    L += ["## Debrief", f"**Average score:** {summary.get('avg_score')}/10 · **Verdict:** {summary.get('overall_verdict', '')}"]
    for k in ("strengths", "gaps", "next_steps"):
        if summary.get(k):
            L.append(f"\n**{k.replace('_', ' ').title()}**")
            L += [f"- {x}" for x in summary[k]]
    return "\n".join(L)


def start(graph, target: str, n: int, focus: list | None = None, jd: dict | None = None) -> tuple[str, Any]:
    sid = memory.new_id()
    cfg = {"configurable": {"thread_id": sid}}
    out = graph.invoke({"session_id": sid, "target": target, "n": n, "focus": focus or [], "jd": jd}, cfg)
    return sid, out


def answer(graph, sid: str, text: str) -> Any:
    return graph.invoke(Command(resume=text), {"configurable": {"thread_id": sid}})


def pending(out: Any) -> dict | None:
    """Return the interrupt payload (the next question) or None if the session is finished."""
    intr = out.get("__interrupt__") if isinstance(out, dict) else None
    return intr[0].value if intr else None
