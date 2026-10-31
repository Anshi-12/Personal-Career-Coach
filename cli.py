"""Terminal mock interview (handy for recording a demo transcript).

    python cli.py                        # ML/AI intern loop, 6 questions
    python cli.py security_intern 4      # security loop, 4 questions

Multi-line answers: finish with an empty line. Type 'skip' to see the model answer.
The transcript is saved to sample_output/ automatically.
"""
import sys

from agent import interview, memory, skills
from agent.llm import LLM


def read_answer() -> str:
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def show_eval(last: dict) -> None:
    ev = last["eval"]
    print(f"\n  ▶ Score {last['score']}/10 · {ev.get('verdict', '')}")
    print(f"  ▶ {ev.get('feedback', '')}")
    if ev.get("missing_points"):
        print("  ▶ Missing: " + "; ".join(ev["missing_points"]))
    if ev.get("model_answer"):
        print(f"  ▶ Model answer: {ev['model_answer']}")
    if last.get("mastery_before") is not None:
        print(f"  ▶ Mastery {last['q']['skill']}: {last['mastery_before']:.2f} → {last['mastery_after']:.2f}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "ml_ai_intern"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    profile, taxonomy = skills.load_yaml("profile.yaml"), skills.load_yaml("skills.yaml")
    memory.ensure_mastery(profile, taxonomy)
    llm = LLM()
    print(f"LLM chain: {llm.status()}\nMock interview: {profile['targets'][target]['label']} · {n} questions\n")
    graph = interview.build_interview_graph(llm, profile, taxonomy)
    sid, out = interview.start(graph, target, n)
    while (p := interview.pending(out)):
        if p.get("last"):
            show_eval(p["last"])
        q = p["question"]
        tag = "FOLLOW-UP" if q.get("is_followup") else f"Q{p['index']}/{p['total']}"
        print(f"\n[{tag} · {q['skill']} · {q['difficulty']}]\n{q['question']}\n\nYour answer (blank line to submit):")
        out = interview.answer(graph, sid, read_answer())
    show_eval(out["last"])
    s = out["summary"]
    print(f"\n=== Debrief: avg {s['avg_score']}/10 · {s.get('overall_verdict', '')} ===")
    for k in ("strengths", "gaps", "next_steps"):
        print(f"\n{k.upper()}:\n" + "\n".join(f" - {x}" for x in s.get(k, [])))
    print("\nReadiness:", {r["label"]: f"{r['score']*100:.0f}/100" for r in skills.readiness(profile).values()})


if __name__ == "__main__":
    main()
