"""Deterministic fake LLM so the coach can be tested offline with no API key."""
import json
import re


def mock_llm(system: str, prompt: str) -> str:
    if "TASK: QUESTION" in prompt:
        sub = re.search(r"Subtopic: (.*?)\. Difficulty", prompt).group(1)
        return json.dumps({"question": f"[test] Explain {sub}.", "expected_points": ["p1", "p2", "p3"], "red_flags": ["r1"]})
    if "TASK: EVALUATE" in prompt:
        ans = re.search(r"CANDIDATE ANSWER: (.*?)\n\nGrade", prompt, re.S).group(1)
        good = "detailed" in ans
        sc = {"correctness": 4, "depth": 3, "communication": 2, "tradeoffs": 1} if good else \
             {"correctness": 2, "depth": 1, "communication": 1, "tradeoffs": 0}
        return json.dumps({"scores": sc, "verdict": "hire" if good else "lean no hire", "strengths": ["s"],
                           "gaps": [] if good else ["missing depth"], "missing_points": [] if good else ["p2"],
                           "feedback": "fb", "model_answer": "ideal", "follow_up": "" if good else "[test] Go deeper on p2?",
                           "explain_concept": "" if good else "Think of it like this: a tiny lesson."})
    if "TASK: SUMMARY" in prompt:
        return json.dumps({"overall_verdict": "lean hire", "strengths": ["a"], "gaps": ["b"], "next_steps": ["c", "d", "e"]})
    if "TASK: CODE_REVIEW" in prompt:
        return json.dumps({"time_complexity": "O(n)", "space_complexity": "O(n)", "optimal_complexity": "O(n)",
                           "is_optimal": True, "code_quality": 8, "bugs": [], "edge_cases_missed": [],
                           "feedback": "Clean.", "interviewer_followups": ["q1", "q2"]})
    if "TASK: FIT" in prompt:
        return json.dumps({"fit": 80, "why": "good", "matching": ["Python"], "missing": ["Go"], "next_action": "x"})
    if "TASK: JD" in prompt:
        return json.dumps({"title": "ML Intern", "company": "TestCo", "required_skills": [
            {"name": "Python", "importance": "must", "taxonomy": "dsa", "candidate_evidence": "projects"},
            {"name": "ML system design", "importance": "must", "taxonomy": "ml_system_design", "candidate_evidence": ""},
            {"name": "Kubernetes", "importance": "nice", "taxonomy": None, "candidate_evidence": ""}],
            "responsibilities": ["r"], "likely_interview_topics": ["t"], "resume_keywords_missing": ["k8s"], "summary": "ok"})
    if "TASK: PLAN" in prompt:
        return json.dumps({"plan": [{"days": "Day 1-2", "focus": "DSA", "tasks": ["t1"], "done_when": "d"}]})
    if "TASK: IDEAS" in prompt:
        return json.dumps({"ideas": [{"name": "n", "problem": "p", "solution": "s", "why_you": "w",
                                      "first_step_this_week": "f", "biggest_risk": "r"}]})
    if "TASK: GEN_PROBLEM" in prompt:
        return json.dumps({"id": "sum_list", "title": "Sum list", "prompt": "Sum it.", "signature": "def sum_list(xs: list) -> int:",
                           "reference_solution": "def sum_list(xs):\n    return sum(xs)\n",
                           "tests": [{"args": [[1, 2]], "expected": 3}, {"args": [[]], "expected": 0},
                                     {"args": [[5]], "expected": 5}, {"args": [[1]], "expected": 999}]})
    return "{}"
