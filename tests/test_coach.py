"""Offline end-to-end test of the coach:  python -m tests.test_coach"""
import inspect, json, os, sys, tempfile
os.environ["COACH_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["COACH_FIXTURES"] = os.path.join(os.path.dirname(__file__), "fixture_jobs.json")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import coding, evaluate, interview, jd, memory, opportunities, report, skills  # noqa: E402
from agent.llm import LLM                                                                # noqa: E402
from tests import reference_solutions as R                                               # noqa: E402
from tests.mock_llm import mock_llm                                                      # noqa: E402

llm = LLM(mock=mock_llm)
profile, taxonomy = skills.load_yaml("profile.yaml"), skills.load_yaml("skills.yaml")
memory.ensure_mastery(profile, taxonomy)
m0 = {k: v["value"] for k, v in memory.get_mastery().items()}

# --- interview: weak answer -> follow-up -> good answer; skip; good
g = interview.build_interview_graph(llm, profile, taxonomy)
sid, out = interview.start(g, "ml_ai_intern", 3, focus=["dsa"])
p = interview.pending(out); assert p and p["index"] == 1, p
out = interview.answer(g, sid, "short answer")
p = interview.pending(out); assert p["question"]["is_followup"], "weak answer must trigger a follow-up"
out = interview.answer(g, sid, "a detailed answer")
p = interview.pending(out); assert not p["question"]["is_followup"] and p["index"] == 2, p
out = interview.answer(g, sid, "skip")
assert out["__interrupt__"][0].value["last"]["score"] == 0.0
p = interview.pending(out); assert p["question"]["is_followup"]       # skip is weak → probe once
out = interview.answer(g, sid, "a detailed answer")
out = interview.answer(g, sid, "a detailed answer")                   # Q3
assert interview.pending(out) is None and out["summary"]["avg_score"] > 0
print("interview ok:", [r["score"] for r in out["results"]], out["summary"]["avg_score"])
assert memory.rows("SELECT COUNT(*) n FROM sessions")[0]["n"] == 1
assert memory.rows("SELECT COUNT(*) n FROM answers")[0]["n"] == 5
print(out["transcript_md"][:600])

# --- coding lab: reference solutions pass every stored test; bad code fails; infinite loop times out
for prob in coding.problems():
    src = inspect.getsource(getattr(R, prob["id"]))
    header = "from collections import Counter, defaultdict, deque\nimport statistics\n"
    res = coding.run_tests(prob, header + src)
    assert res["passed"] == res["total"], (prob["id"], res)
bad = coding.run_tests(coding.problems()[0], "def two_sum(nums, target):\n    return [0, 0]\n")
assert 0 <= bad["passed"] < bad["total"]
loop = coding.run_tests(coding.problems()[0], "def two_sum(nums, target):\n    while True: pass\n", timeout=2)
assert "Timed out" in loop["error"]
syn = coding.run_tests(coding.problems()[0], "def two_sum(:\n")
assert "SyntaxError" in syn["error"]
prob = coding.problems()[0]
res = coding.run_tests(prob, header + inspect.getsource(R.two_sum))
score = coding.grade_and_record(prob, "code", res, coding.review(llm, prob, "code", res, "faang_strict"))
assert score == 9.4, score
coding.GENERATED = os.path.join(tempfile.mkdtemp(), "gen.json")
newp = coding.generate_problem(llm, "arrays & hashing", "easy", "ML")
assert len(newp["tests"]) == 3, "the wrong test (expected 999) must be dropped by self-verification"
print("coding ok")

# --- opportunities: filters + track classification
jobs, health = opportunities.scan(profile, skills.load_yaml("opportunities.yaml"))
titles = {j["title"]: j["track"] for j in jobs}
assert titles == {"Machine Learning Engineer Intern (Summer 2027)": "ml_ai_intern", "Security Engineering Intern": "security_intern"}, titles
jobs = opportunities.llm_fit(llm, profile, jobs); opportunities.save(jobs)
opportunities.set_status(jobs[0]["id"], "saved"); opportunities.set_status(jobs[1]["id"], "dismissed")
print("opportunities ok:", [(j["title"], j["fit"]) for j in jobs])

# --- JD analysis
a = jd.analyze(llm, profile, taxonomy, "TEST JD text", "ML Intern", "TestCo")
assert a["must_have_coverage"] == 0.5 and "ml_system_design" in a["mapped_skills"]
sid2, out2 = interview.start(g, "ml_ai_intern", 2, jd=a)
assert interview.pending(out2) is not None
print("jd ok")

# --- report + docx
md, d = report.build(llm, profile, taxonomy)
assert "Readiness snapshot" in md and "14-day preparation plan" in md
path = report.to_docx(md, os.path.join(tempfile.mkdtemp(), "r.docx"))
from docx import Document
doc = Document(path)
fonts = {r.font.name for p in doc.paragraphs for r in p.runs}
sizes = {r.font.size.pt for p in doc.paragraphs for r in p.runs if r.font.size}
assert fonts == {"Times New Roman"} and sizes <= {16, 14, 12}, (fonts, sizes)
print("report ok:", sizes)

# --- evaluation
memory.rate_question(memory.rows("SELECT id FROM answers LIMIT 1")[0]["id"], 1)
mt = evaluate.metrics()
assert mt["opportunity_precision"] == 0.5 and mt["followup_gain"] > 0 and mt["question_relevance"] == 1
print("metrics:", {k: v for k, v in mt.items()})
assert not evaluate.readiness_history(profile).empty

# --- calibration harness runs and does not touch the real DB
real_n = memory.rows("SELECT COUNT(*) n FROM answers")[0]["n"]
from eval import run_calibration
res = run_calibration.main(llm)
assert memory.rows("SELECT COUNT(*) n FROM answers")[0]["n"] == real_n, "calibration must not write to the user DB"
print("calibration ok:", {k: v for k, v in res.items() if k != "rows"})
m1 = {k: v["value"] for k, v in memory.get_mastery().items()}
print("mastery change:", {k: round(m1[k] - m0[k], 3) for k in m0})
print("ALL TESTS PASSED")
