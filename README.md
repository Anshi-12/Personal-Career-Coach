# nextLevelPersonalCoachForCareerLevel

A personal AI career coach for **ML/AI engineering** and **cybersecurity** internships. It covers five jobs:

- **Find opportunities:** scouts internships, research leads and entrepreneurial programs.
- **Analyze job descriptions:** compares each one with your profile.
- **Mock interviews:** runs **FAANG-strict interviews** that grade every answer and probe weak ones with a follow-up.
- **Coding lab:** runs your code against tests and reviews it like an interviewer.
- **Progress tracking:** follows each skill over time and writes a **Career Readiness Report** (.md + .docx).

Built with **LangGraph** (human-in-the-loop interrupts) + **Streamlit** + a **free multi-provider LLM layer** (Gemini → Groq → Cohere → OpenRouter → Ollama).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                  # add GEMINI_API_KEY (free): https://aistudio.google.com/apikey
streamlit run app.py
```

Terminal interview, useful for recording a demo: `python cli.py ml_ai_intern 6` or `python cli.py security_intern 4`.

## Features

| Tab | What the agent does |
|---|---|
| **Dashboard** | Readiness per target role, skill mastery, readiness over time, next best actions |
| **Mock Interview** | **Voice mode:** the coach speaks each question (Edge neural TTS → gTTS), you answer out loud (Groq Whisper → Gemini transcription), and when an answer scores below 7 it speaks a short mini-lesson explaining the concept. LangGraph loop: `plan_session → ask_question → await_answer (interrupt) → evaluate_answer → follow_up / next / wrap_up`. Picks skills by *role weight × gap × uncertainty*. Adapts difficulty to your mastery. Personalizes questions to your real projects (sepsis Transformer, RBAC dashboards, XGBoost, SOC triage, IEEE leadership) or to a chosen job description. Grades with a 4-part rubric (correctness 4, depth 3, communication 2, trade-offs 1). Gives a model answer, and asks one follow-up when you score below 7. Transcripts are saved to `sample_output/` |
| **Coding Lab** | 9 built-in problems, including domain ones: brute-force IP detection (SOC), RBAC role inheritance (IAM) and rolling z-score anomalies on vitals. Tests run in a subprocess with a timeout. LLM code review covers complexity, bugs and follow-ups. **Generate new problems:** the agent writes a reference solution and keeps only the tests it passes (self-verification) |
| **Opportunities** | SimplifyJobs internship list (covers Big Tech), Greenhouse, Lever and Ashby boards, plus optional Adzuna. Filters to internships in your tracks, excludes "US citizenship required" postings, scores fit (heuristic + LLM reading the JD), and tracks status (saved / applied / dismissed). Also finds recent arXiv research in your areas, curated research/CTF/hackathon/startup programs, and generates startup-idea suggestions |
| **JD Analyzer** | Paste any posting (Handshake, Workday, LinkedIn). Extracts must-have vs nice-to-have requirements, maps them to the skill taxonomy, and returns must-have coverage, gaps, likely interview topics and missing resume keywords. One click starts a **tailored mock interview** |
| **Readiness Report** | Readiness snapshot, skill table with 7-day change, strongest and weakest answers, coding results, strengths, gaps with resources, JD alignment, a **14-day preparation plan** and the opportunity pipeline. Exports to Markdown and Word (Times New Roman 16/14/12) |
| **Evaluation** | Grader calibration, question relevance, follow-up gain, opportunity precision, score trend, mastery and readiness charts |

## Customize

- `config/profile.yaml`: experience, skills, target roles & skill weights, starting self-assessment, coach style (`faang_strict | tough_fair | supportive`), filters
- `config/skills.yaml`: skill taxonomy, subtopics, recommended resources
- `config/opportunities.yaml`: company boards (add any Greenhouse/Lever/Ashby slug), career links, research queries, programs
- `config/coding_problems.yaml`: coding problems and tests

## Evaluate the grader

```bash
python -m eval.run_calibration     # ~22 LLM calls; uses a temp DB so your progress is untouched
```

Grades 11 reference answers (strong, medium, weak and skipped, across ML, IAM, DSA, AppSec and behavioral) twice each. It reports % within the human-labelled range, error, Spearman correlation and re-grade drift. Results appear in the Evaluation tab.

## Tests (offline, no key)

```bash
python -m tests.test_coach
```

Checks the interview flow (follow-ups, skip handling, mastery updates) and the coding harness: every reference solution passes, and wrong code, infinite loops and syntax errors are caught. It also checks problem self-verification, opportunity filters, JD gap analysis, report fonts, metrics, and calibration isolation.

## Notes

- Job-board slugs and the Simplify repo name change over time. Failures show in *Source health*; edit `config/opportunities.yaml`.
- Big Tech careers sites (Google, Microsoft, Amazon, Meta, Apple, NVIDIA) have no public API. They're covered through the Simplify list and quick links, and you can paste their postings into the JD Analyzer.
- Your progress lives in `data/coach.db`. Delete it to start over.
