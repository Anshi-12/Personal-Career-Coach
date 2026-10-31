"""Long-term memory for the career coach (SQLite): skill mastery, every answer and its score,
sessions, coding attempts, opportunities and JD analyses. Progress tracking reads from here."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime

DB_PATH = os.getenv("COACH_DB", os.path.join(os.path.dirname(__file__), "..", "data", "coach.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS mastery(skill TEXT PRIMARY KEY, value REAL, n INTEGER, updated TEXT);
CREATE TABLE IF NOT EXISTS mastery_history(skill TEXT, value REAL, ts TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, ts TEXT, target TEXT, jd_id TEXT, n_questions INTEGER,
  avg_score REAL, summary_json TEXT, transcript_md TEXT);
CREATE TABLE IF NOT EXISTS answers(id TEXT PRIMARY KEY, session_id TEXT, ts TEXT, skill TEXT, subtopic TEXT,
  difficulty TEXT, is_followup INTEGER, question TEXT, answer TEXT, score REAL, eval_json TEXT, q_rating INTEGER);
CREATE TABLE IF NOT EXISTS coding_attempts(id TEXT PRIMARY KEY, ts TEXT, problem_id TEXT, title TEXT, difficulty TEXT,
  passed INTEGER, total INTEGER, review_json TEXT, score REAL, code TEXT);
CREATE TABLE IF NOT EXISTS opportunities(id TEXT PRIMARY KEY, title TEXT, company TEXT, location TEXT, url TEXT,
  source TEXT, category TEXT, track TEXT, posted TEXT, fit REAL, fit_json TEXT, status TEXT, found TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS jd_analyses(id TEXT PRIMARY KEY, ts TEXT, title TEXT, company TEXT, jd_text TEXT, analysis_json TEXT);
CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY, ts TEXT, markdown TEXT, readiness_json TEXT);
"""


def conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rows(sql: str, args: tuple = ()) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args)]


def execute(sql: str, args: tuple = ()) -> None:
    with conn() as c:
        c.execute(sql, args)


# ------------------------------------------------------------- mastery
def ensure_mastery(profile: dict, skills: dict) -> None:
    with conn() as c:
        for s in skills:
            if not c.execute("SELECT 1 FROM mastery WHERE skill=?", (s,)).fetchone():
                v = float(profile.get("initial_mastery", {}).get(s, 0.4))
                c.execute("INSERT INTO mastery VALUES(?,?,?,?)", (s, v, 0, now()))
                c.execute("INSERT INTO mastery_history VALUES(?,?,?,?)", (s, v, now(), "prior"))


def get_mastery() -> dict[str, dict]:
    return {r["skill"]: {"value": r["value"], "n": r["n"]} for r in rows("SELECT * FROM mastery")}


def set_mastery(skill: str, value: float, source: str) -> None:
    with conn() as c:
        c.execute("UPDATE mastery SET value=?, n=n+1, updated=? WHERE skill=?", (value, now(), skill))
        c.execute("INSERT INTO mastery_history VALUES(?,?,?,?)", (skill, value, now(), source))


# ------------------------------------------------------------- answers & sessions
def save_answer(session_id: str, q: dict, answer: str, score: float, ev: dict) -> str:
    aid = new_id()
    execute("INSERT INTO answers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, session_id, now(), q["skill"], q.get("subtopic"), q.get("difficulty"), int(q.get("is_followup", False)),
             q["question"], answer, score, json.dumps(ev), None))
    return aid


def recent_questions(skill: str, n: int = 12) -> list[str]:
    return [r["question"] for r in rows("SELECT question FROM answers WHERE skill=? ORDER BY ts DESC LIMIT ?", (skill, n))]


def subtopic_counts(skill: str) -> dict[str, int]:
    return {r["subtopic"]: r["n"] for r in rows("SELECT subtopic, COUNT(*) n FROM answers WHERE skill=? GROUP BY subtopic", (skill,))}


def save_session(sid: str, target: str, jd_id: str | None, n: int, avg: float, summary: dict, transcript: str) -> None:
    execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?,?,?)",
            (sid, now(), target, jd_id, n, avg, json.dumps(summary), transcript))


def rate_question(answer_id: str, rating: int) -> None:
    execute("UPDATE answers SET q_rating=? WHERE id=?", (rating, answer_id))
