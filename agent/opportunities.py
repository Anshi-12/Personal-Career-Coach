"""Opportunity scout: internships (Simplify list, Greenhouse, Lever, Ashby, optional Adzuna),
research leads (arXiv) and entrepreneurial programs; scored for fit against your profile + mastery."""
from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import requests
from bs4 import BeautifulSoup

from . import memory, skills
from .llm import LLM, LLMError

UA = {"User-Agent": "Mozilla/5.0 (nextLevelPersonalCoach; SJSU student project)"}
T = 25


def _id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:14]


def _txt(html: str, n: int = 6000) -> str:
    return re.sub(r"\s+", " ", BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True))[:n]


def _job(title, company, location, url, source, category, posted, desc="", extra=None) -> dict:
    return {"id": _id(company, title, url), "title": title.strip(), "company": company, "location": location or "",
            "url": url, "source": source, "category": category, "posted": posted, "description": desc, **(extra or {})}


# ------------------------------------------------------------------ fetchers
def fetch_simplify(cfg: dict, season: str) -> list[dict]:
    last_err = None
    for repo in cfg["repos"]:
        for br in cfg.get("branches", ["dev", "main"]):
            try:
                r = requests.get(f"https://raw.githubusercontent.com/{repo}/{br}/.github/scripts/listings.json", headers=UA, timeout=T)
                r.raise_for_status()
                out = []
                for j in r.json():
                    if not j.get("active", True) or not j.get("is_visible", True):
                        continue
                    terms = " ".join(j.get("terms", []) or [j.get("season", "")])
                    if season and season.split()[-1] not in terms:   # match the year, e.g. "2027"
                        continue
                    ts = j.get("date_posted") or j.get("date_updated")
                    posted = datetime.fromtimestamp(ts, timezone.utc).isoformat() if isinstance(ts, (int, float)) else None
                    out.append(_job(j.get("title", ""), j.get("company_name", ""), ", ".join(j.get("locations", [])[:3]),
                                    j.get("url", ""), f"Simplify ({repo.split('/')[1]})", "internship list", posted,
                                    extra={"sponsorship": j.get("sponsorship", "")}))
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise RuntimeError(f"Simplify list unavailable: {last_err}")


def fetch_greenhouse(slug: str, category: str) -> list[dict]:
    r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true", headers=UA, timeout=T)
    r.raise_for_status()
    import html as h
    return [_job(j["title"], slug.title(), (j.get("location") or {}).get("name"), j["absolute_url"], "Greenhouse",
                 category, j.get("updated_at"), _txt(h.unescape(j.get("content", "")))) for j in r.json().get("jobs", [])]


def fetch_lever(slug: str, category: str) -> list[dict]:
    r = requests.get(f"https://api.lever.co/v0/postings/{slug}?mode=json", headers=UA, timeout=T)
    r.raise_for_status()
    return [_job(j["text"], slug.title(), (j.get("categories") or {}).get("location"), j["hostedUrl"], "Lever", category,
                 datetime.fromtimestamp(j["createdAt"] / 1000, timezone.utc).isoformat() if j.get("createdAt") else None,
                 (j.get("descriptionPlain") or "")[:6000]) for j in r.json()]


def fetch_ashby(slug: str, category: str) -> list[dict]:
    r = requests.get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}", headers=UA, timeout=T)
    r.raise_for_status()
    return [_job(j["title"], slug.title(), j.get("location"), j.get("jobUrl"), "Ashby", category, j.get("publishedAt"),
                 (j.get("descriptionPlain") or "")[:6000]) for j in r.json().get("jobs", [])]


def fetch_adzuna(cfg: dict) -> list[dict]:
    app_id, key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and key):
        return []
    out = []
    for q in cfg["queries"]:
        r = requests.get(f"https://api.adzuna.com/v1/api/jobs/{cfg['country']}/search/1",
                         params={"app_id": app_id, "app_key": key, "what": q, "where": cfg["where"], "results_per_page": 20},
                         headers=UA, timeout=T)
        r.raise_for_status()
        out += [_job(j["title"], (j.get("company") or {}).get("display_name", ""), (j.get("location") or {}).get("display_name"),
                     j["redirect_url"], "Adzuna", "search", j.get("created"), j.get("description", "")) for j in r.json().get("results", [])]
    return out


def fetch_research(queries: list[str]) -> list[dict]:
    papers = []
    for q in queries:
        try:
            r = requests.get("https://export.arxiv.org/api/query?search_query=" + quote(q) +
                             "&sortBy=submittedDate&sortOrder=descending&max_results=5", headers=UA, timeout=T)
            for e in feedparser.parse(r.content).entries:
                papers.append({"title": re.sub(r"\s+", " ", e.title), "url": e.link, "published": e.get("published", "")[:10],
                               "authors": ", ".join(a.name for a in e.get("authors", [])[:4]),
                               "abstract": re.sub(r"\s+", " ", e.get("summary", ""))[:600]})
        except Exception:
            continue
    return papers


# ------------------------------------------------------------------ pipeline
def classify_track(title: str, profile: dict) -> str | None:
    t = title.lower()
    best, hits = None, 0
    for key, cfg in profile["targets"].items():
        n = sum(1 for k in cfg["title_keywords"] if re.search(r"\b" + re.escape(k), t))
        if n > hits:
            best, hits = key, n
    return best


def scan(profile: dict, cfg: dict) -> tuple[list[dict], dict]:
    if os.getenv("COACH_FIXTURES"):
        data = json.load(open(os.getenv("COACH_FIXTURES")))
        raw, health = data["jobs"], {"fixture": "ok"}
    else:
        tasks = [("Simplify internship list", lambda: fetch_simplify(cfg["simplify_internships"], profile["target_season"]))]
        fetch = {"greenhouse": fetch_greenhouse, "lever": fetch_lever, "ashby": fetch_ashby}
        for cat, boards in cfg["companies"].items():
            for kind, slugs in boards.items():
                for s in slugs:
                    tasks.append((f"{kind}:{s}", (lambda f=fetch[kind], s=s, c=cat: f(s, c))))
        tasks.append(("Adzuna", lambda: fetch_adzuna(cfg["adzuna"])))
        health, raw = {}, []

        def run(t):
            name, fn = t
            try:
                items = fn()
                health[name] = f"ok ({len(items)})"
                return items
            except Exception as e:  # noqa: BLE001
                health[name] = f"failed: {type(e).__name__}"
                return []

        with ThreadPoolExecutor(max_workers=8) as pool:
            for items in pool.map(run, tasks):
                raw += items

    f = profile.get("opportunity_filters", {})
    locs = [l.lower() for l in profile.get("preferred_locations", [])]
    known = [s.lower() for s in profile["skills"]]
    ready = skills.readiness(profile)
    out, seen = [], set()
    for j in raw:
        title_l = j["title"].lower()
        if f.get("internships_only") and not re.search(r"\bintern|co-?op\b", title_l):
            continue
        track = classify_track(j["title"], profile)
        if not track:
            continue
        text = (j.get("description") or "").lower()
        if f.get("exclude_us_citizenship_required") and (
                "citizenship is required" in (j.get("sponsorship") or "").lower()
                or re.search(r"u\.?s\.? citizen(ship)? (is )?required|security clearance", text)):
            continue
        posted = j.get("posted")
        if posted and f.get("max_age_days"):
            try:
                d = datetime.fromisoformat(posted.replace("Z", "+00:00"))
                d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - d).days > f["max_age_days"]:
                    continue
            except Exception:
                pass
        key = (j["company"].lower(), title_l)
        if key in seen:
            continue
        seen.add(key)
        loc_ok = any(l in j["location"].lower() for l in locs) or not j["location"]
        overlap = sum(1 for s in known if s.split("/")[0] in text) if text else 0
        heur = 40 + 15 * loc_ok + min(25, 4 * overlap) + 20 * ready[track]["score"]
        out.append({**j, "track": track, "fit": round(heur, 1), "fit_json": json.dumps({"method": "heuristic"})})
    out.sort(key=lambda x: x["fit"], reverse=True)
    return out, health


def llm_fit(llm: LLM, profile: dict, jobs: list[dict], top: int = 8) -> list[dict]:
    """Deeper fit assessment for the top-N jobs using the actual description."""
    m = {k: round(v["value"], 2) for k, v in memory.get_mastery().items()}
    for j in jobs[:top]:
        prompt = (f"TASK: FIT\nCandidate: {profile['name']}, {profile['program']}. Experience: {' | '.join(profile['experience'])}.\n"
                  f"Skills: {', '.join(profile['skills'])}. Current assessed mastery (0-1): {m}.\n\n"
                  f"JOB: {j['title']} at {j['company']} ({j['location']}).\nDESCRIPTION: {(j.get('description') or '(title only)')[:4000]}\n\n"
                  "Assess fit honestly. Return JSON: {\"fit\": 0-100, \"why\": \"1-2 sentences\", \"matching\": [str], "
                  "\"missing\": [str], \"next_action\": \"one concrete step before applying\"}")
        try:
            r = llm.chat_json(prompt)
            j["fit"] = round(0.7 * float(r.get("fit", j["fit"])) + 0.3 * j["fit"], 1)
            j["fit_json"] = json.dumps({**r, "method": "llm+heuristic"})
        except (LLMError, ValueError, TypeError):
            pass
    jobs.sort(key=lambda x: x["fit"], reverse=True)
    return jobs


def save(jobs: list[dict]) -> None:
    with memory.conn() as c:
        for j in jobs:
            prev = c.execute("SELECT status FROM opportunities WHERE id=?", (j["id"],)).fetchone()
            c.execute("INSERT OR REPLACE INTO opportunities VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (j["id"], j["title"], j["company"], j["location"], j["url"], j["source"], j["category"], j["track"],
                       j.get("posted"), j["fit"], j["fit_json"], prev["status"] if prev else "new", memory.now(),
                       (j.get("description") or "")[:8000]))


def set_status(opp_id: str, status: str) -> None:
    memory.execute("UPDATE opportunities SET status=? WHERE id=?", (status, opp_id))


def startup_ideas(llm: LLM, profile: dict, papers: list[dict]) -> list[dict]:
    titles = "\n".join(f"- {p['title']}" for p in papers[:12]) or "(no papers fetched)"
    prompt = (f"TASK: IDEAS\nCandidate skills: {', '.join(profile['skills'])}. Experience: {' | '.join(profile['experience'][:4])}.\n"
              f"Recent research in their areas:\n{titles}\n\nPropose 3 realistic student-founder or research-project opportunities "
              "that combine their healthcare-ML, security/IAM and agentic-AI skills. Return JSON: {\"ideas\": [{\"name\": str, "
              "\"problem\": str, \"solution\": str, \"why_you\": str, \"first_step_this_week\": str, \"biggest_risk\": str}]}")
    return llm.chat_json(prompt, temperature=0.8).get("ideas", [])
