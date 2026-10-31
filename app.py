"""Streamlit UI for nextLevelPersonalCoachForCareerLevel.  Run:  streamlit run app.py"""
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

from agent import coding, evaluate, interview, jd, memory, opportunities, report, skills, voice
from agent.llm import LLM, LLMError

st.set_page_config(page_title="nextLevel Career Coach", page_icon="🎯", layout="wide")


@st.cache_resource
def get_llm():
    return LLM()


profile = skills.load_yaml("profile.yaml")
taxonomy = skills.load_yaml("skills.yaml")
opp_cfg = skills.load_yaml("opportunities.yaml")
memory.ensure_mastery(profile, taxonomy)
llm = get_llm()


@st.cache_resource
def get_graph(_profile_key: str):
    return interview.build_interview_graph(llm, profile, taxonomy)


graph = get_graph(json.dumps(profile, sort_keys=True))
ss = st.session_state
LAB = {k: v["label"] for k, v in taxonomy.items()}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("nextLevel Career Coach")
    st.caption(f"Personal AI career coach for {profile['name']}")
    st.markdown(f"**LLM chain:** `{llm.status()}`")
    if not llm.available():
        st.warning("No LLM key found. Copy `.env.example` to `.env` and add a free GEMINI_API_KEY, then restart.")
    st.markdown(f"**Coach style:** `{profile.get('coach_style')}`")
    st.divider()
    for r in skills.readiness(profile).values():
        st.metric(r["label"], f"{r['score']*100:.0f}/100", help=f"{r['band']} · confidence {r['confidence']}")
        st.caption(r["band"])

tabs = st.tabs(["Dashboard", "Mock Interview", "Coding Lab", "Opportunities", "JD Analyzer",
                "Readiness Report", "Evaluation", "Profile"])

# ================================================================ dashboard
with tabs[0]:
    st.subheader("Where you stand")
    m = memory.get_mastery()
    df = pd.DataFrame([{"skill": LAB[s], "mastery": round(v["value"] * 100), "graded answers": v["n"]} for s, v in m.items()])
    c1, c2 = st.columns([1, 1])
    c1.bar_chart(df.set_index("skill")["mastery"], horizontal=True)
    rh = evaluate.readiness_history(profile)
    if len(rh) > 1:
        c2.markdown("**Readiness over time**")
        c2.line_chart(rh)
    else:
        c2.info("Readiness history appears after your first session.")
    st.markdown("**Next best actions**")
    for key, t in profile["targets"].items():
        top = skills.priority_skills(profile, key)[:2]
        st.markdown(f"- **{t['label']}:** practice " + " and ".join(f"*{LAB[s]}*" for s, _ in top))
    last = memory.rows("SELECT ts, target, avg_score, summary_json FROM sessions ORDER BY ts DESC LIMIT 1")
    if last:
        s = json.loads(last[0]["summary_json"])
        st.markdown(f"**Last session** ({last[0]['ts'][:16]}): {last[0]['avg_score']}/10 · {s.get('overall_verdict', '')}")

# ================================================================ interview
with tabs[1]:
    if not ss.get("sid"):
        st.subheader("Start a mock interview")
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Target role", list(profile["targets"]), format_func=lambda k: profile["targets"][k]["label"])
        n = c2.number_input("Questions", 2, 12, profile.get("questions_per_session", 6))
        focus = c3.multiselect("Extra focus", list(taxonomy), format_func=LAB.get,
                               default=["dsa", "ml_theory", "ml_system_design", "security_fundamentals", "behavioral"])
        saved_jds = jd.saved()
        use_jd = st.selectbox("Tailor to a job description (optional)", [None] + saved_jds,
                              format_func=lambda j: "No, general practice" if j is None else f"{j['title']} · {j.get('company', '')}",
                              index=0 if not ss.get("jd_for_interview") else 1 + next(
                                  (i for i, j in enumerate(saved_jds) if j["id"] == ss["jd_for_interview"]), 0))
        ss["voice"] = st.toggle("🎙️ Voice mode: the coach speaks each question and explains concepts you miss; you answer out loud",
                                value=ss.get("voice", True))
        st.caption("Answer as you would out loud. Type or say 'skip' to hear the model answer. The coach probes weak answers once.")
        if st.button("Start interview", type="primary"):
            if not llm.available():
                st.error("No LLM key configured. Copy .env.example to .env and add a free GEMINI_API_KEY (see README).")
            else:
                with st.spinner("Planning your session and writing the first question…"):
                    sid, out = interview.start(graph, target, int(n), focus, use_jd)
                    ss.update(sid=sid, payload=interview.pending(out), history=[], done=None)
                st.rerun()
    else:
        p = ss.get("payload")
        for h in ss.get("history", []):
            with st.chat_message("assistant", avatar="🎯"):
                st.markdown(h["q"])
            with st.chat_message("user"):
                st.markdown(h["a"] or "_(no answer)_")
        if p and p.get("last"):
            r = p["last"]
            ev = r["eval"]
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                c1.metric("Score", f"{r['score']}/10" if r["score"] is not None else "–")
                if r.get("mastery_before") is not None:
                    c1.caption(f"{LAB[r['q']['skill']]}: {r['mastery_before']*100:.0f} → {r['mastery_after']*100:.0f}")
                c2.markdown(f"**{ev.get('verdict', '').title()}**: {ev.get('feedback', '')}")
                if ev.get("missing_points"):
                    c2.markdown("**Missing:** " + "; ".join(ev["missing_points"]))
                if ev.get("explain_concept"):
                    c2.markdown(f"**Concept explained:** {ev['explain_concept']}")
                with c2.expander("Model answer"):
                    st.write(ev.get("model_answer", ""))
                g1, g2, _ = c2.columns([1, 1, 6])
                if g1.button("👍 good question", key=f"up{r['answer_id']}"):
                    memory.rate_question(r["answer_id"], 1); st.toast("Thanks, logged.")
                if g2.button("👎 off-target", key=f"dn{r['answer_id']}"):
                    memory.rate_question(r["answer_id"], 0); st.toast("Logged. Used in evaluation.")
        turn = f"{ss['sid']}_{len(ss.get('history', []))}"
        played = ss.setdefault("played", set())

        def coach_audio(script: str, key: str):
            if not ss.get("voice") or not script:
                return
            audio = voice.speak(script)
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=key not in played)
                played.add(key)
            else:
                st.caption("🔇 Voice unavailable right now (no internet to the speech service). Text works as normal.")

        if p:
            q = p["question"]
            label = "Follow-up" if q.get("is_followup") else f"Question {p['index']} of {p['total']}"
            st.progress(min(1.0, (p["index"] - 1) / p["total"]), text=f"{label} · {LAB[q['skill']]} · {q.get('subtopic')} · {q['difficulty']}")
            with st.chat_message("assistant", avatar="🎯"):
                st.markdown(f"**{label}:** {q['question']}")
                coach_audio(voice.turn_script(p.get("last"), q, label), f"q_{turn}")
            draft_key = f"draft_{turn}"
            if ss.get("voice"):
                rec = st.audio_input("🎙️ Record your answer (click the mic, speak, click again to stop)", key=f"rec_{turn}")
                if rec is not None:
                    raw = rec.getvalue()
                    sig = voice.hashlib.sha1(raw).hexdigest()
                    if ss.get(f"sig_{turn}") != sig:
                        with st.spinner("Transcribing…"):
                            text, engine = voice.transcribe(raw, "audio/wav")
                        ss[f"sig_{turn}"] = sig
                        if text:
                            ss[draft_key] = (ss.get(draft_key, "") + " " + text).strip()
                            st.toast(f"Transcribed ({engine}). Edit below if needed, then submit.")
                        else:
                            st.warning("Couldn't transcribe that. Check your GROQ_API_KEY / GEMINI_API_KEY, or type your answer.")
            with st.form(key=f"ans_{turn}_{len(ss.get(draft_key, ''))}", clear_on_submit=True):
                a = st.text_area("Your answer" + (" (transcribed from your voice, edit freely)" if ss.get(draft_key) else ""),
                                 value=ss.get(draft_key, ""), height=180)
                sent = st.form_submit_button("Submit answer", type="primary")
            if sent:
                with st.spinner("Evaluating…"):
                    try:
                        out = interview.answer(graph, ss["sid"], a)
                        ss["history"].append({"q": q["question"], "a": a})
                        ss["payload"] = interview.pending(out)
                        ss.pop(draft_key, None)
                        if not ss["payload"]:
                            ss["done"] = out
                    except LLMError as e:
                        st.error(str(e))
                st.rerun()
        elif ss.get("done"):
            d = ss["done"]
            s = d.get("summary", {})
            st.success(f"Session complete · average {s.get('avg_score')}/10 · {s.get('overall_verdict', '')}")
            last = d.get("last")
            if last and last["eval"].get("explain_concept"):
                st.markdown(f"**Last answer: concept explained:** {last['eval']['explain_concept']}")
            coach_audio(voice.turn_script(last, None, "") +
                        f" That's the end of the session. Your average was {s.get('avg_score')} out of 10. "
                        f"{s.get('overall_verdict', '')}. Your next steps are on screen.", f"done_{ss['sid']}")
            c1, c2, c3 = st.columns(3)
            c1.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in s.get("strengths", [])))
            c2.markdown("**Gaps**\n" + "\n".join(f"- {x}" for x in s.get("gaps", [])))
            c3.markdown("**Next steps**\n" + "\n".join(f"- {x}" for x in s.get("next_steps", [])))
            st.download_button("Download transcript (.md)", d.get("transcript_md", ""), file_name=f"interview_{ss['sid']}.md")
            with st.expander("Full transcript"):
                st.markdown(d.get("transcript_md", ""))
        if st.button("End / new session"):
            for k in ("sid", "payload", "history", "done", "played"):
                ss.pop(k, None)
            st.rerun()

# ================================================================ coding lab
with tabs[2]:
    probs = coding.problems()
    c1, c2 = st.columns([3, 2])
    pid = c1.selectbox("Problem", [p["id"] for p in probs],
                       format_func=lambda i: next(f"{p['title']} · {p['difficulty']}" for p in probs if p["id"] == i))
    with c2.popover("Generate a new problem"):
        sub = st.selectbox("Subtopic", taxonomy["dsa"]["subtopics"])
        dif = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        fl = st.selectbox("Flavor", ["ML/AI engineering", "cybersecurity", "healthcare data"])
        if st.button("Generate + self-verify"):
            with st.spinner("Generating and verifying tests against a reference solution…"):
                try:
                    newp = coding.generate_problem(llm, sub, dif, fl)
                    st.success(f"Added: {newp['title']} ({len(newp['tests'])} verified tests)")
                except (LLMError, Exception) as e:  # noqa: BLE001
                    st.error(str(e))
    prob = next(p for p in probs if p["id"] == pid)
    st.markdown(f"### {prob['title']}\n{prob['prompt']}")
    st.code(prob["signature"], language="python")
    st.caption(f"{len(prob['tests'])} hidden-ish tests · first example: args={json.dumps(prob['tests'][0]['args'])[:150]} → {json.dumps(prob['tests'][0]['expected'])[:80]}")
    code = st.text_area("Your solution (Python)", value=ss.get(f"code_{pid}", prob["signature"] + "\n    pass\n"),
                        height=300, key=f"editor_{pid}")
    if st.button("Run tests + get review", type="primary"):
        ss[f"code_{pid}"] = code
        res = coding.run_tests(prob, code)
        if res.get("error"):
            st.error(res["error"])
        else:
            st.markdown(f"**Passed {res['passed']}/{res['total']}**")
            st.dataframe(pd.DataFrame([{"test": i + 1, "pass": t["ok"], "got": t.get("got", t.get("error")),
                                        "expected": json.dumps(prob["tests"][i]["expected"])[:80], "ms": t.get("ms")}
                                       for i, t in enumerate(res["tests"])]), hide_index=True)
        with st.spinner("Interviewer reviewing your code…"):
            rev = coding.review(llm, prob, code, res, profile.get("coach_style"))
        score = coding.grade_and_record(prob, code, res, rev)
        c1, c2 = st.columns([1, 3])
        c1.metric("Score", f"{score}/10")
        c1.caption(f"Time {rev.get('time_complexity', '?')} · Space {rev.get('space_complexity', '?')} · optimal: {rev.get('optimal_complexity', '?')}")
        c2.markdown(rev.get("feedback", ""))
        for k in ("bugs", "edge_cases_missed", "interviewer_followups"):
            if rev.get(k):
                c2.markdown(f"**{k.replace('_', ' ').capitalize()}:** " + "; ".join(rev[k]))

# ================================================================ opportunities
with tabs[3]:
    st.subheader("Internships & jobs")
    c1, c2, c3 = st.columns([1, 1, 2])
    deep = c2.checkbox("LLM fit analysis (top 8)", value=True)
    if c1.button("Scan opportunities", type="primary"):
        with st.spinner("Scanning job boards…"):
            jobs, health = opportunities.scan(profile, opp_cfg)
            if deep and jobs:
                jobs = opportunities.llm_fit(llm, profile, jobs)
            opportunities.save(jobs)
            ss["opp_health"] = health
        st.success(f"Found {len(jobs)} matching opportunities.")
    if ss.get("opp_health"):
        with c3.expander("Source health"):
            st.table(pd.DataFrame({"source": list(ss["opp_health"]), "status": list(ss["opp_health"].values())}))
    f1, f2 = st.columns(2)
    track = f1.selectbox("Track", ["all"] + list(profile["targets"]),
                         format_func=lambda k: "All" if k == "all" else profile["targets"][k]["label"])
    status = f2.selectbox("Status", ["active", "new", "saved", "applied", "dismissed"])
    q = "SELECT * FROM opportunities WHERE 1=1"
    args = []
    if track != "all":
        q += " AND track=?"; args.append(track)
    q += " AND status!='dismissed'" if status == "active" else " AND status=?"
    if status != "active":
        args.append(status)
    opps = memory.rows(q + " ORDER BY fit DESC LIMIT 60", tuple(args))
    for o in opps:
        fj = json.loads(o["fit_json"] or "{}")
        with st.expander(f"{o['fit']:.0f}/100 · {o['title']} · {o['company']} · {o['location'] or 'n/a'} · {o['status']}"):
            st.markdown(f"[Open posting]({o['url']}) · source: {o['source']} · posted {str(o['posted'])[:10]}")
            if fj.get("why"):
                st.markdown(f"**Why:** {fj['why']}")
                st.markdown(f"**Matching:** {', '.join(fj.get('matching', []))}  \n**Missing:** {', '.join(fj.get('missing', []))}")
                st.markdown(f"**Before applying:** {fj.get('next_action', '')}")
            b = st.columns(4)
            for i, s in enumerate(["saved", "applied", "dismissed"]):
                if b[i].button(s.capitalize(), key=f"{s}{o['id']}"):
                    opportunities.set_status(o["id"], s); st.rerun()
            if o["description"] and b[3].button("Analyze JD", key=f"jd{o['id']}"):
                with st.spinner("Analyzing…"):
                    a = jd.analyze(llm, profile, taxonomy, o["description"], o["title"], o["company"])
                ss["jd_for_interview"] = a["id"]
                st.success("Analysis saved. See the JD Analyzer tab, or start a tailored mock interview.")
    with st.expander("Big Tech & security career sites (no public API: paste postings into the JD Analyzer)"):
        st.markdown(" · ".join(f"[{k}]({v})" for k, v in opp_cfg["career_links"].items()))
    st.divider()
    st.subheader("Research & entrepreneurial opportunities")
    st.dataframe(pd.DataFrame(opp_cfg["programs"]), hide_index=True, width="stretch",
                 column_config={"url": st.column_config.LinkColumn("url")})
    c1, c2 = st.columns(2)
    if c1.button("Find recent research in my areas (arXiv)"):
        with st.spinner("Searching arXiv…"):
            ss["papers"] = opportunities.fetch_research(opp_cfg["research_queries"])
    if c2.button("Suggest startup / research project ideas"):
        with st.spinner("Thinking…"):
            try:
                ss["ideas"] = opportunities.startup_ideas(llm, profile, ss.get("papers", []))
            except LLMError as e:
                st.error(str(e))
    for p in ss.get("papers", []):
        st.markdown(f"- [{p['title']}]({p['url']}) · {p['authors']} · {p['published']}")
    for idea in ss.get("ideas", []):
        with st.container(border=True):
            st.markdown(f"**{idea.get('name')}**: {idea.get('problem')}\n\n*Solution:* {idea.get('solution')}\n\n"
                        f"*Why you:* {idea.get('why_you')}\n\n*First step this week:* {idea.get('first_step_this_week')} · "
                        f"*Risk:* {idea.get('biggest_risk')}")

# ================================================================ JD analyzer
with tabs[4]:
    st.subheader("Analyze a job description")
    c1, c2 = st.columns(2)
    t = c1.text_input("Role title")
    co = c2.text_input("Company")
    txt = st.text_area("Paste the full job description", height=220)
    if st.button("Analyze", type="primary") and txt.strip():
        with st.spinner("Extracting requirements and comparing with your profile…"):
            try:
                a = jd.analyze(llm, profile, taxonomy, txt, t, co)
                ss["jd_for_interview"] = a["id"]
            except LLMError as e:
                st.error(str(e))
    all_jds = jd.saved()
    if all_jds:
        sel = st.selectbox("Saved analyses", all_jds, format_func=lambda j: f"{j['title']} · {j.get('company', '')}")
        cov = sel.get("must_have_coverage")
        c = st.columns(3)
        c[0].metric("Must-have coverage", f"{cov*100:.0f}%" if cov is not None else "–")
        c[1].metric("Gaps", len(sel.get("gaps", [])))
        c[2].metric("Mapped skill areas", len(sel.get("mapped_skills", [])))
        st.markdown(sel.get("summary", ""))
        st.dataframe(pd.DataFrame(sel.get("required_skills", [])), hide_index=True, width="stretch")
        l, r = st.columns(2)
        l.markdown("**Likely interview topics**\n" + "\n".join(f"- {x}" for x in sel.get("likely_interview_topics", [])))
        r.markdown("**Resume keywords missing** (add only if true)\n" + "\n".join(f"- {x}" for x in sel.get("resume_keywords_missing", [])))
        if st.button("Practice for this role → Mock Interview tab"):
            ss["jd_for_interview"] = sel["id"]
            st.success("Selected. Open the Mock Interview tab and click Start.")

# ================================================================ report
with tabs[5]:
    st.subheader("Personalized Career Readiness Report")
    if st.button("Generate report", type="primary"):
        with st.spinner("Compiling assessments, gaps and a 14-day plan…"):
            md, _ = report.build(llm, profile, taxonomy)
            path = os.path.join(report.OUT, f"career_readiness_report_{datetime.now():%Y%m%d}.docx")
            report.to_docx(md, path)
            open(path.replace(".docx", ".md"), "w").write(md)
            ss["report"] = (md, path)
    if ss.get("report"):
        md, path = ss["report"]
        c1, c2 = st.columns(2)
        c1.download_button("Download .md", md, file_name=os.path.basename(path).replace(".docx", ".md"))
        c2.download_button("Download .docx (Times New Roman)", open(path, "rb").read(), file_name=os.path.basename(path))
        st.markdown(md)

# ================================================================ evaluation
with tabs[6]:
    m = evaluate.metrics()
    st.subheader("Is the coach accurate and are you improving?")
    c = st.columns(4)
    c[0].metric("Graded answers", m["answers"])
    c[1].metric("Avg score", m["avg_score"] or "–")
    c[2].metric("Score trend (2nd half − 1st half)", m["score_trend"] if m["score_trend"] is not None else "–")
    c[3].metric("Coding pass rate", m["coding_pass_rate"] if m["coding_pass_rate"] is not None else "–")
    c = st.columns(4)
    c[0].metric("Question relevance (👍 rate)", m["question_relevance"] if m["question_relevance"] is not None else "–",
                help=f"{m['questions_rated']} questions rated")
    c[1].metric("Follow-up gain", m["followup_gain"] if m["followup_gain"] is not None else "–",
                help="Avg score change from a weak answer to the follow-up answer")
    c[2].metric("Opportunity precision", m["opportunity_precision"] if m["opportunity_precision"] is not None else "–",
                help=f"saved+applied / judged ({m['opportunities_judged']} judged)")
    cal = m["calibration"]
    c[3].metric("Grader within human range", cal["within_range"] if cal else "run calibration")
    if cal:
        st.markdown(f"**Grader calibration** ({cal['n_items']} reference answers, {cal['provider']}): within range {cal['within_range']}, "
                    f"MAE outside range {cal['mae_outside_range']}, Spearman vs human {cal['spearman_vs_human']}, "
                    f"re-grade drift {cal['regrade_drift']} pts")
    if st.button("Run grader calibration (~22 LLM calls)"):
        from eval.run_calibration import main as calib_main
        with st.spinner("Grading 11 reference answers twice…"):
            res = calib_main(llm)
        st.json({k: v for k, v in res.items() if k != "rows"})
    l, r = st.columns(2)
    mh = evaluate.mastery_history()
    if not mh.empty:
        l.markdown("**Skill mastery over time**")
        l.line_chart(mh)
    stt = evaluate.session_trend()
    if not stt.empty:
        r.markdown("**Session average score**")
        r.line_chart(stt)

# ================================================================ profile
with tabs[7]:
    st.caption("Edit, then Save. Restart the app after changing coach style.")
    for fname in ("profile.yaml", "skills.yaml", "opportunities.yaml"):
        path = os.path.join(skills.CONFIG_DIR, fname)
        with st.expander(fname, expanded=fname == "profile.yaml"):
            txt = st.text_area(fname, open(path).read(), height=380, key=f"cfg_{fname}")
            if st.button(f"Save {fname}"):
                try:
                    yaml.safe_load(txt)
                    open(path, "w").write(txt)
                    st.success("Saved.")
                except yaml.YAMLError as e:
                    st.error(f"Invalid YAML: {e}")
