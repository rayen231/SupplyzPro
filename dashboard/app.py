"""Developer dashboard. Imports the services directly; FastAPI exposes the same services over HTTP.

Run: streamlit run dashboard/app.py
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.ingestion.loader import load_ground_truth, load_sessions  # noqa: E402
from app.models.finding import FAILURE_CODES, FAILURE_LABELS  # noqa: E402
from app.services import store  # noqa: E402
from app.services.pipeline import analyze_session  # noqa: E402

st.set_page_config(page_title="Agent Failure Intelligence", layout="wide")

OUTCOME_COLOR = {"success": "#2e7d32", "recovered": "#1565c0", "clarification_needed": "#6a1b9a", "failure": "#c62828"}
TIER_COLOR = {"high": "#c62828", "medium": "#ef6c00", "low": "#757575"}


@st.cache_data
def _sessions():
    return {s.session_id: s for s in load_sessions()}


@st.cache_data
def _gt():
    return load_ground_truth()


def badge(text, color):
    return f"<span style='background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em'>{html.escape(str(text))}</span>"


configs = store.list_configs()
if not configs:
    st.error("No precomputed results yet. Run `python -m evaluation.evaluate` first.")
    st.stop()
keys = [c["config"] for c in configs]
default = next((k for k in (f"hybrid:{config.JUDGE_MODEL}", "rules") if k in keys), keys[0])

st.sidebar.title("Agent Failure Intelligence")
st.sidebar.caption("Semantic failure detection for SupplyzPro-style ERP agents")
page = st.sidebar.radio("Page", ["Overview & ranked issues", "Issue detail", "Trace explorer", "Evaluation, uncertainty & cost"],
                        key="page")
cfg = st.sidebar.selectbox("Analyzer configuration", keys, index=keys.index(default),
                           help="rules = deterministic only; llm = LLM judge on every session; hybrid = rules + LLM on triaged sessions")
analyses = {a.session_id: a for a in store.load_analyses(cfg)}
issues = store.load_issues(cfg)
SESS = _sessions()
GT = _gt()


def goto_trace(sid):
    st.session_state["trace_sid"] = sid
    st.session_state["page"] = "Trace explorer"


# ------------------------------------------------------------------ Overview
if page == "Overview & ranked issues":
    st.title("Agent reliability")
    st.caption("Did the agent achieve what the user asked, based on the evidence in the trace? No exception does not mean no failure.")
    outcomes = pd.Series([a.outcome for a in analyses.values()]).value_counts()
    c = st.columns(5)
    c[0].metric("Sessions analyzed", len(analyses))
    c[1].metric("Successful", int(outcomes.get("success", 0)))
    c[2].metric("Recovered", int(outcomes.get("recovered", 0)))
    c[3].metric("Clarification needed", int(outcomes.get("clarification_needed", 0)))
    c[4].metric("Potential failures", int(outcomes.get("failure", 0)))
    tiers = pd.Series([a.primary.tier for a in analyses.values() if a.primary]).value_counts()
    notes = sum(bool(a.uncertain_notes) for a in analyses.values())
    st.markdown(
        f"Confidence of failure findings: {badge('high ' + str(tiers.get('high', 0)), TIER_COLOR['high'])} "
        f"{badge('medium ' + str(tiers.get('medium', 0)), TIER_COLOR['medium'])} "
        f"{badge('low ' + str(tiers.get('low', 0)), TIER_COLOR['low'])} &nbsp; "
        f"{notes} sessions with rule/LLM disagreements or rejected findings", unsafe_allow_html=True)

    st.subheader("Ranked issues: what to investigate first")
    st.caption("priority = affected sessions × impact weight (high 5, medium 3, low 1) × mean confidence. "
               "A prioritization heuristic, not an objective truth.")
    if not issues:
        st.info("No failures found in this configuration.")
    df = pd.DataFrame([{
        "Rank": i["rank"], "Code": i["code"], "Issue": i["title"], "Sessions": i["affected_sessions"],
        "Impact": i["impact"], "Mean confidence": i["mean_confidence"], "Priority": i["priority"],
        "Common pattern": i["pattern"]} for i in issues])
    if len(df):
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"Priority": st.column_config.ProgressColumn(
                         "Priority", min_value=0, max_value=float(df["Priority"].max()), format="%.1f")})

# ------------------------------------------------------------------ Issue detail
elif page == "Issue detail":
    st.title("Issue detail")
    if not issues:
        st.info("No issues in this configuration.")
        st.stop()
    labels = [f"#{i['rank']} {i['code']} {i['title']} ({i['affected_sessions']} sessions)" for i in issues]
    idx = st.selectbox("Issue", range(len(issues)), format_func=lambda k: labels[k])
    iss = issues[idx]
    c = st.columns(4)
    c[0].metric("Affected sessions", iss["affected_sessions"])
    c[1].metric("Impact", iss["impact"])
    c[2].metric("Mean confidence", f"{iss['mean_confidence']:.0%}")
    c[3].metric("Priority", iss["priority"])
    st.markdown(f"**Common pattern.** {iss['pattern']}")
    st.markdown(f"**Tools involved:** {', '.join(iss['tools']) or 'none'} &nbsp; · &nbsp; "
                f"**Frequent terms (TF-IDF):** {', '.join(iss['top_terms'])} &nbsp; · &nbsp; "
                f"**Sub-clusters:** {iss['n_subclusters']}")
    st.markdown("Confidence spread: " + " ".join(badge(f"{t} {n}", TIER_COLOR[t]) for t, n in iss["tiers"].items()),
                unsafe_allow_html=True)
    st.subheader("Sessions in this issue")
    for sid in iss["sessions"]:
        a = analyses[sid]
        f = a.primary
        with st.container(border=True):
            cols = st.columns([5, 1])
            first_user = next((e.content for e in SESS[sid].events if e.role == "user"), "")
            cols[0].markdown(f"**{sid}** {badge(f.tier, TIER_COLOR[f.tier])} {badge('+'.join(f.sources), '#455a64')} "
                             f"sub-cluster {f.subcluster}<br><i>“{html.escape(first_user)}”</i><br>{html.escape(f.explanation)}"
                             f"<br><small>Evidence: {', '.join(f.evidence_event_ids)}</small>", unsafe_allow_html=True)
            cols[1].button("Open trace", key=f"open_{sid}", on_click=goto_trace, args=(sid,))

# ------------------------------------------------------------------ Trace explorer
elif page == "Trace explorer":
    st.title("Trace explorer")
    ids = sorted(SESS)
    only_fail = st.checkbox("Only sessions with findings", value=False)
    if only_fail:
        ids = [i for i in ids if analyses.get(i) and analyses[i].primary]
    cur = st.session_state.get("trace_sid", ids[0] if ids else None)
    if cur not in ids and ids:
        cur = ids[0]

    def _fmt(i):
        a = analyses.get(i)
        return f"{i} · {a.outcome}" + (f" · {a.primary.failure_type}" if a and a.primary else "") if a else i

    sid = st.selectbox("Session", ids, index=ids.index(cur) if cur in ids else 0, format_func=_fmt)
    st.session_state["trace_sid"] = sid
    s, a = SESS[sid], analyses.get(sid)

    left, right = st.columns([3, 2])
    evidence = set(a.primary.evidence_event_ids) if a and a.primary else set()
    with left:
        st.subheader("Execution timeline")
        for e in s.events:
            hl = e.event_id in evidence
            border = "#c62828" if hl else "#cfd8dc"
            bg = "rgba(198,40,40,0.07)" if hl else "transparent"
            if e.event_type == "message":
                who = "👤 User" if e.role == "user" else "🤖 Agent"
                body = html.escape(e.content or "")
            elif e.event_type == "tool_call":
                who = f"🔧 Call {e.tool_name}"
                body = f"<code>{html.escape(json.dumps(e.tool_arguments, ensure_ascii=False))}</code>"
            else:
                who = f"📦 Result {e.tool_name}"
                body = f"<code style='white-space:pre-wrap'>{html.escape(json.dumps(e.tool_result, ensure_ascii=False))}</code>"
            st.markdown(
                f"<div style='border-left:4px solid {border};background:{bg};padding:6px 10px;margin:4px 0'>"
                f"<small><b>{e.event_id}</b> · {who}{' · <b style=color:#c62828>EVIDENCE</b>' if hl else ''}</small><br>{body}</div>",
                unsafe_allow_html=True)
    with right:
        st.subheader("Analysis")
        if a:
            st.markdown(f"Outcome {badge(a.outcome, OUTCOME_COLOR[a.outcome])}", unsafe_allow_html=True)
            if a.primary:
                f = a.primary
                st.markdown(f"### 🚨 {FAILURE_CODES[f.failure_type]} {FAILURE_LABELS[f.failure_type]}")
                st.markdown(f"{badge('confidence ' + f.tier, TIER_COLOR[f.tier])} {badge('impact ' + f.impact, '#455a64')} "
                            f"{badge('sources: ' + '+'.join(f.sources), '#455a64')}", unsafe_allow_html=True)
                st.write(f.explanation)
                st.caption(f"Evidence events: {', '.join(f.evidence_event_ids)} · group {f.group_id}")
                if f.note:
                    st.warning(f.note)
                if f.invalid_evidence_ids:
                    st.error(f"Rejected evidence IDs (not in trace): {f.invalid_evidence_ids}")
            for n in a.uncertain_notes:
                st.info(n)
            with st.expander("Rule hits"):
                st.json([h.model_dump() for h in a.rule_hits] or "none")
            with st.expander("LLM judge verdict"):
                st.json(a.judge or "not called")
            with st.expander("Triage, runtime and tokens"):
                st.write({"triage_reasons": a.triage_reasons, "rules_ms": round(a.rules_ms, 2), "llm_ms": round(a.llm_ms),
                          "tokens_in": a.tokens_in, "tokens_out": a.tokens_out, "model": a.model})
        g = GT.get(sid)
        if g:
            with st.expander("Ground truth (evaluation only, never shown to the analyzer)"):
                st.write({k: g[k] for k in ("split", "family", "expected_outcome", "expected_failure_type", "key_event_ids")})
        st.divider()
        live_mode = st.selectbox("Live re-analysis mode", ["hybrid", "llm", "rules"])
        if st.button("▶ Analyze this trace live", disabled=live_mode != "rules" and not config.llm_available()):
            with st.spinner("Analyzing..."):
                live = analyze_session(s, live_mode, use_cache=False)
            st.success(f"Live result: {live.outcome}" + (f" · {live.primary.failure_type} ({live.primary.tier})" if live.primary else "")
                       + f" · {live.latency_ms:.0f} ms · {live.tokens_in}+{live.tokens_out} tokens")
            if live.primary:
                st.write(live.primary.explanation)

# ------------------------------------------------------------------ Evaluation
else:
    st.title("Evaluation, uncertainty & cost")
    st.caption(f"Ground truth: {len(GT)} synthetic sessions. 'tuned' = rules were written against these; "
               "'heldout' = paraphrased and French sessions the rules never saw.")
    split = st.radio("Split", ["all", "tuned", "heldout"], horizontal=True)
    cols = ["failure_precision", "failure_recall", "failure_f1", "false_positive_rate", "outcome_accuracy",
            "failure_type_macro_f1", "recovery_precision", "recovery_recall", "evidence_grounding_rate",
            "llm_call_share", "tokens_in_per_llm_call", "tokens_out_per_llm_call", "rules_ms_mean",
            "llm_ms_mean_per_call", "cost_usd_per_1k_sessions"]
    rows = []
    for c_ in configs:
        m = c_["metrics"].get(split) or {}
        rows.append({"config": c_["config"], **{k: m.get(k) for k in cols}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("""
**Metric notes.** Failure metrics treat *failure* as the positive class. The false positive rate counts successful, recovered
or clarification sessions that were flagged. Evidence grounding is the share of true-failure findings whose cited event IDs
include at least one gold key event. Cost uses Groq list prices from `app/config.py`. Verify them before quoting.
""")

    st.subheader(f"Uncertainty in {cfg}")
    by_tier = {"high": [], "medium": [], "low": []}
    for a in analyses.values():
        if a.primary:
            by_tier[a.primary.tier].append(a)
    c = st.columns(3)
    for col, t in zip(c, ("high", "medium", "low")):
        col.metric(f"{t.capitalize()} confidence", len(by_tier[t]))
    st.markdown("**High**: a rule fired and the LLM agrees. **Medium**: one source fired and its evidence validated "
                "(LLM-only findings are capped here). **Low**: sources disagree or evidence was partly invalid.")
    with st.expander(f"Inspect low-confidence findings ({len(by_tier['low'])})", expanded=bool(by_tier["low"])):
        for a in by_tier["low"]:
            st.markdown(f"**{a.session_id}** {a.primary.failure_type}: {a.primary.explanation}  \n*{a.primary.note}*")
            st.button("Open trace", key=f"low_{a.session_id}", on_click=goto_trace, args=(a.session_id,))
    overridden = [a for a in analyses.values() if a.uncertain_notes]
    with st.expander(f"Overridden rule suspicions, rejected LLM findings, fallbacks ({len(overridden)})"):
        for a in overridden:
            st.markdown(f"**{a.session_id}** → final outcome *{a.outcome}*")
            for n in a.uncertain_notes:
                st.caption(n)

    st.subheader("Limitations")
    st.markdown("""
- **Synthetic data, one author.** Traces and held-out variants were written by the same team that wrote the rules. The held-out split reduces, but does not remove, that bias. Real SupplyzPro traces are the next validation step.
- **Rules are English-centric.** Success-claim patterns are English. French claims are left to the LLM; entity matching is language-agnostic.
- **Small sample.** 87 sessions. Per-type metrics rest on 5 to 13 examples each and have wide error bars.
- **LLM-only types** (incomplete request, wrong interpretation, unsupported inference) have no deterministic backstop, so their confidence is capped at medium.
- **Trace-only view.** The analyzer never reads ERP state. It cannot detect a wrong answer that is consistent with every tool result it saw.
- **Grouping** uses signatures plus TF-IDF. Sentence embeddings with HDBSCAN are the scale-up path for discovering new categories.
""")
