"""End-to-end analysis: rules -> triage -> LLM judge -> merge -> grouping -> ranking.

Modes
  rules   deterministic rules only, no LLM calls
  llm     LLM judge on every session, rules ignored
  hybrid  rules on every session; LLM only on sessions the triage flags; merge policy
"""
import time

from app import config
from app.detection.rules import recovery_signals, run_rules, triage
from app.grouping.clustering import build_issues
from app.models.finding import SessionAnalysis
from app.models.trace import Session
from app.reasoning.judge import judge
from app.services.merge import merge


def analyze_session(s: Session, mode: str = "hybrid", model: str | None = None, use_cache: bool = True) -> SessionAnalysis:
    model = model or config.JUDGE_MODEL
    t0 = time.perf_counter()
    hits = run_rules(s)
    recovery = recovery_signals(s)
    reasons = triage(s, hits, recovery)
    rules_ms = (time.perf_counter() - t0) * 1000

    call_llm = mode == "llm" or (mode == "hybrid" and bool(reasons))
    jr = judge(s, model, use_cache=use_cache) if call_llm else None
    verdict = jr.verdict if jr else None

    a = merge(s, mode, hits, recovery, verdict)
    a.triage_reasons = reasons
    a.rules_ms = rules_ms
    if jr:
        a.judge_called = jr.verdict is not None
        a.model = model
        a.tokens_in, a.tokens_out, a.llm_ms = jr.tokens_in, jr.tokens_out, jr.latency_ms
        if jr.error and not jr.verdict:
            a.uncertain_notes.append(f"LLM unavailable ({jr.error[:120]}); fell back to rules.")
    a.latency_ms = rules_ms + a.llm_ms
    return a


def analyze_all(sessions: list[Session], mode: str = "hybrid", model: str | None = None,
                use_cache: bool = True, progress=None) -> tuple[list[SessionAnalysis], list[dict]]:
    analyses = []
    for i, s in enumerate(sessions):
        analyses.append(analyze_session(s, mode, model, use_cache))
        if progress:
            progress(i + 1, len(sessions), s.session_id)
    findings = [f for a in analyses for f in a.findings]
    issues = build_issues(findings)
    return analyses, issues
