"""Combine rule hits and the LLM verdict into one SessionAnalysis.

Policy
  * Hard rule hits cannot be overridden. LLM agreement -> high, no LLM -> medium, LLM disagrees -> low + disagreement.
  * Soft rule hits can be downgraded by the LLM (outcome follows the LLM, the override is kept as an uncertainty note).
  * LLM-only failures are capped at medium; partially invalid evidence -> low; no valid evidence -> rejected.
Confidence tiers
  high   = a rule fired and the LLM agrees on the failure type
  medium = one source fired and its evidence validated
  low    = sources disagree, or the evidence was only partly valid
"""
from app.detection.rules import ends_with_question
from app.detection.text import domain, is_write
from app.models.finding import Finding, RuleHit, SessionAnalysis
from app.models.trace import Session
from app.reasoning.schemas import JudgeVerdict

TIER_SCORE = {"high": 0.9, "medium": 0.6, "low": 0.3}


def impact_of(failure_type: str, tool: str | None, s: Session) -> str:
    if failure_type in ("repeated_question", "no_progress_loop"):
        return "low"
    if failure_type == "unsupported_success_claim" or is_write(tool):
        return "high"
    if failure_type == "incomplete_request" and any(is_write(c.tool_name) for c in s.tool_calls()):
        return "high"
    return "medium"


def _primary_tool(s: Session, ids: list[str]) -> str | None:
    evs = [s.event(i) for i in ids if s.event(i)]
    writes = [e.tool_name for e in evs if e.tool_name and is_write(e.tool_name)]
    if writes:
        return writes[0]
    calls = [e.tool_name for e in evs if e.event_type == "tool_call" and e.tool_name]
    if calls:
        return calls[-1]
    tools = [e.tool_name for e in evs if e.tool_name]
    return tools[-1] if tools else None


def validate_evidence(s: Session, ids: list[str]) -> tuple[list[str], list[str]]:
    known = s.event_ids
    valid = [i for i in dict.fromkeys(ids) if i in known]
    invalid = [i for i in dict.fromkeys(ids) if i not in known]
    return valid, invalid


def _finalize(s: Session, f: Finding) -> Finding:
    f.tool = f.tool or _primary_tool(s, f.evidence_event_ids)
    f.entity_kind = f.entity_kind or domain(f.tool) or "conversation"
    f.impact = impact_of(f.failure_type, f.tool, s)
    f.signature = f"{f.failure_type}|{f.entity_kind}"
    f.evidence_event_ids = sorted(dict.fromkeys(f.evidence_event_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    return f


def rules_outcome(s: Session, hits: list[RuleHit], recovery: list[str]) -> str:
    if [h for h in hits if not h.recovered]:
        return "failure"
    if any(h.recovered for h in hits) or recovery:
        return "recovered"
    if ends_with_question(s):
        return "clarification_needed"
    return "success"


def merge(s: Session, mode: str, hits: list[RuleHit], recovery: list[str], verdict: JudgeVerdict | None) -> SessionAnalysis:
    a = SessionAnalysis(session_id=s.session_id, mode=mode, outcome="success", rule_hits=hits,
                        judge=verdict.model_dump() if verdict else None)
    if mode == "llm":
        hits, recovery = [], []

    j = verdict
    j_valid, j_invalid = validate_evidence(s, j.evidence_event_ids) if j else ([], [])
    j_fail = bool(j and j.outcome == "failure")
    j_type = (j.failure_type or "incomplete_request") if j_fail else None

    active = [h for h in hits if not h.recovered]
    hard = [h for h in active if h.strength == "hard"]
    soft = [h for h in active if h.strength == "soft"]

    def llm_note():
        return f"LLM judged '{j.outcome}' ({j.confidence:.2f}): {j.explanation}" if j else ""

    if hard:
        h = hard[0]
        f = Finding(session_id=s.session_id, failure_type=h.failure_type, tier="medium", explanation=h.explanation,
                    evidence_event_ids=list(h.evidence_event_ids), sources=["rule"], entity_kind=h.entity_kind, tool=h.tool)
        if j_fail and j_type == h.failure_type:
            f.tier, f.sources = "high", ["rule", "llm"]
            f.evidence_event_ids += j_valid
        elif j_fail:
            f.note = f"LLM agrees it is a failure but classified it as {j_type}."
        elif j:
            f.tier, f.disagreement, f.note = "low", True, "Hard rule kept despite LLM: " + llm_note()
        a.outcome, a.findings = "failure", [_finalize(s, f)]
    elif soft:
        h = soft[0]
        f = Finding(session_id=s.session_id, failure_type=h.failure_type, tier="medium", explanation=h.explanation,
                    evidence_event_ids=list(h.evidence_event_ids), sources=["rule"], entity_kind=h.entity_kind, tool=h.tool)
        if j is None:
            a.outcome, a.findings = "failure", [_finalize(s, f)]
        elif j_fail and j_type == h.failure_type:
            f.tier, f.sources = "high", ["rule", "llm"]
            f.evidence_event_ids += j_valid
            a.outcome, a.findings = "failure", [_finalize(s, f)]
        elif j_fail:
            f.note = f"LLM agrees it is a failure but classified it as {j_type}: {j.explanation}"
            a.outcome, a.findings = "failure", [_finalize(s, f)]
        else:
            a.outcome = j.outcome
            a.uncertain_notes.append(f"Rule {h.rule_id} suspected {h.failure_type} ({h.explanation}) but was overridden. " + llm_note())
    else:
        base = rules_outcome(s, hits, recovery)
        if j_fail:
            if not j_valid:
                a.outcome = base
                a.uncertain_notes.append(f"Rejected LLM finding {j_type}: none of its evidence IDs {j.evidence_event_ids} exist.")
            else:
                f = Finding(session_id=s.session_id, failure_type=j_type, tier="medium", explanation=j.explanation,
                            evidence_event_ids=j_valid, invalid_evidence_ids=j_invalid, sources=["llm"])
                if j_invalid:
                    f.tier, f.note = "low", f"LLM cited non-existent events {j_invalid}."
                if base in ("recovered",) and mode != "llm":
                    f.tier, f.disagreement = "low", True
                    f.note += " Rules saw a recovery pattern."
                if j.failure_type is None:
                    f.tier, f.note = "low", f.note + " LLM gave no failure type."
                a.outcome, a.findings = "failure", [_finalize(s, f)]
        elif j:
            a.outcome = j.outcome
            # Deterministic recovery evidence (empty search refined, failed write retried) beats an LLM "success":
            # this only changes the success/recovered label, never a failure decision.
            if j.outcome == "success" and base == "recovered" and mode != "llm":
                a.outcome = "recovered"
            elif base != j.outcome and base != "success" and mode != "llm":
                a.uncertain_notes.append(f"Rules suggested '{base}', LLM judged '{j.outcome}'.")
        else:
            a.outcome = base
    return a
