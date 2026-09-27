"""Deterministic detectors. They read only the trace (never ERP state).

Hard rules: a direct contradiction between tool evidence and the final answer. The LLM cannot override them.
Soft rules: a strong pattern that can still be legitimate. The LLM may downgrade them.
"""
import json
from collections import defaultdict

from app.detection.text import (
    ACTION_RE, NON_FINAL_STATUSES, SPECULATIVE_RE, SUCCESS_STATUSES, claim_sentences, content_tokens,
    domain, is_write, tokens,
)
from app.models.finding import RuleHit
from app.models.trace import Event, Session

DESC_FIELDS = ("name", "company", "city")


def _final_agent(s: Session) -> Event | None:
    msgs = s.agent_messages()
    return msgs[-1] if msgs else None


def _status(result_event: Event | None):
    if result_event and isinstance(result_event.tool_result, dict):
        return result_event.tool_result.get("status")
    return None


# ---------------------------------------------------------------- F001
def rule_unsupported_success(s: Session) -> list[RuleHit]:
    final = _final_agent(s)
    if not final:
        return []
    claims = claim_sentences(final.content)
    if not claims:
        return []
    writes = [c for c in s.tool_calls() if is_write(c.tool_name)]
    if not writes:
        asked = [u for u in s.user_messages() if ACTION_RE.search(u.content or "")]
        if asked:
            return [RuleHit(
                rule_id="R-F001-no-write", failure_type="unsupported_success_claim", strength="hard",
                explanation=f"The final answer claims a completed action (\"{claims[0]}\") but no state-changing tool was ever called.",
                evidence_event_ids=[asked[0].event_id, final.event_id], entity_kind=None, tool=None)]
        return []
    last = writes[-1]
    res = s.result_for(last)
    status = _status(res)
    if status in SUCCESS_STATUSES:
        return []
    detail = ""
    if res and isinstance(res.tool_result, dict):
        detail = res.tool_result.get("error") or res.tool_result.get("message") or ""
    return [RuleHit(
        rule_id="R-F001-status", failure_type="unsupported_success_claim", strength="hard",
        explanation=(f"The final answer claims success (\"{claims[0]}\") but {last.tool_name} returned status "
                     f"'{status}'" + (f" ({detail})" if detail else "") + "."),
        evidence_event_ids=[last.event_id] + ([res.event_id] if res else []) + [final.event_id],
        entity_kind=domain(last.tool_name), tool=last.tool_name)]


# ---------------------------------------------------------------- F002
def _candidate_scores(records: list[dict], user_toks: set[str]) -> list[tuple[int, int]]:
    toks = [set(tokens(" ".join(str(r.get(f, "")) for f in DESC_FIELDS))) for r in records]
    common = set.intersection(*toks) if toks else set()
    scores = []
    for t in toks:
        uniq = t - common
        scores.append((len(uniq & user_toks), -len(uniq - user_toks)))
    return scores


def rule_wrong_entity(s: Session) -> list[RuleHit]:
    candidates: dict[str, tuple[list[dict], str]] = {}
    user_toks: set[str] = set()
    user_events: list[Event] = []
    uses = defaultdict(list)
    for e in s.events:
        if e.event_type == "message" and e.role == "user":
            user_toks |= set(tokens(e.content))
            user_events.append(e)
        elif e.event_type == "tool_result" and isinstance(e.tool_result, dict):
            recs = e.tool_result.get("results")
            if isinstance(recs, list) and len(recs) >= 2 and all(isinstance(r, dict) for r in recs):
                for key in recs[0]:
                    if key.endswith("_id"):
                        candidates[key] = (recs, e.event_id)
        elif e.event_type == "tool_call" and e.tool_arguments:
            for k, v in e.tool_arguments.items():
                if k not in candidates or not isinstance(v, str):
                    continue
                recs, res_id = candidates[k]
                ids = [r.get(k) for r in recs]
                if v not in ids:
                    continue
                scores = _candidate_scores(recs, user_toks)
                best = max(scores)
                best_ids = [ids[i] for i, sc in enumerate(scores) if sc == best]
                chosen_score = scores[ids.index(v)]
                wrong = chosen_score < best and len(best_ids) == 1
                best_rec = recs[ids.index(best_ids[0])] if len(best_ids) == 1 else None
                uses[k].append({"call": e, "chosen": v, "best": best_ids[0] if best_rec else None,
                                "best_rec": best_rec, "chosen_rec": recs[ids.index(v)], "wrong": wrong,
                                "cand_event": res_id, "users": list(user_events)})
    hits = []
    final = _final_agent(s)
    for key, lst in uses.items():
        wrongs = [u for u in lst if u["wrong"]]
        if not wrongs:
            continue
        w = wrongs[0]
        last = lst[-1]
        recovered = (not last["wrong"]) and last["chosen"] == w["best"]
        res = s.result_for(w["call"])
        best_toks = set(tokens(" ".join(str(w["best_rec"].get(f, "")) for f in DESC_FIELDS)))
        user_ev = [u.event_id for u in w["users"] if set(tokens(u.content)) & best_toks] or [w["users"][0].event_id]
        ev = user_ev[:1] + [w["cand_event"], w["call"].event_id] + ([res.event_id] if res else [])
        if final and not recovered:
            ev.append(final.event_id)
        kind = key[:-3]
        desc = lambda r: " / ".join(str(r[f]) for f in DESC_FIELDS if f in r)  # noqa: E731
        hits.append(RuleHit(
            rule_id="R-F002-candidate-mismatch", failure_type="wrong_entity", strength="soft",
            explanation=(f"The user's words match {kind} {w['best']} ({desc(w['best_rec'])}), but the agent called "
                         f"{w['call'].tool_name} with {w['chosen']} ({desc(w['chosen_rec'])})."
                         + (" The agent later switched to the matching record." if recovered else "")),
            evidence_event_ids=ev, entity_kind=kind, tool=w["call"].tool_name, recovered=recovered))
    return hits


# ---------------------------------------------------------------- F004
def rule_repeated_question(s: Session) -> list[RuleHit]:
    qs = [(i, e) for i, e in enumerate(s.events) if e.event_type == "message" and e.role == "agent" and "?" in (e.content or "")]
    for a in range(len(qs)):
        for b in range(a + 1, len(qs)):
            (i, q1), (j, q2) = qs[a], qs[b]
            t1, t2 = content_tokens(q1.content), content_tokens(q2.content)
            if not t1 or not t2:
                continue
            overlap = len(t1 & t2) / min(len(t1), len(t2))
            if overlap < 0.6:
                continue
            replies = [e for e in s.events[i + 1:j] if e.event_type == "message" and e.role == "user"]
            answered = [r for r in replies if content_tokens(r.content) & t1]
            if answered:
                given = sorted(content_tokens(answered[0].content) & t1)
                return [RuleHit(
                    rule_id="R-F004-reask", failure_type="repeated_question", strength="soft",
                    explanation=(f"The agent asked the same question twice ({q1.event_id}, {q2.event_id}) although the user "
                                 f"had already answered in {answered[0].event_id} ({', '.join(given)})."),
                    evidence_event_ids=[q1.event_id, answered[0].event_id, q2.event_id], entity_kind=None, tool=None)]
    return []


# ---------------------------------------------------------------- F005
def _norm_args(args):
    if isinstance(args, dict):
        return {k: _norm_args(v) for k, v in sorted(args.items())}
    if isinstance(args, str):
        return " ".join(args.lower().split())
    return args


def rule_no_progress_loop(s: Session, min_repeats: int = 3) -> list[RuleHit]:
    groups = defaultdict(list)
    for c in s.tool_calls():
        res = s.result_for(c)
        key = (c.tool_name, json.dumps(_norm_args(c.tool_arguments or {}), sort_keys=True))
        groups[key].append((c, res, json.dumps(res.tool_result if res else None, sort_keys=True)))
    hits = []
    for (tool, args), lst in groups.items():
        by_result = defaultdict(list)
        for c, r, rj in lst:
            by_result[rj].append((c, r))
        same = max(by_result.values(), key=len)
        if len(same) >= min_repeats:
            ev = [x.event_id for pair in same for x in pair if x]
            hits.append(RuleHit(
                rule_id="R-F005-identical-calls", failure_type="no_progress_loop", strength="soft",
                explanation=f"{tool} was called {len(same)} times with the same arguments {args} and returned the same result each time.",
                evidence_event_ids=ev, entity_kind=domain(tool), tool=tool))
    return hits


# ---------------------------------------------------------------- recovery / clarification signals
def recovery_signals(s: Session) -> list[str]:
    out = []
    calls = s.tool_calls()
    for i, c in enumerate(calls):
        r = s.result_for(c)
        st = _status(r)
        if is_write(c.tool_name) and st == "failed":
            later = [x for x in calls[i + 1:] if x.tool_name == c.tool_name and _status(s.result_for(x)) in SUCCESS_STATUSES]
            if later:
                out.append(f"{c.tool_name} failed at {c.event_id} and succeeded on retry at {later[0].event_id}")
        if c.tool_name.startswith("search_") and r and isinstance(r.tool_result, dict) and r.tool_result.get("count") == 0:
            later = [x for x in calls[i + 1:] if x.tool_name == c.tool_name and x.tool_arguments != c.tool_arguments
                     and (s.result_for(x) and (s.result_for(x).tool_result or {}).get("count", 0) > 0)]
            if later:
                out.append(f"empty {c.tool_name} at {c.event_id} was refined at {later[0].event_id}")
    return out


def ends_with_question(s: Session) -> bool:
    return bool(s.events) and s.events[-1].event_type == "message" and s.events[-1].role == "agent" and "?" in (s.events[-1].content or "")


ALL_RULES = [rule_unsupported_success, rule_wrong_entity, rule_repeated_question, rule_no_progress_loop]


def run_rules(s: Session) -> list[RuleHit]:
    hits = []
    for rule in ALL_RULES:
        hits.extend(rule(s))
    # hard first, then soft
    return sorted(hits, key=lambda h: (h.strength != "hard", h.recovered))


def triage(s: Session, hits: list[RuleHit], recovery: list[str]) -> list[str]:
    """Why a trace deserves LLM review in hybrid mode. Empty list = clean enough to skip."""
    reasons = []
    if hits:
        reasons.append("rule hit")
    if recovery:
        reasons.append("recovery pattern")
    if any(is_write(c.tool_name) for c in s.tool_calls()):
        reasons.append("state-changing tool")
    if len(s.user_messages()) > 1:
        reasons.append("multi-turn")
    final = _final_agent(s)
    if final and (claim_sentences(final.content) or SPECULATIVE_RE.search(final.content or "")):
        reasons.append("claim or speculative language in final answer")
    for e in s.events:
        if e.event_type == "tool_result" and isinstance(e.tool_result, dict):
            vals = {str(v) for v in e.tool_result.values() if isinstance(v, str)}
            if vals & NON_FINAL_STATUSES or (e.tool_result.get("overdue_invoices") or 0) > 0:
                reasons.append("non-final status in tool result")
                break
            for inv in e.tool_result.get("invoices") or []:
                if inv.get("status") in NON_FINAL_STATUSES:
                    reasons.append("non-final status in tool result")
                    break
    if any(e.tool_name and e.tool_name.startswith("search_") and isinstance(e.tool_result, dict)
           and (e.tool_result.get("count") or 0) >= 2 for e in s.events if e.event_type == "tool_result"):
        reasons.append("ambiguous search result")
    first = s.user_messages()[0].content if s.user_messages() else ""
    if " and " in f" {first.lower()} " or " et " in f" {first.lower()} ":
        reasons.append("multi-part request")
    if not final:
        reasons.append("no final answer")
    return sorted(set(reasons))
