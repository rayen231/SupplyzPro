"""Quick diagnostic: rule-only outcome vs ground truth (excluding LLM-only failure types)."""
from app.detection.rules import ends_with_question, recovery_signals, run_rules, triage
from app.ingestion.loader import load_ground_truth, load_sessions

S = load_sessions()
G = load_ground_truth()
bad = skipped = 0
for s in S:
    g = G[s.session_id]
    hits = run_rules(s)
    rec = recovery_signals(s)
    fail = [h for h in hits if not h.recovered]
    if fail:
        out, ft = "failure", fail[0].failure_type
    elif any(h.recovered for h in hits) or rec:
        out, ft = "recovered", None
    elif ends_with_question(s):
        out, ft = "clarification_needed", None
    else:
        out, ft = "success", None
    if not triage(s, hits, rec):
        skipped += 1
    ok = out == g["expected_outcome"] and ft == g["expected_failure_type"]
    llm_only = g["expected_failure_type"] in ("incomplete_request", "wrong_interpretation", "unsupported_inference")
    if not ok and not llm_only:
        bad += 1
        print(g["split"], s.session_id, g["family"], "| expected", g["expected_outcome"], g["expected_failure_type"],
              "| got", out, ft, [h.rule_id for h in hits])
print("mismatches (excluding LLM-only types):", bad, "| triage-skipped:", skipped, "of", len(S))
