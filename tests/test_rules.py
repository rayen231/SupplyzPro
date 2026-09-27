"""Unit tests for the deterministic layer and merge policy."""
from app.detection.rules import rule_no_progress_loop, rule_repeated_question, rule_unsupported_success, rule_wrong_entity
from app.models.trace import Session
from app.reasoning.schemas import JudgeVerdict
from app.services.merge import merge


def S(events):
    evs = []
    for i, e in enumerate(events, start=1):
        evs.append({"event_id": f"E{i:03d}", "timestamp": "2026-09-27T10:00:00Z", **e})
    return Session(session_id="T", timestamp="2026-09-27T10:00:00Z", events=evs)


U = lambda t: {"event_type": "message", "role": "user", "content": t}  # noqa: E731
A = lambda t: {"event_type": "message", "role": "agent", "content": t}  # noqa: E731
C = lambda n, a: {"event_type": "tool_call", "role": "agent", "tool_name": n, "tool_arguments": a}  # noqa: E731
R = lambda n, r: {"event_type": "tool_result", "role": "tool", "tool_name": n, "tool_result": r}  # noqa: E731

AHMEDS = {"status": "ok", "count": 2, "results": [
    {"customer_id": "C001", "name": "Ahmed Ben Ali", "company": "ABC Distribution", "city": "Tunis"},
    {"customer_id": "C002", "name": "Ahmed Ben Ali", "company": "XYZ Trading", "city": "Sfax"}]}


def test_pending_claimed_as_created_is_hard_failure():
    s = S([U("Create a PO"), C("create_purchase_order", {"supplier_id": "SUP1"}), R("create_purchase_order", {"status": "pending_approval"}),
           A("Done! Purchase order PO-1 has been created successfully.")])
    hits = rule_unsupported_success(s)
    assert hits and hits[0].strength == "hard" and hits[0].evidence_event_ids == ["E002", "E003", "E004"]


def test_pending_reported_honestly_is_not_flagged():
    s = S([U("Create a PO"), C("create_purchase_order", {}), R("create_purchase_order", {"status": "pending_approval"}),
           A("PO-1 was submitted but is pending manager approval. It is not confirmed yet.")])
    assert rule_unsupported_success(s) == []


def test_wrong_customer_detected_and_recovery_recognized():
    base = [U("Get the latest invoice for Ahmed Ben Ali from ABC."), C("search_customer", {"name": "Ahmed Ben Ali"}),
            R("search_customer", AHMEDS), C("get_invoices", {"customer_id": "C002"}), R("get_invoices", {"status": "ok"})]
    hit = rule_wrong_entity(S(base + [A("Here it is.")]))[0]
    assert not hit.recovered and hit.entity_kind == "customer"
    fixed = base + [A("Wrong one, fixing."), C("get_invoices", {"customer_id": "C001"}), R("get_invoices", {"status": "ok"}), A("Here it is.")]
    assert rule_wrong_entity(S(fixed))[0].recovered


def test_legit_reask_not_flagged_but_real_reask_is():
    q = "There are two customers named Ahmed Ben Ali: ABC Distribution and XYZ Trading. Which one?"
    q2 = "Which Ahmed Ben Ali do you mean: ABC Distribution or XYZ Trading?"
    assert rule_repeated_question(S([A(q), U("The usual one."), A(q2)])) == []
    assert rule_repeated_question(S([A(q), U("ABC Distribution."), A(q2)]))


def test_loop_needs_three_identical_calls():
    empty = {"status": "ok", "count": 0, "results": []}
    two = [C("search_customer", {"name": "X"}), R("search_customer", empty)] * 2
    assert rule_no_progress_loop(S(two)) == []
    assert rule_no_progress_loop(S(two + [C("search_customer", {"name": "x "}), R("search_customer", empty)]))


def test_hard_rule_survives_llm_disagreement_as_low():
    s = S([U("Create a PO"), C("create_purchase_order", {}), R("create_purchase_order", {"status": "failed"}), A("PO created successfully.")])
    v = JudgeVerdict(user_goal="x", outcome="success", confidence=0.7, severity="low", explanation="ok", evidence_event_ids=["E004"])
    a = merge(s, "hybrid", rule_unsupported_success(s), [], v)
    assert a.outcome == "failure" and a.primary.tier == "low" and a.primary.disagreement


def test_llm_only_failure_capped_medium_and_fake_evidence_rejected():
    s = S([U("Stock?"), C("get_stock", {}), R("get_stock", {"quantity": 0}), A("Plenty in stock.")])
    v = JudgeVerdict(user_goal="x", outcome="failure", failure_type="wrong_interpretation", confidence=0.99,
                     severity="medium", explanation="qty 0", evidence_event_ids=["E003", "E004"])
    assert merge(s, "hybrid", [], [], v).primary.tier == "medium"
    v.evidence_event_ids = ["E999"]
    a = merge(s, "hybrid", [], [], v)
    assert a.outcome != "failure" and a.uncertain_notes
