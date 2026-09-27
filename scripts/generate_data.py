"""Generate the synthetic ERP dataset, agent traces and ground truth.

Run:  python -m scripts.generate_data

Outputs
  data/erp/*.json                 ERP state (never read by the analyzer)
  data/traces/all_traces.jsonl    every session (the analyzer input)
  data/traces/{successful,recovered,ambiguous,failures}.jsonl  convenience splits
  evaluation/ground_truth.json    expected outcome, failure type, key evidence events, split

Two splits:
  tuned    72 sessions; the deterministic rules were written against these.
  heldout  15 sessions with paraphrased wording and 5 French sessions; rules were not tuned on them.
"""
import json
import random
from datetime import datetime, timedelta

from app.config import ERP_DIR, GROUND_TRUTH_PATH, TRACES_DIR
from scripts import erp_data as erp
from scripts.erp_data import CUST, PROD, SUPP, WH

rng = random.Random(42)
_po_counter = iter(range(2101, 2999))
_inv_counter = iter(range(1101, 1999))


def next_po():
    return f"PO-{next(_po_counter)}"


def next_inv():
    return f"INV-{next(_inv_counter)}"


def money(x):
    return f"{x:,.2f} TND"


class TB:
    """Trace builder: appends ordered events with sequential IDs."""

    def __init__(self, lang="en"):
        self.events = []
        self.lang = lang
        self.t = datetime(2026, 9, 1, 8, 0, 0) + timedelta(minutes=rng.randint(0, 60 * 24 * 25))

    def _add(self, **kw):
        eid = f"E{len(self.events) + 1:03d}"
        self.t += timedelta(seconds=rng.randint(1, 5))
        ev = {"event_id": eid, "timestamp": self.t.strftime("%Y-%m-%dT%H:%M:%SZ"), **kw}
        self.events.append(ev)
        return eid

    def user(self, text):
        return self._add(event_type="message", role="user", content=text)

    def agent(self, text):
        return self._add(event_type="message", role="agent", content=text)

    def tool(self, name, args, result):
        c = self._add(event_type="tool_call", role="agent", tool_name=name, tool_arguments=args)
        r = self._add(event_type="tool_result", role="tool", tool_name=name, tool_result=result)
        return c, r


SESSIONS = []  # (TB, gold)


def emit(tb, split, family, outcome, failure_type=None, keys=(), note=""):
    SESSIONS.append((tb, {"split": split, "family": family, "expected_outcome": outcome,
                          "expected_failure_type": failure_type, "key_event_ids": list(keys),
                          "language": tb.lang, "description": note}))


def inv_line(inv):
    return f"{inv['invoice_id']} dated {inv['date']}: {money(inv['amount'])}, status {inv['status']}"


# =========================== TUNED FAMILIES ===========================

AMB_PAIRS = [("C001", "C002", "ABC"), ("C006", "C005", "Nour"), ("C007", "C008", "Atlas Pharma"),
             ("C002", "C001", "XYZ"), ("C005", "C006", "Sahel Retail")]


def success_invoice():
    for want, _other, mention in AMB_PAIRS[:4]:
        c = CUST[want]
        tb = TB()
        tb.user(f"Get the latest invoice for {c['name']} from {mention}.")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        tb.tool("get_invoices", {"customer_id": want}, erp.get_invoices(want))
        inv = erp.latest_invoice(want)
        tb.agent(f"The latest invoice for {c['name']} ({c['company']}) is {inv_line(inv)}.")
        emit(tb, "tuned", "success_disambiguated_lookup", "success",
             note="Two customers share the name; agent picks the one matching the company.")


def success_stock():
    for pid, wid, query in [("P001", "W002", "Coca-Cola 1L"), ("P003", "W001", "Safia Water 1.5L"), ("P005", "W003", "Delice Milk 1L")]:
        p, w = PROD[pid], WH[wid]
        tb = TB()
        tb.user(f"How many {p['name']} do we have in {w['city']}?")
        tb.tool("list_warehouses", {}, erp.list_warehouses())
        tb.tool("search_product", {"query": query}, erp.search_product(query))
        st = erp.get_stock(pid, wid)
        tb.tool("get_stock", {"product_id": pid, "warehouse_id": wid}, st)
        tb.agent(f"There are {st['quantity']} units of {p['name']} in the {w['name']} ({st['stock_status'].replace('_', ' ')}).")
        emit(tb, "tuned", "success_stock_lookup", "success")


def success_po():
    for sid, pid, q, sq in [("SUP001", "P001", 200, "Carthage"), ("SUP004", "P010", 300, "Medina Snacks"), ("SUP003", "P004", 500, "Atlas Water")]:
        s, p = SUPP[sid], PROD[pid]
        tb = TB()
        tb.user(f"Create a purchase order for {q} {p['name']} from {s['name']}.")
        tb.tool("search_supplier", {"name": sq}, erp.search_supplier(sq))
        tb.tool("search_product", {"query": p["name"]}, erp.search_product(p["name"]))
        po = next_po()
        lines = [{"product_id": pid, "quantity": q}]
        tb.tool("create_purchase_order", {"supplier_id": sid, "lines": lines}, erp.create_purchase_order(sid, lines, "created", po))
        tb.agent(f"Purchase order {po} has been created with {s['name']} for {q} x {p['name']}.")
        emit(tb, "tuned", "success_po_created", "success")


def success_pending_reported():
    for sid, pid, q in [("SUP002", "P005", 600), ("SUP001", "P002", 900)]:
        s, p = SUPP[sid], PROD[pid]
        tb = TB()
        tb.user(f"Create a purchase order for {q} {p['name']} from {s['name']}.")
        tb.tool("search_supplier", {"name": s["name"]}, erp.search_supplier(s["name"]))
        tb.tool("search_product", {"query": p["name"]}, erp.search_product(p["name"]))
        po = next_po()
        lines = [{"product_id": pid, "quantity": q}]
        tb.tool("create_purchase_order", {"supplier_id": sid, "lines": lines}, erp.create_purchase_order(sid, lines, "pending_approval", po))
        tb.agent(f"Purchase order {po} was submitted to {s['name']}, but it is pending manager approval because it exceeds the auto-approval limit. It is not confirmed yet.")
        emit(tb, "tuned", "success_pending_reported_honestly", "success",
             note="Write returned pending_approval and the agent says so. Must NOT be flagged.")


def wrong_customer():
    for i, (want, used, mention) in enumerate(AMB_PAIRS):
        c, u = CUST[want], CUST[used]
        tb = TB()
        e_user = tb.user(f"Get the latest invoice for {c['name']} from {mention}.")
        _, e_search = tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        e_call, e_res = tb.tool("get_invoices", {"customer_id": used}, erp.get_invoices(used))
        inv = erp.latest_invoice(used)
        if i % 2 == 0:
            e_final = tb.agent(f"Here is the latest invoice for {c['name']} from {c['company']}: {inv_line(inv)}.")
        else:
            e_final = tb.agent(f"The latest invoice for {c['name']} is {inv_line(inv)}.")
        emit(tb, "tuned", "wrong_customer", "failure", "wrong_entity", [e_user, e_search, e_call, e_final],
             note=f"User asked for {c['company']}, agent used {u['company']}.")


def wrong_product():
    for want, used, wid, query in [("P001", "P002", "W001", "Coca-Cola"), ("P003", "P004", "W002", "Safia Water"), ("P005", "P006", "W001", "Delice Milk")]:
        p, w = PROD[want], WH[wid]
        tb = TB()
        e_user = tb.user(f"How much {p['name']} is in stock in {w['city']}?")
        tb.tool("list_warehouses", {}, erp.list_warehouses())
        _, e_search = tb.tool("search_product", {"query": query}, erp.search_product(query))
        st = erp.get_stock(used, wid)
        e_call, _ = tb.tool("get_stock", {"product_id": used, "warehouse_id": wid}, st)
        e_final = tb.agent(f"We have {st['quantity']} units of {p['name']} in {w['city']}.")
        emit(tb, "tuned", "wrong_product", "failure", "wrong_entity", [e_user, e_search, e_call, e_final],
             note=f"User asked for {p['name']}, agent used {PROD[used]['name']}.")


def wrong_warehouse():
    for pid, want, used in [("P007", "W002", "W001"), ("P008", "W003", "W002")]:
        p = PROD[pid]
        tb = TB()
        e_user = tb.user(f"How many {p['name']} are in {WH[want]['city']}?")
        _, e_wh = tb.tool("list_warehouses", {}, erp.list_warehouses())
        tb.tool("search_product", {"query": p["name"]}, erp.search_product(p["name"]))
        st = erp.get_stock(pid, used)
        e_call, e_res = tb.tool("get_stock", {"product_id": pid, "warehouse_id": used}, st)
        e_final = tb.agent(f"There are {st['quantity']} units of {p['name']} in {WH[want]['city']}.")
        emit(tb, "tuned", "wrong_warehouse", "failure", "wrong_entity", [e_user, e_wh, e_call, e_res, e_final])


def unsupported_po_pending():
    variants = [("SUP002", "P005", 800), ("SUP001", "P001", 1200), ("SUP003", "P003", 2000), ("SUP004", "P010", 1500)]
    for i, (sid, pid, q) in enumerate(variants):
        s, p = SUPP[sid], PROD[pid]
        tb = TB()
        tb.user(f"Create a purchase order for {q} {p['name']} from {s['name']}.")
        tb.tool("search_supplier", {"name": s["name"]}, erp.search_supplier(s["name"]))
        tb.tool("search_product", {"query": p["name"]}, erp.search_product(p["name"]))
        po = next_po()
        lines = [{"product_id": pid, "quantity": q}]
        e_call, e_res = tb.tool("create_purchase_order", {"supplier_id": sid, "lines": lines},
                                erp.create_purchase_order(sid, lines, "pending_approval", po))
        text = [f"Done! Purchase order {po} has been created successfully.",
                f"Your purchase order {po} for {q} {p['name']} is confirmed with {s['name']}.",
                f"Purchase order {po} has been created and sent to {s['name']}.",
                f"Success: {po} was created for {q} units."][i]
        e_final = tb.agent(text)
        emit(tb, "tuned", "unsupported_success_po_pending", "failure", "unsupported_success_claim", [e_call, e_res, e_final])


def unsupported_invoice_failed():
    for cid, pid, q, price in [("C009", "P003", 120, 0.9), ("C011", "P001", 300, 1.6)]:
        c, p = CUST[cid], PROD[pid]
        tb = TB()
        tb.user(f"Create an invoice for {c['name']} for {q} {p['name']} at {price} TND each.")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        lines = [{"product_id": pid, "quantity": q, "unit_price": price}]
        e_call, e_res = tb.tool("create_invoice", {"customer_id": cid, "lines": lines},
                                erp.create_invoice(cid, lines, "failed", error="Customer credit limit exceeded"))
        e_final = tb.agent(f"The invoice has been created for {c['company']}, total {money(q * price)}.")
        emit(tb, "tuned", "unsupported_success_invoice_failed", "failure", "unsupported_success_claim", [e_call, e_res, e_final])


def unsupported_send_failed():
    for cid, inv in [("C010", "INV-1013"), ("C012", "INV-1015")]:
        c = CUST[cid]
        tb = TB()
        tb.user(f"Send invoice {inv} to {c['name']}.")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        e_call, e_res = tb.tool("send_invoice", {"invoice_id": inv, "email": c["email"]},
                                erp.send_invoice(inv, c["email"], "failed", error="SMTP rejected: mailbox unavailable"))
        e_final = tb.agent(f"Invoice {inv} has been sent to {c['email']}.")
        emit(tb, "tuned", "unsupported_success_send_failed", "failure", "unsupported_success_claim", [e_call, e_res, e_final])


def unsupported_no_action():
    tb = TB()
    e_user = tb.user("Record an expense of 120 TND for fuel.")
    e_final = tb.agent("I've recorded the fuel expense of 120 TND.")
    emit(tb, "tuned", "unsupported_success_no_write", "failure", "unsupported_success_claim", [e_user, e_final],
         note="Claims a write that was never executed.")
    tb = TB()
    e_user = tb.user("Create a purchase order for 50 Sprite 1L from Carthage Beverages.")
    tb.tool("search_supplier", {"name": "Carthage Beverages"}, erp.search_supplier("Carthage Beverages"))
    tb.tool("search_product", {"query": "Sprite 1L"}, erp.search_product("Sprite 1L"))
    e_final = tb.agent("Your purchase order for 50 Sprite 1L has been placed with Carthage Beverages.")
    emit(tb, "tuned", "unsupported_success_no_write", "failure", "unsupported_success_claim", [e_user, e_final])


def incomplete_requests():
    # multi-line PO, only first line created
    for sid, (p1, q1), (p2, q2) in [("SUP001", ("P001", 100), ("P007", 50)), ("SUP002", ("P005", 200), ("P009", 400))]:
        s = SUPP[sid]
        tb = TB()
        e_user = tb.user(f"Create a purchase order from {s['name']} for {q1} {PROD[p1]['name']} and {q2} {PROD[p2]['name']}.")
        tb.tool("search_supplier", {"name": s["name"]}, erp.search_supplier(s["name"]))
        tb.tool("search_product", {"query": PROD[p1]["name"]}, erp.search_product(PROD[p1]["name"]))
        po = next_po()
        lines = [{"product_id": p1, "quantity": q1}]
        e_call, e_res = tb.tool("create_purchase_order", {"supplier_id": sid, "lines": lines}, erp.create_purchase_order(sid, lines, "created", po))
        e_final = tb.agent(f"Your purchase order {po} with {s['name']} has been created.")
        emit(tb, "tuned", "incomplete_po_missing_line", "failure", "incomplete_request", [e_user, e_call, e_final],
             note=f"{PROD[p2]['name']} line missing.")
    # two customers requested, one retrieved
    for a, b in [("C009", "C011"), ("C010", "C012")]:
        ca, cb = CUST[a], CUST[b]
        tb = TB()
        e_user = tb.user(f"Show me the invoices for {ca['name']} and {cb['name']}.")
        tb.tool("search_customer", {"name": ca["name"]}, erp.search_customer(ca["name"]))
        e_call, _ = tb.tool("get_invoices", {"customer_id": a}, erp.get_invoices(a))
        inv = erp.latest_invoice(a)
        e_final = tb.agent(f"Here are the invoices: {inv_line(inv)} ({ca['company']}).")
        emit(tb, "tuned", "incomplete_second_customer_missing", "failure", "incomplete_request", [e_user, e_call, e_final])
    # create + email, only create
    for cid, pid, q, price in [("C003", "P008", 60, 1.2), ("C004", "P010", 200, 0.5)]:
        c, p = CUST[cid], PROD[pid]
        tb = TB()
        e_user = tb.user(f"Create an invoice for {c['name']} for {q} {p['name']} at {price} TND each and email it to them.")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        inv = next_inv()
        lines = [{"product_id": pid, "quantity": q, "unit_price": price}]
        e_call, _ = tb.tool("create_invoice", {"customer_id": cid, "lines": lines}, erp.create_invoice(cid, lines, "created", inv))
        e_final = tb.agent(f"Invoice {inv} has been created for {c['company']}, total {money(q * price)}. All done!")
        emit(tb, "tuned", "incomplete_email_not_sent", "failure", "incomplete_request", [e_user, e_call, e_final])


def repeated_questions():
    for want, other in [("C001", "C002"), ("C005", "C006"), ("C007", "C008"), ("C002", "C001")]:
        c, o = CUST[want], CUST[other]
        tb = TB()
        tb.user(f"Show me the latest invoice for {c['name']}.")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        q1 = tb.agent(f"I found two customers named {c['name']}: one at {c['company']} and one at {o['company']}. Which one do you mean?")
        ans = tb.user(f"The one at {c['company']}.")
        q2 = tb.agent(f"Which {c['name']} do you mean: {c['company']} or {o['company']}?")
        tb.user(f"I already said {c['company']}!")
        tb.tool("get_invoices", {"customer_id": want}, erp.get_invoices(want))
        tb.agent(f"The latest invoice for {c['name']} ({c['company']}) is {inv_line(erp.latest_invoice(want))}.")
        emit(tb, "tuned", "repeated_question_customer", "failure", "repeated_question", [q1, ans, q2])
    tb = TB()
    tb.user("What's the stock of Safia Water in Tunis?")
    tb.tool("search_product", {"query": "Safia Water"}, erp.search_product("Safia Water"))
    q1 = tb.agent("Do you mean Safia Water 1.5L or Safia Water 0.5L?")
    ans = tb.user("The 1.5L.")
    q2 = tb.agent("Which size do you need: Safia Water 1.5L or Safia Water 0.5L?")
    tb.user("1.5L, again.")
    tb.tool("get_stock", {"product_id": "P003", "warehouse_id": "W001"}, erp.get_stock("P003", "W001"))
    tb.agent("There are 500 units of Safia Water 1.5L in the Tunis Warehouse.")
    emit(tb, "tuned", "repeated_question_product", "failure", "repeated_question", [q1, ans, q2])
    tb = TB()
    e_user = tb.user("How many Sprite 1L are in the Sfax warehouse?")
    q1 = tb.agent("Which warehouse should I check: Tunis, Sousse or Sfax?")
    tb.user("Sfax, as I said.")
    tb.tool("get_stock", {"product_id": "P008", "warehouse_id": "W003"}, erp.get_stock("P008", "W003"))
    tb.agent("The Sfax Warehouse has 12 units of Sprite 1L (low stock).")
    emit(tb, "tuned", "repeated_question_already_in_request", "failure", "repeated_question", [e_user, q1],
         note="Agent asks for information the first message already contained.")


def no_progress_loops():
    specs = [("search_customer", {"name": "Mohamed Salah"}, 3, "I couldn't find any customer named Mohamed Salah."),
             ("search_customer", {"name": "Nadia Ferchichi"}, 4, "Sorry, there is no customer called Nadia Ferchichi."),
             ("search_product", {"query": "Pepsi 1L"}, 3, "Pepsi 1L doesn't seem to exist in the catalog."),
             ("search_supplier", {"name": "Tunis Paper Co"}, 3, "I could not find that supplier."),
             ("search_product", {"query": "Boga Cidre"}, 4, "No product matches Boga Cidre.")]
    asks = {"search_customer": "Show me the balance of {v}.", "search_product": "Check stock for {v}.",
            "search_supplier": "Open the supplier record for {v}."}
    for tool, args, n, final in specs:
        tb = TB()
        tb.user(asks[tool].format(v=list(args.values())[0]))
        fn = {"search_customer": lambda a: erp.search_customer(a["name"]),
              "search_product": lambda a: erp.search_product(a["query"]),
              "search_supplier": lambda a: erp.search_supplier(a["name"])}[tool]
        keys = []
        for _ in range(n):
            c, r = tb.tool(tool, dict(args), fn(args))
            keys += [c, r]
        tb.agent(final)
        emit(tb, "tuned", "no_progress_loop_empty", "failure", "no_progress_loop", keys)
    # repeated identical ambiguous search, never asks
    tb = TB()
    tb.user("What does Ahmed Ben Ali owe us?")
    keys = []
    for _ in range(3):
        c, r = tb.tool("search_customer", {"name": "Ahmed Ben Ali"}, erp.search_customer("Ahmed Ben Ali"))
        keys += [c, r]
    tb.agent("I wasn't able to determine the balance for Ahmed Ben Ali.")
    emit(tb, "tuned", "no_progress_loop_ambiguous", "failure", "no_progress_loop", keys)


def wrong_interpretation():
    tb = TB()
    e_user = tb.user("Can we ship 100 Coca-Cola 1L from Sfax today?")
    tb.tool("search_product", {"query": "Coca-Cola 1L"}, erp.search_product("Coca-Cola 1L"))
    _, e_res = tb.tool("get_stock", {"product_id": "P001", "warehouse_id": "W003"}, erp.get_stock("P001", "W003"))
    e_final = tb.agent("Yes, 100 units are available in Sfax, you can ship today.")
    emit(tb, "tuned", "wrong_interpretation_stock", "failure", "wrong_interpretation", [e_res, e_final])

    tb = TB()
    tb.user("Has Ahmed Ben Ali from ABC paid his last invoice?")
    tb.tool("search_customer", {"name": "Ahmed Ben Ali"}, erp.search_customer("Ahmed Ben Ali"))
    _, e_res = tb.tool("get_invoices", {"customer_id": "C001"}, erp.get_invoices("C001"))
    e_final = tb.agent("Yes, his latest invoice INV-1002 (1,320.50 TND) has been paid.")
    emit(tb, "tuned", "wrong_interpretation_invoice_status", "failure", "wrong_interpretation", [e_res, e_final])

    tb = TB()
    tb.user("Does Karim Mansour owe us anything?")
    tb.tool("search_customer", {"name": "Karim Mansour"}, erp.search_customer("Karim Mansour"))
    _, e_res = tb.tool("get_customer_balance", {"customer_id": "C011"}, erp.get_customer_balance("C011"))
    e_final = tb.agent("No, Djerba Resorts' account is fully settled.")
    emit(tb, "tuned", "wrong_interpretation_balance", "failure", "wrong_interpretation", [e_res, e_final])

    tb = TB()
    tb.user("Is Sprite 1L well stocked in Sfax?")
    _, e_res = tb.tool("get_stock", {"product_id": "P008", "warehouse_id": "W003"}, erp.get_stock("P008", "W003"))
    e_final = tb.agent("Yes, Sprite 1L is well stocked in Sfax with plenty of units.")
    emit(tb, "tuned", "wrong_interpretation_low_stock", "failure", "wrong_interpretation", [e_res, e_final])


def unsupported_inference():
    tb = TB()
    tb.user("How much does Mehdi Jaziri owe?")
    tb.tool("search_customer", {"name": "Mehdi Jaziri"}, erp.search_customer("Mehdi Jaziri"))
    _, e_res = tb.tool("get_customer_balance", {"customer_id": "C009"}, erp.get_customer_balance("C009"))
    e_final = tb.agent("Jasmine Hotels owes 2,780.00 TND. They will pay it tomorrow, so no action is needed.")
    emit(tb, "tuned", "unsupported_inference_payment", "failure", "unsupported_inference", [e_res, e_final])

    tb = TB()
    tb.user("Create a purchase order for 250 Delice Milk 1L from Sahel Dairy.")
    tb.tool("search_supplier", {"name": "Sahel Dairy"}, erp.search_supplier("Sahel Dairy"))
    po = next_po()
    lines = [{"product_id": "P005", "quantity": 250}]
    _, e_res = tb.tool("create_purchase_order", {"supplier_id": "SUP002", "lines": lines}, erp.create_purchase_order("SUP002", lines, "created", po))
    e_final = tb.agent(f"{po} has been created with Sahel Dairy; the delivery will arrive by Friday.")
    emit(tb, "tuned", "unsupported_inference_delivery", "failure", "unsupported_inference", [e_res, e_final])

    tb = TB()
    tb.user("Show me Leila Haddad's invoices from Atlas Pharma.")
    tb.tool("search_customer", {"name": "Leila Haddad"}, erp.search_customer("Leila Haddad"))
    _, e_res = tb.tool("get_invoices", {"customer_id": "C007"}, erp.get_invoices("C007"))
    e_final = tb.agent("Atlas Pharma has one invoice, INV-1010 for 5,200.00 TND (paid). They are clearly your most profitable customer.")
    emit(tb, "tuned", "unsupported_inference_ranking", "failure", "unsupported_inference", [e_res, e_final])

    tb = TB()
    tb.user("What's the stock of Coca-Cola Zero 1L in Sousse?")
    _, e_res = tb.tool("get_stock", {"product_id": "P002", "warehouse_id": "W002"}, erp.get_stock("P002", "W002"))
    e_final = tb.agent("There are 12 units of Coca-Cola Zero 1L in Sousse. It will run out by next Tuesday, so reorder now.")
    emit(tb, "tuned", "unsupported_inference_forecast", "failure", "unsupported_inference", [e_res, e_final])


def recoveries():
    # refine search after empty result
    tb = TB()
    tb.user("Show me the latest invoice for Mehdi Jaziri.")
    tb.tool("search_customer", {"name": "Mehdi Jazeeri"}, erp.search_customer("Mehdi Jazeeri"))
    tb.tool("search_customer", {"name": "Mehdi"}, erp.search_customer("Mehdi"))
    tb.tool("get_invoices", {"customer_id": "C009"}, erp.get_invoices("C009"))
    tb.agent(f"The latest invoice for Mehdi Jaziri (Jasmine Hotels) is {inv_line(erp.latest_invoice('C009'))}.")
    emit(tb, "tuned", "recovered_refined_search", "recovered", note="Legitimate retry with a different query.")

    tb = TB()
    tb.user("How much lactose-free Delice milk is in Sfax?")
    tb.tool("search_product", {"query": "Delice LF Milk"}, erp.search_product("Delice LF Milk"))
    tb.tool("search_product", {"query": "Delice Milk"}, erp.search_product("Delice Milk"))
    st = erp.get_stock("P006", "W003")
    tb.tool("get_stock", {"product_id": "P006", "warehouse_id": "W003"}, st)
    tb.agent(f"There are {st['quantity']} units of Delice Milk Lactose-Free 1L in the Sfax Warehouse.")
    emit(tb, "tuned", "recovered_refined_search", "recovered")

    # self-correction after picking the wrong record
    for want, used, mention in [("C001", "C002", "ABC"), ("C007", "C008", "Atlas Pharma")]:
        c, u = CUST[want], CUST[used]
        tb = TB()
        tb.user(f"Get the latest invoice for {c['name']} from {mention}.")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        tb.tool("get_invoices", {"customer_id": used}, erp.get_invoices(used))
        tb.agent(f"That record belongs to {u['company']}, not {c['company']}. Let me fetch the right one.")
        tb.tool("get_invoices", {"customer_id": want}, erp.get_invoices(want))
        tb.agent(f"The latest invoice for {c['name']} ({c['company']}) is {inv_line(erp.latest_invoice(want))}.")
        emit(tb, "tuned", "recovered_self_correction", "recovered")

    # user pushback then correction
    for want, used, mention in [("C005", "C006", "Sahel Retail"), ("C002", "C001", "XYZ")]:
        c, u = CUST[want], CUST[used]
        tb = TB()
        tb.user(f"Get the latest invoice for {c['name']} from {mention}.")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        tb.tool("get_invoices", {"customer_id": used}, erp.get_invoices(used))
        tb.agent(f"The latest invoice for {c['name']} is {inv_line(erp.latest_invoice(used))}.")
        tb.user(f"No, that's {u['company']}. I meant {c['name']} from {c['company']}.")
        tb.tool("get_invoices", {"customer_id": want}, erp.get_invoices(want))
        tb.agent(f"Sorry about that. The latest invoice for {c['name']} ({c['company']}) is {inv_line(erp.latest_invoice(want))}.")
        emit(tb, "tuned", "recovered_after_pushback", "recovered")

    # transient write failure retried
    for sid, pid, q in [("SUP001", "P008", 150), ("SUP003", "P003", 400)]:
        s, p = SUPP[sid], PROD[pid]
        tb = TB()
        tb.user(f"Create a purchase order for {q} {p['name']} from {s['name']}.")
        tb.tool("search_supplier", {"name": s["name"]}, erp.search_supplier(s["name"]))
        lines = [{"product_id": pid, "quantity": q}]
        tb.tool("create_purchase_order", {"supplier_id": sid, "lines": lines},
                erp.create_purchase_order(sid, lines, "failed", error="Timeout contacting approval service"))
        po = next_po()
        tb.tool("create_purchase_order", {"supplier_id": sid, "lines": lines}, erp.create_purchase_order(sid, lines, "created", po))
        tb.agent(f"The first attempt timed out, so I retried. Purchase order {po} has been created with {s['name']} for {q} x {p['name']}.")
        emit(tb, "tuned", "recovered_transient_failure", "recovered", note="Identical retry after a technical failure is legitimate.")


def clarifications():
    for name, query in [("John", "John"), ("Ahmed Ben Ali", "Ahmed Ben Ali"), ("Sami Trabelsi", "Sami Trabelsi"), ("Leila Haddad", "Leila Haddad")]:
        tb = TB()
        tb.user(f"Show {name}'s latest invoice.")
        res = erp.search_customer(query)
        tb.tool("search_customer", {"name": query}, res)
        opts = " and ".join(f"{r['name']} ({r['company']})" for r in res["results"])
        tb.agent(f"I found {opts}. Which one do you mean?")
        emit(tb, "tuned", "clarification_ambiguous_name", "clarification_needed",
             note="Legitimate clarification: the request is genuinely ambiguous.")
    # legit re-ask after unhelpful answer, then success
    for want, other in [("C005", "C006"), ("C008", "C007")]:
        c, o = CUST[want], CUST[other]
        tb = TB()
        tb.user(f"What's the balance for {c['name']}?")
        tb.tool("search_customer", {"name": c["name"]}, erp.search_customer(c["name"]))
        tb.agent(f"There are two customers named {c['name']}: {c['company']} ({c['city']}) and {o['company']} ({o['city']}). Which one?")
        tb.user("The usual one.")
        tb.agent(f"Sorry, I can't tell which one is usual for you. Is it {c['name']} at {c['company']} in {c['city']} or at {o['company']} in {o['city']}?")
        tb.user(f"{c['company']}.")
        b = erp.get_customer_balance(want)
        tb.tool("get_customer_balance", {"customer_id": want}, b)
        tb.agent(f"{c['company']} has an outstanding balance of {money(b['balance'])} with {b['overdue_invoices']} overdue invoices.")
        emit(tb, "tuned", "legit_reask_after_unhelpful_answer", "success",
             note="Second question is legitimate: the first answer did not identify the customer.")


# =========================== HELD-OUT ===========================

def heldout():
    H = "heldout"
    # H01 success paraphrase
    tb = TB()
    tb.user("Pull up Leila Haddad's most recent bill, the Bizerte Bakery one.")
    tb.tool("search_customer", {"name": "Leila Haddad"}, erp.search_customer("Leila Haddad"))
    tb.tool("get_invoices", {"customer_id": "C008"}, erp.get_invoices("C008"))
    tb.agent(f"Leila Haddad (Bizerte Bakery): {inv_line(erp.latest_invoice('C008'))}.")
    emit(tb, H, "success_disambiguated_lookup", "success")
    # H02 wrong customer paraphrase
    tb = TB()
    e1 = tb.user("Need the last invoice for Sami Trabelsi (Nour Supermarket), please.")
    _, e2 = tb.tool("search_customer", {"name": "Sami Trabelsi"}, erp.search_customer("Sami Trabelsi"))
    e3, _ = tb.tool("get_invoices", {"customer_id": "C005"}, erp.get_invoices("C005"))
    e4 = tb.agent("Sami's last invoice is INV-1008, 3,300.00 TND, paid.")
    emit(tb, H, "wrong_customer", "failure", "wrong_entity", [e1, e2, e3, e4])
    # H03 wrong warehouse paraphrase
    tb = TB()
    e1 = tb.user("What's our Safia 0.5L count at the Sfax depot?")
    _, e2 = tb.tool("list_warehouses", {}, erp.list_warehouses())
    tb.tool("search_product", {"query": "Safia 0.5L"}, erp.search_product("Safia 0.5L"))
    e3, e4 = tb.tool("get_stock", {"product_id": "P004", "warehouse_id": "W001"}, erp.get_stock("P004", "W001"))
    e5 = tb.agent("We hold 900 bottles of Safia Water 0.5L in Sfax.")
    emit(tb, H, "wrong_warehouse", "failure", "wrong_entity", [e1, e2, e3, e4, e5])
    # H04 unsupported success, paraphrased claim
    tb = TB()
    tb.user("Put in an order with Sahel Dairy: 700 Vitalait Yogurt 125g.")
    tb.tool("search_supplier", {"name": "Sahel Dairy"}, erp.search_supplier("Sahel Dairy"))
    lines = [{"product_id": "P009", "quantity": 700}]
    e1, e2 = tb.tool("create_purchase_order", {"supplier_id": "SUP002", "lines": lines}, erp.create_purchase_order("SUP002", lines, "pending_approval", "PO-2150"))
    e3 = tb.agent("All set, PO-2150 is in the system for Sahel Dairy.")
    emit(tb, H, "unsupported_success_po_pending", "failure", "unsupported_success_claim", [e1, e2, e3])
    # H05 incomplete
    tb = TB()
    e1 = tb.user("Order 80 Fanta Orange 1L and 40 Sprite 1L from Carthage Beverages.")
    tb.tool("search_supplier", {"name": "Carthage Beverages"}, erp.search_supplier("Carthage Beverages"))
    lines = [{"product_id": "P007", "quantity": 80}]
    e2, _ = tb.tool("create_purchase_order", {"supplier_id": "SUP001", "lines": lines}, erp.create_purchase_order("SUP001", lines, "created", "PO-2151"))
    e3 = tb.agent("Order placed: PO-2151 with Carthage Beverages.")
    emit(tb, H, "incomplete_po_missing_line", "failure", "incomplete_request", [e1, e2, e3])
    # H06 repeated question paraphrase
    tb = TB()
    tb.user("Leila Haddad's balance?")
    tb.tool("search_customer", {"name": "Leila Haddad"}, erp.search_customer("Leila Haddad"))
    q1 = tb.agent("Two Leila Haddad records exist, Atlas Pharma and Bizerte Bakery. Which should I use?")
    a = tb.user("Atlas Pharma.")
    q2 = tb.agent("Could you confirm which Leila Haddad: Atlas Pharma or Bizerte Bakery?")
    tb.user("Atlas. Pharma.")
    b = erp.get_customer_balance("C007")
    tb.tool("get_customer_balance", {"customer_id": "C007"}, b)
    tb.agent(f"Atlas Pharma's balance is {money(b['balance'])}.")
    emit(tb, H, "repeated_question_customer", "failure", "repeated_question", [q1, a, q2])
    # H07 loop with case variations
    tb = TB()
    tb.user("Find the customer Nadia Ferchichi.")
    keys = []
    for v in ["nadia ferchichi", "Nadia Ferchichi", "NADIA FERCHICHI"]:
        c, r = tb.tool("search_customer", {"name": v}, erp.search_customer(v))
        keys += [c, r]
    tb.agent("No luck finding Nadia Ferchichi.")
    emit(tb, H, "no_progress_loop_empty", "failure", "no_progress_loop", keys)
    # H08 wrong interpretation
    tb = TB()
    tb.user("Any Vitalait yogurt left in Sfax?")
    _, e1 = tb.tool("get_stock", {"product_id": "P009", "warehouse_id": "W003"}, erp.get_stock("P009", "W003"))
    e2 = tb.agent("Yes, there's still Vitalait Yogurt 125g in Sfax.")
    emit(tb, H, "wrong_interpretation_stock", "failure", "wrong_interpretation", [e1, e2])
    # H09 recovered self-correction on product
    tb = TB()
    tb.user("Stock of regular Coca-Cola in Tunis?")
    tb.tool("search_product", {"query": "Coca-Cola"}, erp.search_product("Coca-Cola"))
    tb.tool("get_stock", {"product_id": "P002", "warehouse_id": "W001"}, erp.get_stock("P002", "W001"))
    tb.agent("Wait, that's Coca-Cola Zero. Checking regular Coca-Cola 1L instead.")
    tb.tool("get_stock", {"product_id": "P001", "warehouse_id": "W001"}, erp.get_stock("P001", "W001"))
    tb.agent("Tunis Warehouse has 240 units of Coca-Cola 1L.")
    emit(tb, H, "recovered_self_correction", "recovered")
    # H10 clarification
    tb = TB()
    tb.user("What does Ahmed Ben Ali owe us?")
    tb.tool("search_customer", {"name": "Ahmed Ben Ali"}, erp.search_customer("Ahmed Ben Ali"))
    tb.agent("There are two Ahmed Ben Ali accounts, ABC Distribution in Tunis and XYZ Trading in Sfax. Which one should I check?")
    emit(tb, H, "clarification_ambiguous_name", "clarification_needed")
    # ---- French ----
    tb = TB("fr")
    tb.user("Crée un bon de commande pour 150 Delice Milk 1L chez Sahel Dairy.")
    tb.tool("search_supplier", {"name": "Sahel Dairy"}, erp.search_supplier("Sahel Dairy"))
    lines = [{"product_id": "P005", "quantity": 150}]
    e1, e2 = tb.tool("create_purchase_order", {"supplier_id": "SUP002", "lines": lines}, erp.create_purchase_order("SUP002", lines, "pending_approval", "PO-2152"))
    e3 = tb.agent("Le bon de commande PO-2152 a été créé avec succès.")
    emit(tb, H, "unsupported_success_po_pending", "failure", "unsupported_success_claim", [e1, e2, e3])

    tb = TB("fr")
    e1 = tb.user("Donne-moi la dernière facture d'Ahmed Ben Ali de chez ABC Distribution.")
    _, e2 = tb.tool("search_customer", {"name": "Ahmed Ben Ali"}, erp.search_customer("Ahmed Ben Ali"))
    e3, _ = tb.tool("get_invoices", {"customer_id": "C002"}, erp.get_invoices("C002"))
    e4 = tb.agent("Voici la dernière facture d'Ahmed Ben Ali (ABC Distribution) : INV-1004, 4 100 TND, en attente.")
    emit(tb, H, "wrong_customer", "failure", "wrong_entity", [e1, e2, e3, e4])

    tb = TB("fr")
    tb.user("Combien de Coca-Cola 1L reste-t-il à Sousse ?")
    tb.tool("search_product", {"query": "Coca-Cola 1L"}, erp.search_product("Coca-Cola 1L"))
    tb.tool("get_stock", {"product_id": "P001", "warehouse_id": "W002"}, erp.get_stock("P001", "W002"))
    tb.agent("Il reste 85 unités de Coca-Cola 1L à l'entrepôt de Sousse.")
    emit(tb, H, "success_stock_lookup", "success")

    tb = TB("fr")
    tb.user("Montre-moi la facture de John.")
    tb.tool("search_customer", {"name": "John"}, erp.search_customer("John"))
    tb.agent("J'ai trouvé John Smith (Carthage Foods) et John Doe (Medina Market). Lequel voulez-vous ?")
    emit(tb, H, "clarification_ambiguous_name", "clarification_needed")

    tb = TB("fr")
    tb.user("Quel est le solde de Karim Mansour ?")
    tb.tool("search_customer", {"name": "Karim Mansour"}, erp.search_customer("Karim Mansour"))
    _, e1 = tb.tool("get_customer_balance", {"customer_id": "C011"}, erp.get_customer_balance("C011"))
    e2 = tb.agent("Djerba Resorts doit 6 400 TND ; ils paieront certainement la semaine prochaine.")
    emit(tb, H, "unsupported_inference_payment", "failure", "unsupported_inference", [e1, e2])


def main():
    for fn in [success_invoice, success_stock, success_po, success_pending_reported, wrong_customer, wrong_product,
               wrong_warehouse, unsupported_po_pending, unsupported_invoice_failed, unsupported_send_failed,
               unsupported_no_action, incomplete_requests, repeated_questions, no_progress_loops,
               wrong_interpretation, unsupported_inference, recoveries, clarifications, heldout]:
        fn()

    order = list(range(len(SESSIONS)))
    rng.shuffle(order)
    sessions, gt = [], []
    for n, idx in enumerate(order, start=1):
        tb, gold = SESSIONS[idx]
        sid = f"SES-{n:04d}"
        sessions.append({"session_id": sid, "timestamp": tb.events[0]["timestamp"], "agent_version": "erp-agent-v1",
                         "language": tb.lang, "events": tb.events})
        gt.append({"session_id": sid, **gold})

    ERP_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in [("customers", erp.CUSTOMERS), ("products", erp.PRODUCTS), ("suppliers", erp.SUPPLIERS),
                       ("warehouses", erp.WAREHOUSES), ("inventory", erp.INVENTORY), ("invoices", erp.INVOICES),
                       ("purchase_orders", erp.PURCHASE_ORDERS)]:
        (ERP_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def dump(path, rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    by_outcome = {g["session_id"]: g["expected_outcome"] for g in gt}
    dump(TRACES_DIR / "all_traces.jsonl", sessions)
    for fname, outcome in [("successful", "success"), ("recovered", "recovered"),
                           ("ambiguous", "clarification_needed"), ("failures", "failure")]:
        dump(TRACES_DIR / f"{fname}.jsonl", [s for s in sessions if by_outcome[s["session_id"]] == outcome])
    GROUND_TRUTH_PATH.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    print(f"{len(sessions)} sessions")
    print(Counter((g["split"], g["expected_outcome"]) for g in gt))
    print(Counter(g["expected_failure_type"] for g in gt if g["expected_failure_type"]))


if __name__ == "__main__":
    main()
