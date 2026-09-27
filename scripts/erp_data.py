"""Synthetic SupplyzPro-like ERP state + mock tool functions.

Only the trace generator uses this module. The analyzer never reads ERP state.
"""

CUSTOMERS = [
    {"customer_id": "C001", "name": "Ahmed Ben Ali", "company": "ABC Distribution", "city": "Tunis", "phone": "+21622111111", "email": "ahmed@abc.tn"},
    {"customer_id": "C002", "name": "Ahmed Ben Ali", "company": "XYZ Trading", "city": "Sfax", "phone": "+21655222222", "email": "ahmed@xyz.tn"},
    {"customer_id": "C003", "name": "John Smith", "company": "Carthage Foods", "city": "Tunis", "phone": "+21620333333", "email": "john@carthagefoods.tn"},
    {"customer_id": "C004", "name": "John Doe", "company": "Medina Market", "city": "Sousse", "phone": "+21621444444", "email": "john@medinamarket.tn"},
    {"customer_id": "C005", "name": "Sami Trabelsi", "company": "Sahel Retail", "city": "Sousse", "phone": "+21698555555", "email": "sami@sahelretail.tn"},
    {"customer_id": "C006", "name": "Sami Trabelsi", "company": "Nour Supermarket", "city": "Tunis", "phone": "+21697666666", "email": "sami@nour.tn"},
    {"customer_id": "C007", "name": "Leila Haddad", "company": "Atlas Pharma", "city": "Tunis", "phone": "+21623777777", "email": "leila@atlaspharma.tn"},
    {"customer_id": "C008", "name": "Leila Haddad", "company": "Bizerte Bakery", "city": "Bizerte", "phone": "+21624888888", "email": "leila@bizertebakery.tn"},
    {"customer_id": "C009", "name": "Mehdi Jaziri", "company": "Jasmine Hotels", "city": "Hammamet", "phone": "+21625999999", "email": "mehdi@jasmine.tn"},
    {"customer_id": "C010", "name": "Ines Gharbi", "company": "Cap Bon Catering", "city": "Nabeul", "phone": "+21626101010", "email": "ines@capbon.tn"},
    {"customer_id": "C011", "name": "Karim Mansour", "company": "Djerba Resorts", "city": "Djerba", "phone": "+21627121212", "email": "karim@djerbaresorts.tn"},
    {"customer_id": "C012", "name": "Salma Ayari", "company": "Green Grocers", "city": "Sfax", "phone": "+21628131313", "email": "salma@greengrocers.tn"},
]

PRODUCTS = [
    {"product_id": "P001", "name": "Coca-Cola 1L", "sku": "CC1L", "category": "Beverages"},
    {"product_id": "P002", "name": "Coca-Cola Zero 1L", "sku": "CCZ1L", "category": "Beverages"},
    {"product_id": "P003", "name": "Safia Water 1.5L", "sku": "SAF15", "category": "Water"},
    {"product_id": "P004", "name": "Safia Water 0.5L", "sku": "SAF05", "category": "Water"},
    {"product_id": "P005", "name": "Delice Milk 1L", "sku": "DLM1L", "category": "Dairy"},
    {"product_id": "P006", "name": "Delice Milk Lactose-Free 1L", "sku": "DLMLF1L", "category": "Dairy"},
    {"product_id": "P007", "name": "Fanta Orange 1L", "sku": "FO1L", "category": "Beverages"},
    {"product_id": "P008", "name": "Sprite 1L", "sku": "SP1L", "category": "Beverages"},
    {"product_id": "P009", "name": "Vitalait Yogurt 125g", "sku": "VY125", "category": "Dairy"},
    {"product_id": "P010", "name": "Saida Wafers 45g", "sku": "SW45", "category": "Snacks"},
]

SUPPLIERS = [
    {"supplier_id": "SUP001", "name": "Carthage Beverages", "city": "Tunis"},
    {"supplier_id": "SUP002", "name": "Sahel Dairy", "city": "Sousse"},
    {"supplier_id": "SUP003", "name": "Atlas Water Supply", "city": "Zaghouan"},
    {"supplier_id": "SUP004", "name": "Medina Snacks", "city": "Sfax"},
]

WAREHOUSES = [
    {"warehouse_id": "W001", "name": "Tunis Warehouse", "city": "Tunis"},
    {"warehouse_id": "W002", "name": "Sousse Warehouse", "city": "Sousse"},
    {"warehouse_id": "W003", "name": "Sfax Warehouse", "city": "Sfax"},
]

_QTY = {
    "P001": (240, 85, 0), "P002": (130, 12, 60), "P003": (500, 320, 44), "P004": (900, 0, 210),
    "P005": (75, 150, 18), "P006": (20, 0, 35), "P007": (110, 64, 0), "P008": (95, 40, 12),
    "P009": (400, 260, 0), "P010": (1000, 350, 120),
}
INVENTORY = [
    {"product_id": p, "warehouse_id": w, "quantity": q}
    for p, qs in _QTY.items() for w, q in zip(("W001", "W002", "W003"), qs)
]

INVOICES = [
    {"invoice_id": "INV-1001", "customer_id": "C001", "date": "2026-09-02", "amount": 2450.0, "currency": "TND", "status": "paid"},
    {"invoice_id": "INV-1002", "customer_id": "C001", "date": "2026-09-18", "amount": 1320.5, "currency": "TND", "status": "pending"},
    {"invoice_id": "INV-1003", "customer_id": "C002", "date": "2026-09-10", "amount": 980.0, "currency": "TND", "status": "paid"},
    {"invoice_id": "INV-1004", "customer_id": "C002", "date": "2026-09-21", "amount": 4100.0, "currency": "TND", "status": "pending"},
    {"invoice_id": "INV-1005", "customer_id": "C003", "date": "2026-09-05", "amount": 610.0, "currency": "TND", "status": "paid"},
    {"invoice_id": "INV-1006", "customer_id": "C004", "date": "2026-09-12", "amount": 275.0, "currency": "TND", "status": "cancelled"},
    {"invoice_id": "INV-1007", "customer_id": "C004", "date": "2026-09-22", "amount": 1890.0, "currency": "TND", "status": "pending"},
    {"invoice_id": "INV-1008", "customer_id": "C005", "date": "2026-09-08", "amount": 3300.0, "currency": "TND", "status": "paid"},
    {"invoice_id": "INV-1009", "customer_id": "C006", "date": "2026-09-19", "amount": 745.0, "currency": "TND", "status": "pending"},
    {"invoice_id": "INV-1010", "customer_id": "C007", "date": "2026-09-03", "amount": 5200.0, "currency": "TND", "status": "paid"},
    {"invoice_id": "INV-1011", "customer_id": "C008", "date": "2026-09-15", "amount": 430.0, "currency": "TND", "status": "draft"},
    {"invoice_id": "INV-1012", "customer_id": "C009", "date": "2026-09-11", "amount": 2780.0, "currency": "TND", "status": "pending"},
    {"invoice_id": "INV-1013", "customer_id": "C010", "date": "2026-09-14", "amount": 1150.0, "currency": "TND", "status": "paid"},
    {"invoice_id": "INV-1014", "customer_id": "C011", "date": "2026-09-20", "amount": 6400.0, "currency": "TND", "status": "pending"},
    {"invoice_id": "INV-1015", "customer_id": "C012", "date": "2026-09-09", "amount": 890.0, "currency": "TND", "status": "failed"},
]

PURCHASE_ORDERS = [
    {"purchase_order_id": "PO-2001", "supplier_id": "SUP001", "status": "created", "lines": [{"product_id": "P001", "quantity": 200}]},
    {"purchase_order_id": "PO-2002", "supplier_id": "SUP002", "status": "pending_approval", "lines": [{"product_id": "P005", "quantity": 300}]},
    {"purchase_order_id": "PO-2003", "supplier_id": "SUP003", "status": "received", "lines": [{"product_id": "P003", "quantity": 1000}]},
]

BALANCES = {
    "C001": {"balance": 1320.5, "overdue_invoices": 0},
    "C002": {"balance": 4100.0, "overdue_invoices": 1},
    "C004": {"balance": 1890.0, "overdue_invoices": 1},
    "C009": {"balance": 2780.0, "overdue_invoices": 1},
    "C011": {"balance": 6400.0, "overdue_invoices": 2},
    "C012": {"balance": 890.0, "overdue_invoices": 1},
}


def _by(rows, key):
    return {r[key]: r for r in rows}


CUST = _by(CUSTOMERS, "customer_id")
PROD = _by(PRODUCTS, "product_id")
SUPP = _by(SUPPLIERS, "supplier_id")
WH = _by(WAREHOUSES, "warehouse_id")


def _match(query: str, text: str) -> bool:
    return all(tok in text.lower() for tok in query.lower().split())


# ---------------- mock ERP tools (what the agent sees) ----------------

def search_customer(name: str) -> dict:
    res = [{k: c[k] for k in ("customer_id", "name", "company", "city")}
           for c in CUSTOMERS if _match(name, c["name"])]
    return {"status": "ok", "count": len(res), "results": res}


def get_invoices(customer_id: str) -> dict:
    c = CUST[customer_id]
    inv = [{k: i[k] for k in ("invoice_id", "date", "amount", "currency", "status")}
           for i in INVOICES if i["customer_id"] == customer_id]
    return {"status": "ok", "customer_id": customer_id, "customer_name": c["name"],
            "company": c["company"], "invoices": inv}


def latest_invoice(customer_id: str) -> dict:
    return sorted([i for i in INVOICES if i["customer_id"] == customer_id], key=lambda i: i["date"])[-1]


def search_product(query: str) -> dict:
    res = [{k: p[k] for k in ("product_id", "name", "sku")} for p in PRODUCTS if _match(query, p["name"])]
    return {"status": "ok", "count": len(res), "results": res}


def list_warehouses() -> dict:
    return {"status": "ok", "results": [dict(w) for w in WAREHOUSES]}


def qty(product_id: str, warehouse_id: str) -> int:
    return next(i["quantity"] for i in INVENTORY if i["product_id"] == product_id and i["warehouse_id"] == warehouse_id)


def get_stock(product_id: str, warehouse_id: str) -> dict:
    q = qty(product_id, warehouse_id)
    return {"status": "ok", "product_id": product_id, "product_name": PROD[product_id]["name"],
            "warehouse_id": warehouse_id, "warehouse_name": WH[warehouse_id]["name"],
            "quantity": q, "stock_status": "in_stock" if q > 20 else ("low_stock" if q > 0 else "out_of_stock")}


def search_supplier(name: str) -> dict:
    res = [dict(s) for s in SUPPLIERS if _match(name, s["name"])]
    return {"status": "ok", "count": len(res), "results": res}


def create_purchase_order(supplier_id, lines, status="created", po_id="PO-2100", error=None) -> dict:
    out = {"status": status, "purchase_order_id": po_id if status != "failed" else None,
           "supplier_id": supplier_id, "supplier_name": SUPP[supplier_id]["name"], "lines": lines}
    if status == "pending_approval":
        out["message"] = "Order total exceeds auto-approval limit; manager approval required."
    if error:
        out["error"] = error
    return out


def create_invoice(customer_id, lines, status="created", inv_id="INV-1100", error=None) -> dict:
    total = round(sum(l.get("quantity", 1) * l.get("unit_price", 0) for l in lines), 2)
    out = {"status": status, "invoice_id": inv_id if status != "failed" else None, "customer_id": customer_id,
           "company": CUST[customer_id]["company"], "total": total, "currency": "TND"}
    if error:
        out["error"] = error
    return out


def send_invoice(invoice_id, email, status="sent", error=None) -> dict:
    out = {"status": status, "invoice_id": invoice_id, "email": email}
    if error:
        out["error"] = error
    return out


def get_customer_balance(customer_id: str) -> dict:
    b = BALANCES.get(customer_id, {"balance": 0.0, "overdue_invoices": 0})
    return {"status": "ok", "customer_id": customer_id, "company": CUST[customer_id]["company"],
            "balance": b["balance"], "currency": "TND", "overdue_invoices": b["overdue_invoices"]}


def record_expense(category, amount, status="recorded", exp_id="EXP-501") -> dict:
    return {"status": status, "expense_id": exp_id if status == "recorded" else None,
            "category": category, "amount": amount, "currency": "TND"}
