"""Group findings into issues.

Primary grouping: a signature (failure type + entity kind). Deterministic and explainable.
Semantic layer: TF-IDF cosine + agglomerative clustering inside each group to surface sub-patterns,
and the top TF-IDF terms of each group. Scale-up path: sentence embeddings + HDBSCAN.
"""
from collections import Counter, defaultdict

from app.models.finding import FAILURE_CODES, FAILURE_LABELS, Finding

PATTERNS = {
    ("wrong_entity", "customer"): "The user identifies the customer by name plus company or city. The agent searches by name only, gets several records, and continues with a record whose company does not match.",
    ("wrong_entity", "product"): "The user names a specific product variant. The search returns look-alike products and the agent continues with the wrong variant.",
    ("wrong_entity", "warehouse"): "The user names a location. The agent queries stock in a different warehouse and reports it as the requested one.",
    ("unsupported_success_claim", "purchase_order"): "The purchase-order tool returns pending_approval or failed, but the agent tells the user the order is created or confirmed.",
    ("unsupported_success_claim", "invoice"): "Invoice creation fails (for example on the credit limit) but the agent reports the invoice as created.",
    ("unsupported_success_claim", "invoice_email"): "Sending the invoice fails but the agent says it was sent.",
    ("unsupported_success_claim", "conversation"): "The agent claims a completed write without ever calling a state-changing tool.",
    ("repeated_question", "conversation"): "The agent asks the user for information already given in the request or in an earlier answer.",
    ("no_progress_loop", "customer"): "The agent repeats the same customer search with identical arguments and results instead of refining or asking.",
    ("no_progress_loop", "product"): "The agent repeats the same product search with no new information.",
    ("no_progress_loop", "supplier"): "The agent repeats the same supplier search with no new information.",
    ("incomplete_request", "purchase_order"): "A multi-line order is only partly created, but the agent presents it as done.",
    ("wrong_interpretation", "inventory"): "The stock tool reports zero or low stock, but the agent tells the user stock is available.",
    ("unsupported_inference", "customer_balance"): "The agent adds predictions about customer payments that no tool returned.",
}


def _subclusters(texts: list[str]) -> tuple[list[int], list[str]]:
    if len(texts) < 2:
        return [0] * len(texts), []
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[A-Za-z][A-Za-z_\-]+\b")
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return [0] * len(texts), []
    terms = vec.get_feature_names_out()
    weights = X.sum(axis=0).A1
    top = [terms[i] for i in weights.argsort()[::-1][:6]]
    if len(texts) < 3:
        return [0] * len(texts), top
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8, metric="cosine",
                                     linkage="average").fit_predict(X.toarray())
    return [int(x) for x in labels], top


def build_issues(findings: list[Finding]) -> list[dict]:
    from app.ranking.priority import priority_score
    groups: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        groups[f.signature].append(f)
    issues = []
    for n, (sig, fs) in enumerate(sorted(groups.items()), start=1):
        ftype, kind = sig.split("|")
        subs, top = _subclusters([f.explanation for f in fs])
        gid = f"{FAILURE_CODES[ftype]}-{kind}"
        for f, sc in zip(fs, subs):
            f.group_id, f.subcluster = gid, sc
        impact = Counter(f.impact for f in fs).most_common(1)[0][0]
        tiers = Counter(f.tier for f in fs)
        issue = {
            "group_id": gid, "failure_type": ftype, "code": FAILURE_CODES[ftype],
            "title": f"{FAILURE_LABELS[ftype]} ({kind.replace('_', ' ')})", "entity_kind": kind,
            "sessions": sorted({f.session_id for f in fs}), "n_findings": len(fs),
            "affected_sessions": len({f.session_id for f in fs}), "impact": impact,
            "tiers": {t: tiers.get(t, 0) for t in ("high", "medium", "low")},
            "tools": sorted({f.tool for f in fs if f.tool}), "top_terms": top,
            "n_subclusters": len(set(subs)),
            "pattern": PATTERNS.get((ftype, kind)) or f"Recurring {FAILURE_LABELS[ftype].lower()} involving {kind.replace('_', ' ')}.",
        }
        issue.update(priority_score(fs, impact))
        issues.append(issue)
    issues.sort(key=lambda i: -i["priority"])
    for rank, i in enumerate(issues, start=1):
        i["rank"] = rank
    return issues
