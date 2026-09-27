"""Small text utilities shared by the rules."""
import re

TOKEN_RE = re.compile(r"[a-z0-9àâçéèêëîïôûùüÿœ]+(?:[.\-][a-z0-9àâçéèêëîïôûùüÿœ]+)*")

STOP = {
    "the", "a", "an", "one", "ones", "which", "do", "you", "mean", "is", "it", "or", "and", "of", "to", "for",
    "i", "at", "in", "on", "named", "found", "there", "are", "two", "me", "please", "should", "could", "would",
    "confirm", "use", "what", "need", "that", "this", "with", "from", "my", "your", "we", "our", "us", "be",
    "can", "did", "does", "have", "has", "was", "were", "by", "as", "so", "if", "not", "no", "yes", "let",
    "know", "want", "check", "get", "show", "tell", "about", "how", "many", "much", "any", "also", "just",
    "said", "already", "again", "sorry", "okay", "ok", "le", "la", "les", "de", "du", "des", "et", "un", "une",
}

# A sentence that asserts an action completed.
SUCCESS_RE = re.compile(
    r"\b(successfully|success|created|confirmed|recorded|placed|completed|done|sent|submitted successfully|"
    r"has been (?:created|sent|recorded|placed|approved|updated|completed|submitted)|is (?:now )?(?:complete|done|confirmed))\b",
    re.I,
)
# Words that turn a sentence into a non-claim (hedged, negative or reporting a problem).
NEG_RE = re.compile(
    r"\b(not|pending|failed|fail|unable|cannot|awaiting|waiting|error|draft|timed out|could ?n[o']t|can't|isn't|wasn't)\b|n't\b",
    re.I,
)
# User asked for a state-changing action.
ACTION_RE = re.compile(r"\b(create|record|send|add|place|register|update|make|order|put in|cancel|approve)\b", re.I)
# Predictive / speculative language in a final answer.
SPECULATIVE_RE = re.compile(
    r"\b(will|going to|tomorrow|next week|next (?:monday|tuesday|wednesday|thursday|friday)|by (?:monday|tuesday|wednesday|thursday|friday)|"
    r"certainly|clearly|definitely|probably|likely|most profitable|run out)\b",
    re.I,
)
NON_FINAL_STATUSES = {"pending", "pending_approval", "failed", "draft", "out_of_stock", "low_stock", "cancelled", "error"}

WRITE_PREFIXES = ("create_", "send_", "record_", "update_", "delete_", "approve_", "cancel_")
SUCCESS_STATUSES = {"created", "sent", "recorded", "updated", "approved", "completed", "success", "confirmed", "deleted"}

TOOL_DOMAIN = {
    "search_customer": "customer", "get_invoices": "invoice", "get_customer_balance": "customer_balance",
    "search_product": "product", "get_stock": "inventory", "list_warehouses": "warehouse",
    "search_supplier": "supplier", "create_purchase_order": "purchase_order", "create_invoice": "invoice",
    "send_invoice": "invoice_email", "record_expense": "expense",
}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOP and len(t) > 1}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]


def claim_sentences(text: str) -> list[str]:
    return [s for s in sentences(text) if SUCCESS_RE.search(s) and not NEG_RE.search(s)]


def is_write(tool_name: str | None) -> bool:
    return bool(tool_name) and tool_name.startswith(WRITE_PREFIXES)


def domain(tool_name: str | None) -> str | None:
    if not tool_name:
        return None
    return TOOL_DOMAIN.get(tool_name, tool_name.split("_", 1)[-1])
