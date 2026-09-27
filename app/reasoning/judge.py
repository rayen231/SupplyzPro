"""LLM judge: LangChain + Groq with structured output, disk cache and rate-limit backoff.

The judge sees only the normalized trace. It does not see rule hits, so one cached
verdict per (model, prompt version, session) serves both the llm-only and hybrid modes.
"""
import hashlib
import json
import re
import time
from dataclasses import dataclass

from app import config
from app.ingestion.normalizer import normalize
from app.models.trace import Session
from app.reasoning.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from app.reasoning.schemas import JudgeVerdict


@dataclass
class JudgeResult:
    verdict: JudgeVerdict | None
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None


_llms: dict[str, object] = {}


def _structured_llm(model: str, method: str):
    key = f"{model}|{method}"
    if key not in _llms:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, temperature=0, api_key=config.GROQ_API_KEY, max_retries=0, timeout=60)
        _llms[key] = llm.with_structured_output(JudgeVerdict, method=method, include_raw=True)
    return _llms[key]


def _cache_path(model: str, s: Session, trace_text: str):
    h = hashlib.sha256(f"{model}|{config.PROMPT_VERSION}|{SYSTEM_PROMPT}|{trace_text}".encode()).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    return config.CACHE_DIR / safe / f"{s.session_id}_{h}.json"


def judge(s: Session, model: str | None = None, use_cache: bool = True) -> JudgeResult:
    model = model or config.JUDGE_MODEL
    trace_text = normalize(s)
    path = _cache_path(model, s, trace_text)
    if use_cache and path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        v = JudgeVerdict.model_validate(d["verdict"]) if d.get("verdict") else None
        return JudgeResult(v, model, d["tokens_in"], d["tokens_out"], d["latency_ms"], cached=True, error=d.get("error"))
    if not config.llm_available():
        return JudgeResult(None, model, error="GROQ_API_KEY not set")

    messages = [("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE.format(session_id=s.session_id, trace=trace_text))]
    last_err = None
    for method in ("function_calling", "json_mode"):
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                msgs = messages if method == "function_calling" else messages + [
                    ("human", "Respond with a single JSON object with keys: " + ", ".join(JudgeVerdict.model_fields))]
                out = _structured_llm(model, method).invoke(msgs)
                latency = (time.perf_counter() - t0) * 1000
                raw, parsed = out.get("raw"), out.get("parsed")
                usage = getattr(raw, "usage_metadata", None) or {}
                if parsed is None:
                    last_err = f"parse error: {out.get('parsing_error')}"
                    break  # try next method
                res = JudgeResult(parsed, model, usage.get("input_tokens", 0), usage.get("output_tokens", 0), latency)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"verdict": parsed.model_dump(), "tokens_in": res.tokens_in,
                                            "tokens_out": res.tokens_out, "latency_ms": res.latency_ms}), encoding="utf-8")
                return res
            except Exception as e:  # rate limits and transient API errors
                msg = str(e)
                last_err = msg[:300]
                if "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower():
                    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
                    wait = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 10 * (attempt + 1)
                    if wait > 300:
                        return JudgeResult(None, model, error=f"rate limited for {wait:.0f}s: {last_err}")
                    time.sleep(wait + 0.5)
                    continue
                if "tool_use_failed" in msg or "Failed to call a function" in msg:
                    break  # model could not produce the tool call; try json_mode
                time.sleep(2)
    return JudgeResult(None, model, error=last_err)
