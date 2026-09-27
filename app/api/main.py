"""HTTP API over the same services the dashboard imports.

Run: uvicorn app.api.main:app --port 8000
"""
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config
from app.ingestion.loader import load_sessions
from app.models.finding import SessionAnalysis
from app.models.trace import Session
from app.services import store
from app.services.pipeline import analyze_session

app = FastAPI(title="Agent Failure Intelligence API", version="0.1.0",
              description="Detects semantic failures in ERP agent sessions and ranks recurring issues.")


class AnalyzeRequest(BaseModel):
    session: Session
    mode: Literal["rules", "llm", "hybrid"] = "hybrid"
    model: Optional[str] = None


def _default_config() -> str:
    keys = [c["config"] for c in store.list_configs()]
    for k in (f"hybrid:{config.JUDGE_MODEL}", "rules"):
        if k in keys:
            return k
    if not keys:
        raise HTTPException(404, "No precomputed results. Run: python -m evaluation.evaluate")
    return keys[0]


@app.get("/health")
def health():
    return {"status": "ok", "llm_available": config.llm_available(), "judge_model": config.JUDGE_MODEL}


@app.get("/configs")
def configs():
    return store.list_configs()


@app.get("/issues")
def issues(config_key: Optional[str] = None):
    return store.load_issues(config_key or _default_config())


@app.get("/sessions")
def sessions(config_key: Optional[str] = None, outcome: Optional[str] = None):
    rows = store.load_analyses(config_key or _default_config())
    return [{"session_id": a.session_id, "outcome": a.outcome,
             "failure_type": a.primary.failure_type if a.primary else None,
             "tier": a.primary.tier if a.primary else None, "group_id": a.primary.group_id if a.primary else None}
            for a in rows if outcome is None or a.outcome == outcome]


@app.get("/sessions/{session_id}")
def session_detail(session_id: str, config_key: Optional[str] = None):
    trace = next((s for s in load_sessions() if s.session_id == session_id), None)
    analysis = next((a for a in store.load_analyses(config_key or _default_config()) if a.session_id == session_id), None)
    if not trace:
        raise HTTPException(404, "unknown session")
    return {"trace": trace, "analysis": analysis}


@app.post("/analyze", response_model=SessionAnalysis)
def analyze(req: AnalyzeRequest):
    """Analyze one session posted by any agent runtime (live ingestion entry point)."""
    return analyze_session(req.session, req.mode, req.model)
