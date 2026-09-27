"""Findings produced by rules, the LLM judge, and the merge step."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Outcome = Literal["success", "recovered", "clarification_needed", "failure"]
FailureType = Literal[
    "unsupported_success_claim",   # F001
    "wrong_entity",                # F002
    "incomplete_request",          # F003
    "repeated_question",           # F004
    "no_progress_loop",            # F005
    "wrong_interpretation",        # F006
    "unsupported_inference",       # F007
]
FAILURE_CODES = {
    "unsupported_success_claim": "F001",
    "wrong_entity": "F002",
    "incomplete_request": "F003",
    "repeated_question": "F004",
    "no_progress_loop": "F005",
    "wrong_interpretation": "F006",
    "unsupported_inference": "F007",
}
FAILURE_LABELS = {
    "unsupported_success_claim": "Unsupported success claim",
    "wrong_entity": "Wrong entity selected",
    "incomplete_request": "Incomplete request presented as done",
    "repeated_question": "Repeated question",
    "no_progress_loop": "No-progress tool loop",
    "wrong_interpretation": "Wrong interpretation of tool result",
    "unsupported_inference": "Unsupported inference",
}
Tier = Literal["high", "medium", "low"]


class RuleHit(BaseModel):
    rule_id: str
    failure_type: FailureType
    strength: Literal["hard", "soft"]
    explanation: str
    evidence_event_ids: list[str]
    entity_kind: Optional[str] = None
    tool: Optional[str] = None
    recovered: bool = False  # the agent later corrected the problem


class Finding(BaseModel):
    session_id: str
    failure_type: FailureType
    tier: Tier
    explanation: str
    evidence_event_ids: list[str]
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    sources: list[str]                 # "rule", "llm"
    disagreement: bool = False
    note: str = ""
    entity_kind: Optional[str] = None
    tool: Optional[str] = None
    impact: Literal["high", "medium", "low"] = "medium"
    signature: str = ""
    group_id: str = ""
    subcluster: int = 0


class SessionAnalysis(BaseModel):
    session_id: str
    mode: str                          # rules | llm | hybrid
    outcome: Outcome
    findings: list[Finding] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(default_factory=list)
    judge: Optional[dict] = None       # raw judge verdict (validated JSON)
    judge_called: bool = False
    triage_reasons: list[str] = Field(default_factory=list)
    uncertain_notes: list[str] = Field(default_factory=list)  # overrides / rejected LLM findings
    latency_ms: float = 0.0
    rules_ms: float = 0.0
    llm_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model: Optional[str] = None

    @property
    def primary(self) -> Optional[Finding]:
        return self.findings[0] if self.findings else None
