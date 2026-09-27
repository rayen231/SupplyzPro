"""Structured output schema for the LLM judge."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    user_goal: str = Field(description="One sentence: what the user actually asked for.")
    outcome: Literal["success", "recovered", "clarification_needed", "failure"] = Field(
        description="success: goal achieved and final answer supported by tool evidence. "
                    "recovered: a problem occurred but the agent fixed it and the final answer is correct. "
                    "clarification_needed: the request was genuinely ambiguous and the agent correctly asked. "
                    "failure: the goal was not achieved or the final answer is not supported by evidence.")
    failure_type: Optional[Literal[
        "unsupported_success_claim", "wrong_entity", "incomplete_request", "repeated_question",
        "no_progress_loop", "wrong_interpretation", "unsupported_inference"]] = Field(
        default=None, description="Required when outcome is failure, otherwise null. Pick the primary failure.")
    confidence: float = Field(ge=0.0, le=1.0, description="Your confidence in the outcome, 0 to 1.")
    severity: Literal["low", "medium", "high"] = Field(description="Business impact if this is a failure.")
    explanation: str = Field(description="Two sentences max, citing concrete values from the trace (IDs, names, statuses, numbers).")
    evidence_event_ids: list[str] = Field(description="Event IDs (like E003) that prove the conclusion. Only IDs that appear in the trace.")
