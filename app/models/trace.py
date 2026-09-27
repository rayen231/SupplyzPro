"""Agent trace schema: a session is an ordered list of events."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

EventType = Literal["message", "tool_call", "tool_result"]
Role = Literal["user", "agent", "tool", "system"]


class Event(BaseModel):
    event_id: str
    timestamp: str
    event_type: EventType
    role: Role
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict[str, Any]] = None
    tool_result: Any = None


class Session(BaseModel):
    session_id: str
    timestamp: str
    agent_version: str = "erp-agent-v1"
    language: str = "en"
    events: list[Event] = Field(default_factory=list)

    def event(self, event_id: str) -> Optional[Event]:
        return next((e for e in self.events if e.event_id == event_id), None)

    @property
    def event_ids(self) -> set[str]:
        return {e.event_id for e in self.events}

    def user_messages(self) -> list[Event]:
        return [e for e in self.events if e.event_type == "message" and e.role == "user"]

    def agent_messages(self) -> list[Event]:
        return [e for e in self.events if e.event_type == "message" and e.role == "agent"]

    def tool_calls(self) -> list[Event]:
        return [e for e in self.events if e.event_type == "tool_call"]

    def result_for(self, call: Event) -> Optional[Event]:
        """The tool_result immediately following a tool_call."""
        idx = self.events.index(call)
        for e in self.events[idx + 1:]:
            if e.event_type == "tool_result" and e.tool_name == call.tool_name:
                return e
            if e.event_type == "tool_call":
                return None
        return None
