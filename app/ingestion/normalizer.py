"""Compress a session into a compact, ID-tagged text for the LLM.

Drops timestamps and metadata, keeps every event ID, truncates long result lists.
"""
import json

from app.models.trace import Session

MAX_LIST = 8


def _trim(obj):
    if isinstance(obj, list):
        out = [_trim(x) for x in obj[:MAX_LIST]]
        if len(obj) > MAX_LIST:
            out.append(f"... {len(obj) - MAX_LIST} more")
        return out
    if isinstance(obj, dict):
        return {k: _trim(v) for k, v in obj.items() if v is not None}
    return obj


def normalize(s: Session) -> str:
    lines = []
    for e in s.events:
        if e.event_type == "message":
            lines.append(f"[{e.event_id}] {e.role.upper()}: {e.content}")
        elif e.event_type == "tool_call":
            lines.append(f"[{e.event_id}] AGENT CALLS {e.tool_name} {json.dumps(e.tool_arguments or {}, ensure_ascii=False)}")
        else:
            lines.append(f"[{e.event_id}] TOOL RESULT {e.tool_name}: {json.dumps(_trim(e.tool_result), ensure_ascii=False, separators=(',', ':'))}")
    return "\n".join(lines)
