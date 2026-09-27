"""Load traces and ground truth."""
import json
from pathlib import Path

from app.config import GROUND_TRUTH_PATH, TRACES_DIR
from app.models.trace import Session


def load_sessions(path: Path | None = None) -> list[Session]:
    path = path or TRACES_DIR / "all_traces.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Session.model_validate_json(line) for line in f if line.strip()]


def load_ground_truth(path: Path | None = None) -> dict[str, dict]:
    rows = json.loads((path or GROUND_TRUTH_PATH).read_text(encoding="utf-8"))
    return {r["session_id"]: r for r in rows}
