"""SQLite persistence for precomputed analyses, issues and evaluation metrics."""
import json
import sqlite3
from datetime import datetime, timezone

from app.config import DB_PATH
from app.models.finding import SessionAnalysis

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (config TEXT, session_id TEXT, data TEXT, PRIMARY KEY (config, session_id));
CREATE TABLE IF NOT EXISTS issues (config TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS runs (config TEXT PRIMARY KEY, created_at TEXT, metrics TEXT);
"""


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(SCHEMA)
    return c


def config_key(mode: str, model: str | None) -> str:
    return mode if mode == "rules" else f"{mode}:{model}"


def save_run(key: str, analyses: list[SessionAnalysis], issues: list[dict], metrics: dict | None = None):
    with _conn() as c:
        c.execute("DELETE FROM analyses WHERE config = ?", (key,))
        c.executemany("INSERT INTO analyses VALUES (?, ?, ?)",
                      [(key, a.session_id, a.model_dump_json()) for a in analyses])
        c.execute("INSERT OR REPLACE INTO issues VALUES (?, ?)", (key, json.dumps(issues)))
        c.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                  (key, datetime.now(timezone.utc).isoformat(timespec="seconds"), json.dumps(metrics or {})))


def list_configs() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT config, created_at, metrics FROM runs ORDER BY config").fetchall()
    return [{"config": r[0], "created_at": r[1], "metrics": json.loads(r[2])} for r in rows]


def load_analyses(key: str) -> list[SessionAnalysis]:
    with _conn() as c:
        rows = c.execute("SELECT data FROM analyses WHERE config = ? ORDER BY session_id", (key,)).fetchall()
    return [SessionAnalysis.model_validate_json(r[0]) for r in rows]


def load_issues(key: str) -> list[dict]:
    with _conn() as c:
        row = c.execute("SELECT data FROM issues WHERE config = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else []
