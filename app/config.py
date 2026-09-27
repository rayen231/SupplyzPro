"""Central configuration. Values come from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
ERP_DIR = DATA_DIR / "erp"
TRACES_DIR = DATA_DIR / "traces"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "results.db"
GROUND_TRUTH_PATH = ROOT / "evaluation" / "ground_truth.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "llama-3.1-8b-instant")
PROMPT_VERSION = "v2"

# USD per 1M tokens (input, output). Groq list prices at time of writing;
# verify on https://groq.com/pricing before quoting them.
MODEL_PRICES = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
MLFLOW_EXPERIMENT = "agent_failure_detection"


def llm_available() -> bool:
    return bool(GROQ_API_KEY)
