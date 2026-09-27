"""Run analyzer configurations, score them against ground truth, store results, log to MLflow.

Usage
  python -m evaluation.evaluate                       # rules + llm + hybrid with JUDGE_MODEL
  python -m evaluation.evaluate --modes rules         # no API key needed
  python -m evaluation.evaluate --model llama-3.1-8b-instant --modes llm hybrid
  python -m evaluation.evaluate --no-mlflow
"""
import argparse
import json
import sys
import time

from sklearn.metrics import f1_score

from app import config
from app.ingestion.loader import load_ground_truth, load_sessions
from app.models.finding import SessionAnalysis
from app.services import store
from app.services.pipeline import analyze_all

FAILURE_TYPES = ["unsupported_success_claim", "wrong_entity", "incomplete_request", "repeated_question",
                 "no_progress_loop", "wrong_interpretation", "unsupported_inference"]


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(analyses: list[SessionAnalysis], gt: dict, split: str | None = None) -> dict:
    rows = [(a, gt[a.session_id]) for a in analyses if split is None or gt[a.session_id]["split"] == split]
    if not rows:
        return {}
    tp = sum(a.outcome == "failure" and g["expected_outcome"] == "failure" for a, g in rows)
    fp = sum(a.outcome == "failure" and g["expected_outcome"] != "failure" for a, g in rows)
    fn = sum(a.outcome != "failure" and g["expected_outcome"] == "failure" for a, g in rows)
    neg = sum(g["expected_outcome"] != "failure" for _, g in rows)
    p, r, f = _prf(tp, fp, fn)

    rtp = sum(a.outcome == "recovered" and g["expected_outcome"] == "recovered" for a, g in rows)
    rfp = sum(a.outcome == "recovered" and g["expected_outcome"] != "recovered" for a, g in rows)
    rfn = sum(a.outcome != "recovered" and g["expected_outcome"] == "recovered" for a, g in rows)
    rp, rr, _ = _prf(rtp, rfp, rfn)

    y_true = [g["expected_failure_type"] or "none" for _, g in rows]
    y_pred = [(a.primary.failure_type if a.primary else "none") for a, _ in rows]
    present = [t for t in FAILURE_TYPES if t in y_true]
    type_f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0) if present else 0.0

    grounded = total_tp_findings = 0
    for a, g in rows:
        if a.primary and g["expected_outcome"] == "failure":
            total_tp_findings += 1
            if set(a.primary.evidence_event_ids) & set(g["key_event_ids"]):
                grounded += 1

    n = len(rows)
    llm_rows = [a for a, _ in rows if a.judge_called]
    tin = sum(a.tokens_in for a, _ in rows)
    tout = sum(a.tokens_out for a, _ in rows)
    price = config.MODEL_PRICES.get(next((a.model for a, _ in rows if a.model), ""), (0, 0))
    cost = tin / 1e6 * price[0] + tout / 1e6 * price[1]
    gold_fail = [(a, g) for a, g in rows if g["expected_outcome"] == "failure"]
    return {
        "n_sessions": n,
        "failure_precision": round(p, 4), "failure_recall": round(r, 4), "failure_f1": round(f, 4),
        "false_positive_rate": round(fp / neg, 4) if neg else 0.0,
        "false_negative_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "outcome_accuracy": round(sum(a.outcome == g["expected_outcome"] for a, g in rows) / n, 4),
        "failure_type_macro_f1": round(type_f1, 4),
        "recovery_precision": round(rp, 4), "recovery_recall": round(rr, 4),
        "evidence_grounding_rate": round(grounded / total_tp_findings, 4) if total_tp_findings else 0.0,
        "llm_calls": len(llm_rows), "llm_call_share": round(len(llm_rows) / n, 4),
        "triage_recall": round(sum(bool(a.triage_reasons) for a, _ in gold_fail) / len(gold_fail), 4) if gold_fail else 0.0,
        "tokens_in_total": tin, "tokens_out_total": tout,
        "tokens_in_per_llm_call": round(tin / len(llm_rows), 1) if llm_rows else 0.0,
        "tokens_out_per_llm_call": round(tout / len(llm_rows), 1) if llm_rows else 0.0,
        "rules_ms_mean": round(sum(a.rules_ms for a, _ in rows) / n, 3),
        "llm_ms_mean_per_call": round(sum(a.llm_ms for a in llm_rows) / len(llm_rows), 1) if llm_rows else 0.0,
        "latency_ms_mean_per_session": round(sum(a.latency_ms for a, _ in rows) / n, 1),
        "cost_usd_total": round(cost, 5), "cost_usd_per_1k_sessions": round(cost / n * 1000, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


def predictions_table(analyses, gt):
    out = []
    for a in analyses:
        g = gt[a.session_id]
        out.append({"session_id": a.session_id, "split": g["split"], "family": g["family"],
                    "expected_outcome": g["expected_outcome"], "expected_type": g["expected_failure_type"],
                    "pred_outcome": a.outcome, "pred_type": a.primary.failure_type if a.primary else None,
                    "tier": a.primary.tier if a.primary else None, "judge_called": a.judge_called,
                    "correct": a.outcome == g["expected_outcome"] and (a.primary.failure_type if a.primary else None) == g["expected_failure_type"]})
    return out


def run_config(mode, model, sessions, gt, use_mlflow=True):
    key = store.config_key(mode, model)
    if mode != "rules" and not config.llm_available():
        print(f"[skip] {key}: GROQ_API_KEY not set")
        return None
    print(f"[run] {key} on {len(sessions)} sessions")
    t0 = time.perf_counter()
    analyses, issues = analyze_all(sessions, mode, model,
                                   progress=lambda i, n, sid: print(f"  {i}/{n} {sid}", end="\r", flush=True))
    wall = time.perf_counter() - t0
    fails = [a.session_id for a in analyses if mode != "rules" and not a.judge_called and a.triage_reasons]
    if fails:
        print(f"\n  warning: {len(fails)} sessions had no LLM verdict (rate limit or error); they fell back to rules")
    metrics = {"all": score(analyses, gt), "tuned": score(analyses, gt, "tuned"), "heldout": score(analyses, gt, "heldout"),
               "wall_seconds": round(wall, 1), "missing_llm_verdicts": len(fails)}
    store.save_run(key, analyses, issues, metrics)
    m = metrics["all"]
    print(f"\n  F1 {m['failure_f1']:.3f}  P {m['failure_precision']:.3f}  R {m['failure_recall']:.3f}  "
          f"FPR {m['false_positive_rate']:.3f}  type-F1 {m['failure_type_macro_f1']:.3f}  grounding {m['evidence_grounding_rate']:.3f}  "
          f"heldout F1 {metrics['heldout'].get('failure_f1', 0):.3f}")

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name=key):
                mlflow.log_params({"mode": mode, "model": model if mode != "rules" else "none",
                                   "prompt_version": config.PROMPT_VERSION, "n_sessions": len(sessions)})
                for split in ("all", "tuned", "heldout"):
                    for k, v in metrics[split].items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"{split}.{k}", v)
                mlflow.log_metric("wall_seconds", wall)
                mlflow.log_dict(predictions_table(analyses, gt), "predictions.json")
                mlflow.log_dict(issues, "issues.json")
        except Exception as e:
            print(f"  mlflow logging failed: {e}")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["rules", "llm", "hybrid"])
    ap.add_argument("--model", default=config.JUDGE_MODEL)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    sessions = load_sessions()
    gt = load_ground_truth()
    if not args.no_mlflow and config.llm_available():
        try:
            import mlflow
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
            mlflow.langchain.autolog()  # traces each judge call: prompt, response, latency, tokens
        except Exception as e:
            print(f"mlflow langchain autolog unavailable: {e}")
    summary = {}
    for mode in args.modes:
        summary[store.config_key(mode, args.model)] = run_config(mode, args.model, sessions, gt, not args.no_mlflow)
    json.dump({k: (v or {}).get("all") for k, v in summary.items()}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
