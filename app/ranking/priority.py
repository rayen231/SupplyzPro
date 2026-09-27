"""Issue priority: a prioritization heuristic, not an objective truth.

priority = affected_sessions x impact_weight x mean_confidence
impact_weight: high 5, medium 3, low 1. Confidence tiers map to high 0.9, medium 0.6, low 0.3.
"""
from app.models.finding import Finding
from app.services.merge import TIER_SCORE

IMPACT_WEIGHT = {"high": 5, "medium": 3, "low": 1}


def priority_score(findings: list[Finding], impact: str) -> dict:
    affected = len({f.session_id for f in findings})
    mean_conf = sum(TIER_SCORE[f.tier] for f in findings) / len(findings)
    return {"impact_weight": IMPACT_WEIGHT[impact], "mean_confidence": round(mean_conf, 3),
            "priority": round(affected * IMPACT_WEIGHT[impact] * mean_conf, 2)}
