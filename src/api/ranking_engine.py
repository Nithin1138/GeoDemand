"""
GeoDemand AI — Configurable Multi-Factor Candidate Ranking Engine

Ranks potential relocation candidates based on multi-objective scoring:
  - Expected realized profit (weight: 0.40)
  - Demand predictions / footfall (weight: 0.20)
  - Vendor preferences & category spatial affinity (weight: 0.20)
  - Travel distance friction (weight: -0.10)
  - Competition intensity (weight: -0.05)
  - Prediction uncertainty spread (weight: -0.05)

All weights are configurable via environment variables or parameter overrides.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


# Default ranking weights (must balance profitability, volume, friction, and risk)
DEFAULT_WEIGHT_PROFIT = float(os.getenv("RANK_WEIGHT_PROFIT", "0.40"))
DEFAULT_WEIGHT_DEMAND = float(os.getenv("RANK_WEIGHT_DEMAND", "0.20"))
DEFAULT_WEIGHT_AFFINITY = float(os.getenv("RANK_WEIGHT_AFFINITY", "0.20"))
DEFAULT_WEIGHT_DISTANCE = float(os.getenv("RANK_WEIGHT_DISTANCE", "0.10"))
DEFAULT_WEIGHT_COMPETITION = float(os.getenv("RANK_WEIGHT_COMPETITION", "0.05"))
DEFAULT_WEIGHT_UNCERTAINTY = float(os.getenv("RANK_WEIGHT_UNCERTAINTY", "0.05"))


def _normalize(values: List[float], invert: bool = False) -> List[float]:
    """Min-max normalize a list of numerical values into [0.0, 1.0]."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    span = mx - mn
    if invert:
        return [(mx - v) / span for v in values]
    return [(v - mn) / span for v in values]


def rank_candidates(
    candidates: List[dict],
    vendor_category: str = "food",
    weights_override: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Rank candidate cells using a configurable multi-objective scoring model.

    Formula:
      Score = w_profit * profit_norm
            + w_demand * demand_norm
            + w_affinity * affinity_norm
            - w_distance * distance_norm
            - w_competition * competition_norm
            - w_uncertainty * uncertainty_spread_norm

    Guarantees:
      - Assigns 'recommendation_score' in [0.0, 1.0]
      - Assigns sequential 'rank' (1, 2, 3...)
      - Sorts descending by recommendation_score
    """
    if not candidates:
        return []

    weights = {
        "profit": DEFAULT_WEIGHT_PROFIT,
        "demand": DEFAULT_WEIGHT_DEMAND,
        "affinity": DEFAULT_WEIGHT_AFFINITY,
        "distance": DEFAULT_WEIGHT_DISTANCE,
        "competition": DEFAULT_WEIGHT_COMPETITION,
        "uncertainty": DEFAULT_WEIGHT_UNCERTAINTY,
    }
    if weights_override:
        weights.update(weights_override)

    n = len(candidates)

    # 1. Extract raw feature vectors
    profits = [
        float(c.get("relocation_adjusted_realized_profit_inr", c.get("realized_next_hour_profit_inr", c.get("expected_profit_inr", 0.0))))
        for c in candidates
    ]
    demands = [float(c.get("expected_customers", 0.0)) for c in candidates]
    distances = [float(c.get("distance_km", 0.0)) for c in candidates]
    competitions = [float(c.get("competition_score", 0.3)) for c in candidates]

    # Category affinity (vendor preferences)
    has_affinities = any("category_affinity" in c for c in candidates)
    affinities = [float(c.get("category_affinity", 1.0)) for c in candidates] if has_affinities else [1.0] * n

    # Uncertainty spread: (P90 - P10) / max(P50, 1)
    uncertainties = []
    for c in candidates:
        interval = c.get("prediction_interval") or c.get("confidence_interval") or {}
        p10 = float(interval.get("p10", 0.0))
        p90 = float(interval.get("p90", 0.0))
        p50 = float(interval.get("p50", c.get("expected_customers", 1.0)))
        spread = max(0.0, (p90 - p10) / max(1.0, p50))
        uncertainties.append(spread)

    # 2. Normalize components into [0.0, 1.0]
    p_norm = _normalize(profits)
    d_norm = _normalize(demands)
    dist_norm = _normalize(distances)
    comp_norm = _normalize(competitions)
    aff_norm = _normalize(affinities) if has_affinities else [0.5] * n
    unc_norm = _normalize(uncertainties)

    # 3. Compute composite recommendation score
    w_p = weights["profit"]
    w_d = weights["demand"]
    w_a = weights["affinity"] if has_affinities else 0.0
    w_dist = weights["distance"]
    w_comp = weights["competition"]
    w_unc = weights["uncertainty"]

    # Re-normalize positive weights if affinity not present (backward compatibility)
    if not has_affinities:
        w_p = 0.60
        w_d = 0.25
        w_dist = 0.10
        w_comp = 0.05
        w_unc = 0.00

    for i, c in enumerate(candidates):
        score = (
            w_p * p_norm[i]
            + w_d * d_norm[i]
            + w_a * aff_norm[i]
            - w_dist * dist_norm[i]
            - w_comp * comp_norm[i]
            - w_unc * unc_norm[i]
        )
        c["recommendation_score"] = round(float(score), 4)

    # 4. Sort and assign integer ranks
    ranked = sorted(candidates, key=lambda x: x["recommendation_score"], reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1

    return ranked
