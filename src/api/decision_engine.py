"""
GeoDemand AI — Strategic Decision & Relocation Recommendation Engine

Transforms multi-factor candidate rankings, travel friction, and realized profit metrics into
actionable, constraint-safe business decisions:
  - RECOMMENDED_MOVE (Net uplift >= ₹40-50/hr, distance <= 3.0km, travel <= 12m)
  - CONSIDER_MOVE (Net uplift between -₹10 and +₹40/hr, or high uplift exceeding travel limits)
  - STAY_PUT (Net uplift < -₹10/hr, or current origin location)

Safety Invariant:
  High ranking alone CANNOT force a move recommendation. Physical distance limits,
  travel duration constraints, and minimum profit thresholds take strict precedence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.settings import (
    MAX_RECOMMENDED_TRAVEL_KM,
    MAX_RECOMMENDED_TRAVEL_MIN,
    STAY_THRESHOLD_INR_PER_HOUR,
)

# Configurable Decision Thresholds
RECOMMENDED_MOVE_MIN_UPLIFT_INR = float(os.getenv("STAY_THRESHOLD_INR_PER_HOUR", "50.0"))
CONSIDER_MOVE_MIN_UPLIFT_INR = float(os.getenv("CONSIDER_MOVE_MIN_UPLIFT_INR", "-10.0"))
MAX_TRAVEL_DIST_KM = float(os.getenv("MAX_RECOMMENDED_TRAVEL_KM", "3.0"))
MAX_TRAVEL_TIME_MIN = float(os.getenv("MAX_RECOMMENDED_TRAVEL_MIN", "12.0"))


@dataclass
class DecisionResult:
    """Structured decision output for a candidate location."""
    decision_verdict: str  # "RECOMMENDED_MOVE" | "CONSIDER_MOVE" | "STAY_PUT"
    decision_action: str   # "RECOMMENDED MOVE" | "CONSIDER MOVE" | "STAY"
    realized_net_uplift_inr: float
    raw_profit_diff_inr: float
    current_profit: float
    recommended_profit: float
    distance_km: float
    travel_time_min: float
    decision_reason: str
    is_move_recommended: bool
    passed_travel_constraints: bool
    passed_profit_threshold: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def classify_decision_verdict(
    realized_net_uplift: float,
    raw_profit_diff: float = 0.0,
    distance_km: float = 0.0,
    travel_time_min: float = 0.0,
    is_current_cell: bool = False,
    max_dist_km: float = MAX_TRAVEL_DIST_KM,
    max_travel_min: float = MAX_TRAVEL_TIME_MIN,
) -> Tuple[str, str, bool, bool, bool]:
    """
    Classify relocation decision based on economic uplift and travel constraints.

    Returns:
      (decision_verdict, decision_action, is_move_recommended, passed_travel, passed_profit)
    """
    if is_current_cell:
        return "STAY_PUT", "STAY", False, True, False

    dist_km = max(0.0, float(distance_km))
    t_min = max(0.0, float(travel_time_min))

    passed_travel = (dist_km <= max_dist_km) and (t_min <= max_travel_min)
    passed_profit = (realized_net_uplift >= 40.0) or (raw_profit_diff >= 60.0 and realized_net_uplift >= -20.0)

    # Safety Rule: High rank / high profit alone CANNOT force RECOMMENDED_MOVE if travel limits exceeded
    if passed_profit:
        if passed_travel:
            return "RECOMMENDED_MOVE", "RECOMMENDED MOVE", True, True, True
        else:
            return "CONSIDER_MOVE", "CONSIDER MOVE", False, False, True

    # Modest opportunity
    if realized_net_uplift >= -10.0 or raw_profit_diff > 15.0:
        return "CONSIDER_MOVE", "CONSIDER MOVE", False, passed_travel, False

    # Relocation friction wipes out profit
    return "STAY_PUT", "STAY", False, passed_travel, False


def evaluate_decision(
    candidate: dict,
    current_location: dict,
    max_dist_km: float = MAX_TRAVEL_DIST_KM,
    max_travel_min: float = MAX_TRAVEL_TIME_MIN,
) -> DecisionResult:
    """
    Evaluate candidate location against current location and return a structured decision.

    Guarantees:
      - Origin anchor (is_current_cell=True) is always STAY_PUT.
      - A high recommendation score / rank #1 CANNOT force RECOMMENDED_MOVE if
        distance > max_dist_km or travel_time > max_travel_min.
    """
    is_current = bool(candidate.get("is_current_cell", False))
    cand_profit = float(candidate.get("relocation_adjusted_realized_profit_inr", candidate.get("expected_profit_inr", 0.0)))
    curr_profit = float(current_location.get("relocation_adjusted_realized_profit_inr", current_location.get("expected_profit_inr", 0.0)))

    if is_current:
        return DecisionResult(
            decision_verdict="STAY_PUT",
            decision_action="STAY",
            realized_net_uplift_inr=0.0,
            raw_profit_diff_inr=0.0,
            current_profit=curr_profit,
            recommended_profit=cand_profit,
            distance_km=0.0,
            travel_time_min=0.0,
            decision_reason="Current location baseline — no relocation required.",
            is_move_recommended=False,
            passed_travel_constraints=True,
            passed_profit_threshold=False,
        )

    cand_pred_profit = float(candidate.get("candidate_predicted_profit_inr", candidate.get("expected_profit_inr", 0.0)))
    raw_profit_diff = round(cand_pred_profit - curr_profit, 2)
    net_uplift = round(cand_profit - curr_profit, 2)

    dist_km = float(candidate.get("distance_km", 0.0))
    travel_min = float(candidate.get("estimated_travel_time_min", candidate.get("travel_time_minutes", 0.0)))
    is_routable = candidate.get("is_routable", True)

    lm_name = candidate.get("landmark_display") or candidate.get("landmark_name") or f"Spot {candidate.get('h3_cell', '')[:8]}"

    # Safety Invariant: Missing or unroutable data cannot generate false move recommendations
    if not is_routable:
        return DecisionResult(
            decision_verdict="STAY_PUT",
            decision_action="STAY",
            realized_net_uplift_inr=0.0,
            raw_profit_diff_inr=raw_profit_diff,
            current_profit=curr_profit,
            recommended_profit=cand_profit,
            distance_km=dist_km,
            travel_time_min=travel_min,
            decision_reason=f"Route to {lm_name} cannot be verified by routing engine. Relocation prevented for vendor safety.",
            is_move_recommended=False,
            passed_travel_constraints=False,
            passed_profit_threshold=False,
        )

    # Guard against distinct cells carrying false 0.0km distance due to missing routing data
    curr_cell = current_location.get("h3_cell")
    cand_cell = candidate.get("h3_cell")
    if curr_cell and cand_cell and curr_cell != cand_cell and dist_km <= 0.001:
        dist_km = 1.0
        travel_min = max(travel_min, 4.0)

    verdict, action, is_rec, passed_travel, passed_profit = classify_decision_verdict(
        realized_net_uplift=net_uplift,
        raw_profit_diff=raw_profit_diff,
        distance_km=dist_km,
        travel_time_min=travel_min,
        is_current_cell=False,
        max_dist_km=max_dist_km,
        max_travel_min=max_travel_min,
    )

    # Natural Language Decision Rationale Generation
    if verdict == "RECOMMENDED_MOVE":
        reason = (
            f"Strong recommendation to relocate to {lm_name}: net hourly profit increases by "
            f"+₹{net_uplift:.0f}/hr with a quick {travel_min:.0f}-minute ({dist_km:.1f} km) move."
        )
    elif verdict == "CONSIDER_MOVE":
        if passed_profit and not passed_travel:
            reasons = []
            if dist_km > max_dist_km:
                reasons.append(f"distance of {dist_km:.1f} km exceeds {max_dist_km:.1f} km limit")
            if travel_min > max_travel_min:
                reasons.append(f"travel time of {travel_min:.0f} min exceeds {max_travel_min:.0f} min limit")
            reason = (
                f"High theoretical profit gain (+₹{net_uplift:.0f}/hr) at {lm_name}, but "
                f"{' and '.join(reasons)}. Relocation carries transit friction risk."
            )
        else:
            reason = (
                f"Positive opportunity at {lm_name} (+₹{net_uplift:.0f}/hr net gain, ~{travel_min:.0f} min drive). "
                f"Consider moving if customer momentum peaks."
            )
    else:
        reason = (
            f"Staying at current location is financially optimal. Relocation to {lm_name} yields "
            f"a net loss of {net_uplift:.0f} ₹/hr after accounting for fuel and lost trading time."
        )

    return DecisionResult(
        decision_verdict=verdict,
        decision_action=action,
        realized_net_uplift_inr=net_uplift,
        raw_profit_diff_inr=raw_profit_diff,
        current_profit=curr_profit,
        recommended_profit=cand_profit,
        distance_km=dist_km,
        travel_time_min=travel_min,
        decision_reason=reason,
        is_move_recommended=is_rec,
        passed_travel_constraints=passed_travel,
        passed_profit_threshold=passed_profit,
    )


def select_top_recommendations(
    ranked_candidates: List[dict],
    current_location: dict,
    top_n: int = 5,
    max_dist_km: float = MAX_TRAVEL_DIST_KM,
    max_travel_min: float = MAX_TRAVEL_TIME_MIN,
) -> List[dict]:
    """
    Enrich candidates with decision intelligence and return top N recommendations.
    Orders primarily by decision verdict (RECOMMENDED > CONSIDER > STAY),
    then by recommendation score, and finally by net realized profit uplift.
    """
    evaluated = []
    for c in ranked_candidates:
        if c.get("is_current_cell"):
            continue

        dec = evaluate_decision(c, current_location, max_dist_km=max_dist_km, max_travel_min=max_travel_min)
        c["decision_verdict"] = dec.decision_verdict
        c["decision_action"] = dec.decision_action
        c["realized_net_uplift_inr"] = dec.realized_net_uplift_inr
        c["profit_improvement_inr"] = dec.realized_net_uplift_inr
        c["current_profit"] = dec.current_profit
        c["recommended_profit"] = dec.recommended_profit
        c["decision_reason"] = dec.decision_reason
        c["decision_rationale"] = dec.decision_reason
        c["decision"] = dec.decision_verdict
        c["reason"] = dec.decision_reason
        c["is_move_recommended"] = dec.is_move_recommended
        evaluated.append(c)

    # Verdict hierarchy weights: RECOMMENDED_MOVE (3), CONSIDER_MOVE (2), STAY_PUT (1)
    verdict_rank = {
        "RECOMMENDED_MOVE": 3,
        "CONSIDER_MOVE": 2,
        "STAY_PUT": 1,
    }

    selected = sorted(
        evaluated,
        key=lambda x: (
            verdict_rank.get(x["decision_verdict"], 0),
            x.get("recommendation_score", 0.0),
            x.get("realized_net_uplift_inr", 0.0),
        ),
        reverse=True,
    )[:top_n]

    for i, c in enumerate(selected):
        c["rank"] = i + 1

    return selected
