"""
Unit test suite for the Configurable Candidate Ranking and Strategic Decision Engine.

Validates:
  1. Configurable candidate ranking with weights
  2. Multi-factor components: demand, profit, travel, competition, affinity, uncertainty
  3. Current-location vs candidate-location profit calculations
  4. Travel-adjusted profit uplift
  5. Distance limits and travel-time limits enforcement
  6. Decision verdicts: STAY_PUT, CONSIDER_MOVE, RECOMMENDED_MOVE
  7. Threshold validation
  8. Natural language `decision_reason` generation
  9. Ranked Top-5 recommendations generation
  10. Safety invariant: High ranking alone CANNOT force a move recommendation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

import pytest

from api.decision_engine import (
    MAX_TRAVEL_DIST_KM,
    MAX_TRAVEL_TIME_MIN,
    RECOMMENDED_MOVE_MIN_UPLIFT_INR,
    evaluate_decision,
    select_top_recommendations,
)
from api.ranking_engine import rank_candidates


@pytest.fixture
def baseline_current_location():
    """Mock baseline location where the vendor is currently situated."""
    return {
        "h3_cell": "88619aa6c7fffff",
        "landmark_name": "Benz Circle Origin",
        "distance_km": 0.0,
        "estimated_travel_time_min": 0.0,
        "is_current_cell": True,
        "expected_customers": 35,
        "expected_profit_inr": 1500.0,
        "relocation_adjusted_realized_profit_inr": 1500.0,
        "realized_next_hour_profit_inr": 1500.0,
    }


# ---------------------------------------------------------------------------
# 1 & 2. Configurable Candidate Ranking & Multi-Factor Components
# ---------------------------------------------------------------------------

def test_configurable_candidate_ranking():
    """Verify candidate ranking computes scores and respects configurable weight overrides."""
    cands = [
        {
            "h3_cell": "cell_1",
            "expected_customers": 50,
            "expected_profit_inr": 2500.0,
            "relocation_adjusted_realized_profit_inr": 2300.0,
            "distance_km": 1.0,
            "competition_score": 0.2,
            "category_affinity": 1.5,
            "prediction_interval": {"p10": 40, "p50": 50, "p90": 60},
        },
        {
            "h3_cell": "cell_2",
            "expected_customers": 40,
            "expected_profit_inr": 1800.0,
            "relocation_adjusted_realized_profit_inr": 1600.0,
            "distance_km": 2.5,
            "competition_score": 0.6,
            "category_affinity": 0.8,
            "prediction_interval": {"p10": 20, "p50": 40, "p90": 70},
        },
    ]

    ranked = rank_candidates(cands, vendor_category="food")
    assert len(ranked) == 2
    assert ranked[0]["h3_cell"] == "cell_1"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[0]["recommendation_score"] > ranked[1]["recommendation_score"]

    # Test weight override focusing 100% on distance penalty
    override = {"profit": 0.0, "demand": 0.0, "affinity": 0.0, "distance": 1.0, "competition": 0.0, "uncertainty": 0.0}
    re_ranked = rank_candidates(cands, weights_override=override)
    # Cell 1 has shorter distance (1.0 vs 2.5), so it still achieves higher score (less penalty)
    assert re_ranked[0]["h3_cell"] == "cell_1"


def test_ranking_incorporates_uncertainty_spread():
    """Verify candidates with wide prediction uncertainty (risk) are penalized."""
    # Both candidates have identical profit and demand, but cell_risky has wide P10-P90 spread
    safe_cand = {
        "h3_cell": "cell_safe",
        "expected_customers": 45,
        "relocation_adjusted_realized_profit_inr": 2000.0,
        "distance_km": 1.5,
        "competition_score": 0.3,
        "category_affinity": 1.0,
        "prediction_interval": {"p10": 42, "p50": 45, "p90": 48},  # Tight spread (6)
    }
    risky_cand = {
        "h3_cell": "cell_risky",
        "expected_customers": 45,
        "relocation_adjusted_realized_profit_inr": 2000.0,
        "distance_km": 1.5,
        "competition_score": 0.3,
        "category_affinity": 1.0,
        "prediction_interval": {"p10": 15, "p50": 45, "p90": 85},  # High volatility spread (70)
    }

    ranked = rank_candidates([safe_cand, risky_cand])
    assert ranked[0]["h3_cell"] == "cell_safe"
    assert ranked[0]["recommendation_score"] > ranked[1]["recommendation_score"]


# ---------------------------------------------------------------------------
# 3 & 4. Current vs Candidate Profit & Travel-Adjusted Uplift
# ---------------------------------------------------------------------------

def test_profit_uplift_and_current_location_anchor(baseline_current_location):
    """Verify net uplift is zero for origin anchor and correctly calculated for candidates."""
    # Evaluating current location against itself must return STAY_PUT with 0 uplift
    current_dec = evaluate_decision(baseline_current_location, baseline_current_location)
    assert current_dec.decision_verdict == "STAY_PUT"
    assert current_dec.decision_action == "STAY"
    assert current_dec.realized_net_uplift_inr == 0.0
    assert not current_dec.is_move_recommended

    # Candidate with higher profit
    high_cand = {
        "h3_cell": "cell_high",
        "landmark_name": "MG Road Hotspot",
        "relocation_adjusted_realized_profit_inr": 1850.0,
        "candidate_predicted_profit_inr": 2000.0,
        "distance_km": 1.2,
        "estimated_travel_time_min": 6.0,
        "is_current_cell": False,
    }
    cand_dec = evaluate_decision(high_cand, baseline_current_location)
    assert cand_dec.current_profit == 1500.0
    assert cand_dec.recommended_profit == 1850.0
    assert cand_dec.realized_net_uplift_inr == 350.0  # 1850 - 1500


# ---------------------------------------------------------------------------
# 5, 6 & 7. Decision Policies & Threshold Validation
# ---------------------------------------------------------------------------

def test_recommended_move_policy(baseline_current_location):
    """Verify RECOMMENDED_MOVE requires sufficient uplift AND compliant travel limits."""
    cand = {
        "h3_cell": "cell_rec",
        "landmark_name": "PVP Mall Zone",
        "relocation_adjusted_realized_profit_inr": 1650.0,  # +150 INR/hr uplift
        "candidate_predicted_profit_inr": 1800.0,
        "distance_km": 1.8,  # <= 3.0 km
        "estimated_travel_time_min": 8.0,  # <= 12 min
        "is_current_cell": False,
    }
    dec = evaluate_decision(cand, baseline_current_location)
    assert dec.decision_verdict == "RECOMMENDED_MOVE"
    assert dec.decision_action == "RECOMMENDED MOVE"
    assert dec.is_move_recommended is True
    assert dec.passed_travel_constraints is True
    assert dec.passed_profit_threshold is True
    assert "Strong recommendation" in dec.decision_reason


def test_consider_move_marginal_uplift(baseline_current_location):
    """Verify CONSIDER_MOVE for positive but modest profit gains (e.g. +₹20/hr)."""
    cand = {
        "h3_cell": "cell_modest",
        "landmark_name": "Governorpet Corner",
        "relocation_adjusted_realized_profit_inr": 1520.0,  # +20 INR/hr uplift (< 40 threshold)
        "candidate_predicted_profit_inr": 1530.0,  # modest raw diff (30 < 60)
        "distance_km": 1.0,
        "estimated_travel_time_min": 5.0,
        "is_current_cell": False,
    }
    dec = evaluate_decision(cand, baseline_current_location)
    assert dec.decision_verdict == "CONSIDER_MOVE"
    assert dec.decision_action == "CONSIDER MOVE"
    assert dec.is_move_recommended is False
    assert "Positive opportunity" in dec.decision_reason


def test_stay_put_policy_on_losses(baseline_current_location):
    """Verify STAY_PUT when relocation friction causes financial loss."""
    cand = {
        "h3_cell": "cell_loss",
        "landmark_name": "Outskirts Bypass",
        "relocation_adjusted_realized_profit_inr": 1380.0,  # -120 INR/hr loss vs 1500
        "candidate_predicted_profit_inr": 1450.0,
        "distance_km": 2.5,
        "estimated_travel_time_min": 10.0,
        "is_current_cell": False,
    }
    dec = evaluate_decision(cand, baseline_current_location)
    assert dec.decision_verdict == "STAY_PUT"
    assert dec.decision_action == "STAY"
    assert dec.is_move_recommended is False
    assert "financially optimal" in dec.decision_reason


# ---------------------------------------------------------------------------
# 10. CRUCIAL SAFETY RULE: High Ranking Alone Cannot Force a Move
# ---------------------------------------------------------------------------

def test_high_ranking_alone_cannot_force_move_recommendation(baseline_current_location):
    """
    CRITICAL SAFETY TEST:
    A candidate may rank #1 by score due to massive footfall, BUT if it violates
    distance limits (> 3.0 km) or travel time (> 12 min), it CANNOT be recommended to move!
    """
    # Candidate with huge demand and high score, but 4.8 km away (exceeds 3.0 km limit)
    far_hotspot = {
        "h3_cell": "cell_far_hotspot",
        "landmark_name": "Gunadala Mega Center",
        "expected_customers": 120,
        "expected_profit_inr": 4500.0,
        "candidate_predicted_profit_inr": 4500.0,
        "relocation_adjusted_realized_profit_inr": 2200.0,  # +700 INR/hr uplift!
        "distance_km": 4.8,  # VIOLATION: > 3.0 km
        "estimated_travel_time_min": 22.0,  # VIOLATION: > 12.0 min
        "competition_score": 0.1,
        "category_affinity": 2.0,
        "is_current_cell": False,
    }

    # Verify that in a candidate ranking, it indeed ranks #1
    ranked = rank_candidates([far_hotspot, {
        "h3_cell": "cell_local",
        "landmark_name": "Local Street",
        "expected_customers": 40,
        "expected_profit_inr": 1800.0,
        "candidate_predicted_profit_inr": 1800.0,
        "relocation_adjusted_realized_profit_inr": 1650.0,
        "distance_km": 1.2,
        "estimated_travel_time_min": 6.0,
        "competition_score": 0.3,
        "category_affinity": 1.0,
        "is_current_cell": False,
    }])
    assert ranked[0]["h3_cell"] == "cell_far_hotspot"
    assert ranked[0]["rank"] == 1

    # BUT the decision engine MUST reject RECOMMENDED_MOVE!
    dec = evaluate_decision(far_hotspot, baseline_current_location)
    assert dec.decision_verdict != "RECOMMENDED_MOVE", "High rank violated travel constraints but was recommended!"
    assert dec.decision_verdict == "CONSIDER_MOVE"
    assert dec.is_move_recommended is False
    assert dec.passed_travel_constraints is False
    assert "exceeds" in dec.decision_reason


# ---------------------------------------------------------------------------
# 8 & 9. Decision Reason & Top-5 Recommendations Selection
# ---------------------------------------------------------------------------

def test_top_5_recommendations_selection(baseline_current_location):
    """Verify select_top_recommendations selects and formats top N items with decision metadata."""
    mock_candidates = []
    for i in range(10):
        mock_candidates.append({
            "h3_cell": f"cell_{i}",
            "landmark_name": f"Landmark {i}",
            "expected_customers": 30 + i * 5,
            "expected_profit_inr": 1400.0 + i * 100,
            "candidate_predicted_profit_inr": 1400.0 + i * 100,
            "relocation_adjusted_realized_profit_inr": 1400.0 + i * 80,
            "distance_km": 1.0 + (i * 0.3),
            "estimated_travel_time_min": 4.0 + (i * 1.5),
            "competition_score": 0.2,
            "category_affinity": 1.0,
            "is_current_cell": False,
        })

    ranked = rank_candidates(mock_candidates)
    top_5 = select_top_recommendations(ranked, baseline_current_location, top_n=5)

    assert len(top_5) == 5
    for idx, cand in enumerate(top_5):
        assert cand["rank"] == idx + 1
        assert "decision_verdict" in cand
        assert "decision_reason" in cand
        assert "realized_net_uplift_inr" in cand
        assert isinstance(cand["decision_reason"], str)
        assert len(cand["decision_reason"]) > 0
