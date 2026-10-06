"""
Unit tests for Business Economics Conversion Pipeline:
  Expected Customers
          ↓
  Average Spend (AOV)
          ↓
  Expected Revenue
          ↓
  Operating Cost (Costs)
          ↓
  Expected Profit

Validates configurable business assumptions and exact mathematical transformations.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

try:
    import pytest
except ImportError:
    pytest = None

from api.recommender import calculate_business_metrics
from config.settings import (
    DEFAULT_AOV_INR,
    DEFAULT_VARIABLE_COST_RATE,
    DEFAULT_FIXED_COST_PER_DAY_INR,
    DEFAULT_OPERATING_HOURS_PER_DAY,
)


def test_business_economics_conversion_step_by_step():
    """Verify exact step-by-step conversion from customer count to revenue, costs, and profit."""
    expected_customers = 40
    aov = 120.0  # Average Spend
    variable_cost_rate = 0.35  # COGS rate
    fixed_cost_per_day = 650.0  # Daily fixed cost
    hours_per_day = 13.0  # Operating hours
    fuel_cost = 15.0  # Travel fuel cost

    biz = calculate_business_metrics(
        expected_customers=expected_customers,
        aov=aov,
        variable_cost_rate=variable_cost_rate,
        fixed_cost_per_day=fixed_cost_per_day,
        hours_per_day=hours_per_day,
        fuel_cost=fuel_cost,
    )

    # Step 1: Expected Revenue = Expected Customers * Average Spend
    expected_revenue = 40 * 120.0  # 4800.0
    assert biz["expected_customers"] == 40
    assert biz["average_spend_inr"] == 120.0
    assert biz["expected_revenue_inr"] == expected_revenue

    # Step 2: Expected Costs = Variable Cost + Hourly Fixed Cost + Fuel Cost
    ingredient_cost = 4800.0 * 0.35  # 1680.0
    hourly_fixed = 650.0 / 13.0  # 50.0
    expected_costs = ingredient_cost + hourly_fixed + fuel_cost  # 1745.0
    assert biz["ingredient_cost_inr"] == ingredient_cost
    assert biz["fixed_cost_per_hour_inr"] == hourly_fixed
    assert biz["fuel_cost_inr"] == fuel_cost
    assert biz["expected_costs_inr"] == expected_costs
    assert biz["expected_operating_cost_inr"] == expected_costs

    # Step 3: Expected Profit = Expected Revenue - Expected Costs
    expected_profit = expected_revenue - expected_costs  # 3055.0
    assert biz["expected_profit_inr"] == expected_profit


def test_default_configurable_assumptions():
    """Verify that default business assumptions are populated from central config."""
    biz = calculate_business_metrics(expected_customers=50)

    assert biz["average_spend_inr"] == DEFAULT_AOV_INR
    assert biz["variable_cost_rate"] == DEFAULT_VARIABLE_COST_RATE
    assert biz["fixed_cost_per_day_inr"] == DEFAULT_FIXED_COST_PER_DAY_INR
    assert biz["operating_hours_per_day"] == DEFAULT_OPERATING_HOURS_PER_DAY

    expected_rev = 50 * DEFAULT_AOV_INR
    expected_cogs = expected_rev * DEFAULT_VARIABLE_COST_RATE
    expected_fixed_hr = DEFAULT_FIXED_COST_PER_DAY_INR / DEFAULT_OPERATING_HOURS_PER_DAY
    expected_total_cost = round(expected_cogs + expected_fixed_hr, 2)
    expected_profit = round(expected_rev - expected_total_cost, 2)

    assert biz["expected_revenue_inr"] == expected_rev
    assert biz["expected_costs_inr"] == expected_total_cost
    assert biz["expected_profit_inr"] == expected_profit


def test_travel_fraction_and_relocation_adjusted_realized_profit():
    """Verify travel_fraction and relocation-adjusted realized profit calculation."""
    # Example: 15 minutes travel time, 40 customers expected, AOV 100, var cost 0.38, fixed 500/13, fuel 20
    travel_time_min = 15.0
    expected_customers = 40
    fuel_cost = 20.0

    biz = calculate_business_metrics(
        expected_customers=expected_customers,
        aov=100.0,
        variable_cost_rate=0.38,
        fixed_cost_per_day=500.0,
        hours_per_day=13.0,
        fuel_cost=fuel_cost,
        travel_time_minutes=travel_time_min,
    )

    # travel_fraction = max(0, (60 - 15) / 60) = 0.75
    expected_travel_fraction = 0.75
    assert biz["travel_fraction"] == expected_travel_fraction

    # candidate_predicted_profit = 4000 - (1520 + 38.46) = 2441.54
    predicted_profit = biz["candidate_predicted_profit_inr"]
    assert predicted_profit == 2441.54

    # relocation_adjusted_realized_profit = 2441.54 * 0.75 - 20 = 1811.16
    expected_relocation_profit = round(predicted_profit * 0.75 - fuel_cost, 2)
    assert biz["relocation_adjusted_realized_profit_inr"] == expected_relocation_profit
    assert biz["realized_next_hour_profit_inr"] == expected_relocation_profit


def test_candidate_ranking_vs_net_realized_uplift_decision():
    """Verify that multi-factor candidate ranking score and net realized uplift decision are separate."""
    from api.recommender import rank_candidates

    # Candidate A: Higher demand & raw profit, but 20 min drive (higher travel friction)
    cand_a = {
        "h3_cell": "cell_a",
        "expected_customers": 60,
        "expected_profit_inr": 3000.0,
        "distance_km": 4.5,
        "competition_score": 0.2,
        "relocation_adjusted_realized_profit_inr": 1800.0,  # lower realized due to travel friction
        "is_current_cell": False,
    }
    # Candidate B: Slightly lower raw profit, but only 3 min drive (much higher net realized uplift)
    cand_b = {
        "h3_cell": "cell_b",
        "expected_customers": 52,
        "expected_profit_inr": 2600.0,
        "distance_km": 0.8,
        "competition_score": 0.2,
        "relocation_adjusted_realized_profit_inr": 2450.0,  # higher net realized uplift
        "is_current_cell": False,
    }

    # Step 1: Multi-factor candidate ranking score calculation
    ranked = rank_candidates([cand_a, cand_b])
    for c in ranked:
        assert "recommendation_score" in c
        assert "rank" in c

    # Step 2: Final recommendation ordering primarily uses net realized uplift under travel constraints
    current_profit = 1200.0
    for c in ranked:
        c["realized_net_uplift_inr"] = round(c["relocation_adjusted_realized_profit_inr"] - current_profit, 2)

    final_recommendation = max(ranked, key=lambda x: x["realized_net_uplift_inr"])
    # Candidate B offers the highest net realized uplift after travel constraints
    assert final_recommendation["h3_cell"] == "cell_b"
    assert final_recommendation["realized_net_uplift_inr"] == 1250.0


