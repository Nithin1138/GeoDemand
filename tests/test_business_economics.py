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
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

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
