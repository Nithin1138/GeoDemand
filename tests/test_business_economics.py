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

import pytest

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


# ---------------------------------------------------------------------------
# Additional Category Pricing, Cost Breakdown & Inventory Constraint Tests
# ---------------------------------------------------------------------------

def test_product_categories_and_pricing_configs():
    """Verify all 6 product categories, item catalogs, and weighted margins."""
    from config.product_pricing import PRODUCT_CATEGORIES, get_category_pricing

    expected_categories = {"food", "fruits", "beverages", "repair", "salon", "snacks"}
    assert set(PRODUCT_CATEGORIES.keys()) == expected_categories

    for cat_id, cfg in PRODUCT_CATEGORIES.items():
        assert cfg.category_id == cat_id
        assert cfg.default_aov_inr > 0
        assert 0.10 <= cfg.default_cogs_rate <= 0.50
        assert 0.0 <= cfg.spoilage_rate <= 0.15
        assert cfg.default_inventory_capacity > 0

        # Verify item catalog mix shares sum to 1.0
        mix_sum = sum(item.mix_share for item in cfg.items)
        assert abs(mix_sum - 1.0) < 1e-4, f"Mix shares for {cat_id} sum to {mix_sum}, expected 1.0"

        # Verify weighted ASP and COGS properties
        assert cfg.weighted_asp_inr > 0
        assert cfg.weighted_cogs_inr > 0
        assert cfg.weighted_cogs_rate > 0
        assert cfg.weighted_asp_inr > cfg.weighted_cogs_inr

    # Test helper with fallback
    food_cfg = get_category_pricing("food")
    assert food_cfg.category_id == "food"
    unknown_cfg = get_category_pricing("unknown_cat")
    assert unknown_cfg.category_id == "food"


def test_inventory_capacity_limits():
    """Verify inventory and throughput capacity constraints cap effective customer count."""
    from api.business_economics import calculate_effective_demand

    # Case 1: Demand under capacity
    effective, unserved = calculate_effective_demand(expected_customers=30, inventory_capacity=50)
    assert effective == 30
    assert unserved == 0

    # Case 2: Demand exceeding capacity (stockout)
    effective, unserved = calculate_effective_demand(expected_customers=65, inventory_capacity=45)
    assert effective == 45
    assert unserved == 20

    # Case 3: No capacity limit specified
    effective, unserved = calculate_effective_demand(expected_customers=70, inventory_capacity=None)
    assert effective == 70
    assert unserved == 0

    # Case 4: Full integration into business metrics
    biz = calculate_business_metrics(
        expected_customers=60,
        inventory_capacity=40,
        aov=100.0,
    )
    assert biz["expected_customers"] == 60
    assert biz["effective_customers"] == 40
    assert biz["unserved_customers"] == 20
    assert biz["stockout_loss_revenue_inr"] == 2000.0  # 20 unserved * ₹100 AOV
    assert biz["expected_revenue_inr"] == 4000.0  # 40 served * ₹100 AOV


def test_category_specific_costs_and_spoilage():
    """Verify category-specific cost profiles, spoilage rates, and gross margins."""
    # Fruits: higher COGS rate (40%) and 8% perishable spoilage
    fruits_biz = calculate_business_metrics(
        expected_customers=30,
        vendor_category="fruits",
    )
    assert fruits_biz["average_spend_inr"] == 80.0
    assert fruits_biz["variable_cost_rate"] == 0.40
    assert fruits_biz["spoilage_rate"] == 0.08
    assert fruits_biz["expected_revenue_inr"] == 2400.0  # 30 * 80
    assert fruits_biz["ingredient_cost_inr"] == 960.0   # 2400 * 0.40
    assert fruits_biz["spoilage_cost_inr"] == 76.80     # 960 * 0.08
    assert fruits_biz["total_stock_cost_inr"] == 1036.80

    # Repair: lower COGS rate (24%) and 0% spoilage
    repair_biz = calculate_business_metrics(
        expected_customers=5,
        vendor_category="repair",
    )
    assert repair_biz["average_spend_inr"] == 180.0
    assert repair_biz["variable_cost_rate"] == 0.24
    assert repair_biz["spoilage_rate"] == 0.00
    assert repair_biz["expected_revenue_inr"] == 900.0   # 5 * 180
    assert repair_biz["ingredient_cost_inr"] == 216.0   # 900 * 0.24
    assert repair_biz["spoilage_cost_inr"] == 0.0
    assert repair_biz["total_stock_cost_inr"] == 216.0


def test_operating_expenses_and_fuel_cost():
    """Verify hourly fixed overhead amortization and congestion-adjusted fuel calculations."""
    from api.business_economics import calculate_fuel_cost, calculate_hourly_fixed_cost

    # Fixed overhead amortization
    assert calculate_hourly_fixed_cost(650.0, 13.0) == 50.0
    assert calculate_hourly_fixed_cost(480.0, 12.0) == 40.0

    # Fuel cost without traffic (congestion = 0.0)
    # (18 km/L @ 95 INR/L = 5.2778 INR/km) * 3 km = 15.83 INR
    base_fuel = calculate_fuel_cost(distance_km=3.0, fuel_price_inr_per_l=95.0, fuel_efficiency_km_per_l=18.0)
    assert abs(base_fuel - 15.83) < 0.05

    # Fuel cost with heavy congestion (congestion = 1.0 => 1.35x penalty)
    congested_fuel = calculate_fuel_cost(distance_km=3.0, congestion_ratio=1.0)
    assert congested_fuel > base_fuel
    assert abs(congested_fuel - round(base_fuel * 1.35, 2)) < 0.10


def test_travel_opportunity_time_cost():
    """Verify opportunity cost of lost trading window during relocation."""
    from api.business_economics import calculate_travel_opportunity_cost

    # 0 min drive: full trading hour available
    t_frac, time_cost = calculate_travel_opportunity_cost(candidate_predicted_profit=2000.0, travel_time_minutes=0.0)
    assert t_frac == 1.0
    assert time_cost == 0.0

    # 15 min drive: 75% hour available, 25% lost
    t_frac, time_cost = calculate_travel_opportunity_cost(candidate_predicted_profit=2000.0, travel_time_minutes=15.0)
    assert t_frac == 0.75
    assert time_cost == 500.0

    # 30 min drive: 50% hour available, 50% lost
    t_frac, time_cost = calculate_travel_opportunity_cost(candidate_predicted_profit=2000.0, travel_time_minutes=30.0)
    assert t_frac == 0.50
    assert time_cost == 1000.0


def test_relocation_decision_policies():
    """Verify strategic decision rules: RECOMMENDED_MOVE, CONSIDER_MOVE, and STAY_PUT."""
    from api.business_economics import evaluate_relocation_decision

    # High net gain (+₹300/hr), close distance (1.5km, 8m) -> RECOMMENDED_MOVE
    dec, uplift, reason = evaluate_relocation_decision(
        candidate_realized_profit=1800.0, current_profit=1500.0, travel_time_minutes=8.0, distance_km=1.5
    )
    assert dec == "RECOMMENDED_MOVE"
    assert uplift == 300.0

    # High net gain (+₹400/hr), but long drive (4.5km, 18m) -> CONSIDER_MOVE
    dec, uplift, reason = evaluate_relocation_decision(
        candidate_realized_profit=1900.0, current_profit=1500.0, travel_time_minutes=18.0, distance_km=4.5
    )
    assert dec == "CONSIDER_MOVE"
    assert uplift == 400.0

    # Moderate gain (+₹25/hr) -> CONSIDER_MOVE
    dec, uplift, reason = evaluate_relocation_decision(
        candidate_realized_profit=1525.0, current_profit=1500.0, travel_time_minutes=5.0, distance_km=1.0
    )
    assert dec == "CONSIDER_MOVE"
    assert uplift == 25.0

    # Negative uplift (-₹150/hr) -> STAY_PUT
    dec, uplift, reason = evaluate_relocation_decision(
        candidate_realized_profit=1350.0, current_profit=1500.0, travel_time_minutes=12.0, distance_km=2.5
    )
    assert dec == "STAY_PUT"
    assert uplift == -150.0

    # Current location anchor -> STAY_PUT
    dec, uplift, reason = evaluate_relocation_decision(
        candidate_realized_profit=1500.0, current_profit=1500.0, is_current_cell=True
    )
    assert dec == "STAY_PUT"
    assert uplift == 0.0


# ---------------------------------------------------------------------------
# Manual Concrete Arithmetic Verification Examples
# ---------------------------------------------------------------------------

def test_manual_example_1_food_vendor_move():
    """
    Manual Example 1: Street Food Vendor relocation to commercial hotspot.
    Assumptions:
      - Category: food
      - Expected Demand: 45 customers
      - AOV: ₹100.00
      - COGS Rate: 0.36
      - Spoilage Rate: 0.04
      - Daily Fixed: ₹520, Operating Hours: 13 hr => Hourly Fixed: ₹40.00
      - Travel Distance: 1.5 km, Travel Time: 8 min
      - Fuel Cost: ₹8.00
      - Baseline Current Profit: ₹1,800.00

    Expected Step-by-Step Math:
      - Revenue = 45 * 100 = 4,500.00
      - Ingredient Cost = 4,500 * 0.36 = 1,620.00
      - Spoilage Cost = 1,620 * 0.04 = 64.80
      - Total Stock Cost = 1,620 + 64.80 = 1,684.80
      - Base Operating Cost = 1,620 + 40 = 1,660.00
      - Candidate Predicted Profit = 4,500 - 1,660 = 2,840.00
      - Travel Fraction = (60 - 8) / 60 = 52 / 60 = 0.8667
      - Realized Next-Hour Profit = 2,840 * 0.8667 - 8 = 2,461.43 - 8 = 2,453.43
      - Net Realized Uplift = 2,453.43 - 1,800.00 = +653.43
      - Decision = RECOMMENDED_MOVE
    """
    biz = calculate_business_metrics(
        expected_customers=45,
        aov=100.0,
        variable_cost_rate=0.36,
        spoilage_rate=0.04,
        fixed_cost_per_day=520.0,
        hours_per_day=13.0,
        distance_km=1.5,
        travel_time_minutes=8.0,
        fuel_cost=8.0,
        current_profit=1800.0,
        vendor_category="food",
    )

    assert biz["expected_revenue_inr"] == 4500.0
    assert biz["ingredient_cost_inr"] == 1620.0
    assert biz["spoilage_cost_inr"] == 64.80
    assert biz["total_stock_cost_inr"] == 1684.80
    assert biz["fixed_cost_per_hour_inr"] == 40.0
    assert biz["base_operating_cost_inr"] == 1660.0
    assert biz["candidate_predicted_profit_inr"] == 2840.0
    assert biz["travel_fraction"] == 0.8667
    assert abs(biz["relocation_adjusted_realized_profit_inr"] - 2453.43) < 0.20
    assert biz["relocation_decision"] == "RECOMMENDED_MOVE"
    assert biz["realized_net_uplift_inr"] > 600.0


def test_manual_example_2_fruit_vendor_inventory_stockout():
    """
    Manual Example 2: Fruit Stall with Inventory Capacity Bottleneck.
    Assumptions:
      - Category: fruits
      - Expected Demand: 55 customers
      - Inventory Capacity: 35 customers (can only serve 35)
      - AOV: ₹80.00
      - COGS Rate: 0.40
      - Spoilage Rate: 0.08
      - Daily Fixed: ₹390, Operating Hours: 13 hr => Hourly Fixed: ₹30.00
      - Stationary (Distance: 0, Travel: 0)

    Expected Step-by-Step Math:
      - Effective Served = min(55, 35) = 35
      - Unserved Stockout = 55 - 35 = 20
      - Stockout Revenue Loss = 20 * 80 = 1,600.00
      - Revenue = 35 * 80 = 2,800.00
      - Ingredient Cost = 2,800 * 0.40 = 1,120.00
      - Spoilage Cost = 1,120 * 0.08 = 89.60
      - Total Stock Cost = 1,120 + 89.60 = 1,209.60
      - Base Operating Cost = 1,120 + 30 = 1,150.00
      - Candidate Predicted Profit = 2,800 - 1,150 = 1,650.00
      - Realized Profit = 1,650.00 (since distance=0, travel_fraction=1.0)
    """
    biz = calculate_business_metrics(
        expected_customers=55,
        inventory_capacity=35,
        aov=80.0,
        variable_cost_rate=0.40,
        spoilage_rate=0.08,
        fixed_cost_per_day=390.0,
        hours_per_day=13.0,
        is_current_cell=True,
        vendor_category="fruits",
    )

    assert biz["expected_customers"] == 55
    assert biz["inventory_capacity"] == 35
    assert biz["effective_customers"] == 35
    assert biz["unserved_customers"] == 20
    assert biz["stockout_loss_revenue_inr"] == 1600.0
    assert biz["expected_revenue_inr"] == 2800.0
    assert biz["ingredient_cost_inr"] == 1120.0
    assert biz["spoilage_cost_inr"] == 89.60
    assert biz["fixed_cost_per_hour_inr"] == 30.0
    assert biz["candidate_predicted_profit_inr"] == 1650.0
    assert biz["relocation_adjusted_realized_profit_inr"] == 1650.0
    assert biz["relocation_decision"] == "STAY_PUT"


def test_manual_example_3_beverage_stall_staying_put():
    """
    Manual Example 3: Tea & Coffee Stall staying put at origin hub.
    Assumptions:
      - Category: beverages
      - Expected Demand: 75 customers
      - AOV: ₹40.00
      - COGS Rate: 0.30, Spoilage: 0.02
      - Daily Fixed: ₹420, Hours: 12 hr => Hourly Fixed: ₹35.00
      - Origin anchor: is_current_cell = True

    Expected Step-by-Step Math:
      - Revenue = 75 * 40 = 3,000.00
      - Ingredient Cost = 3,000 * 0.30 = 900.00
      - Fixed Opex = 35.00
      - Candidate Predicted Profit = 3,000 - 935 = 2,065.00
      - Travel Fraction = 1.0, Fuel Cost = 0.0
      - Realized Profit = 2,065.00
      - Decision = STAY_PUT
    """
    biz = calculate_business_metrics(
        expected_customers=75,
        aov=40.0,
        variable_cost_rate=0.30,
        spoilage_rate=0.02,
        fixed_cost_per_day=420.0,
        hours_per_day=12.0,
        is_current_cell=True,
        vendor_category="beverages",
    )

    assert biz["expected_revenue_inr"] == 3000.0
    assert biz["ingredient_cost_inr"] == 900.0
    assert biz["fixed_cost_per_hour_inr"] == 35.0
    assert biz["candidate_predicted_profit_inr"] == 2065.0
    assert biz["relocation_adjusted_realized_profit_inr"] == 2065.0
    assert biz["fuel_cost_inr"] == 0.0
    assert biz["travel_time_minutes"] == 0.0
    assert biz["travel_fraction"] == 1.0
    assert biz["relocation_decision"] == "STAY_PUT"


def test_manual_example_4_repair_van_traffic_delay():
    """
    Manual Example 4: Mobile Repair Van with Heavy Traffic Congestion Delay.
    Assumptions:
      - Category: repair
      - Expected Demand: 4 customers
      - AOV: ₹180.00
      - COGS Rate: 0.24, Spoilage: 0.00
      - Daily Fixed: ₹600, Hours: 12 hr => Hourly Fixed: ₹50.00
      - Distance: 3.5 km, Congestion Ratio: 0.80
      - Travel Time: 18 min
      - Current Baseline Profit: ₹420.00

    Expected Step-by-Step Math:
      - Base Fuel = (3.5 / 18.0) * 95 = 18.472
      - Congestion Penalty = 1.0 + (0.80 * 0.35) = 1.28
      - Fuel Cost = 18.472 * 1.28 = 23.64
      - Revenue = 4 * 180 = 720.00
      - Ingredient Cost = 720 * 0.24 = 172.80
      - Hourly Fixed = 50.00
      - Candidate Predicted Profit = 720 - (172.80 + 50.00) = 497.20
      - Travel Fraction = (60 - 18) / 60 = 0.70
      - Opportunity Time Cost = 497.20 * 0.30 = 149.16
      - Realized Next-Hour Profit = 497.20 * 0.70 - 23.64 = 348.04 - 23.64 = 324.40
      - Net Uplift vs Current (420) = 324.40 - 420.00 = -95.60
      - Decision = STAY_PUT (traffic delay & fuel eliminate relocation benefit)
    """
    from api.business_economics import calculate_fuel_cost

    fuel = calculate_fuel_cost(distance_km=3.5, congestion_ratio=0.80)
    assert abs(fuel - 23.64) < 0.10

    biz = calculate_business_metrics(
        expected_customers=4,
        aov=180.0,
        variable_cost_rate=0.24,
        spoilage_rate=0.00,
        fixed_cost_per_day=600.0,
        hours_per_day=12.0,
        distance_km=3.5,
        travel_time_minutes=18.0,
        fuel_cost=fuel,
        current_profit=420.0,
        vendor_category="repair",
    )

    assert biz["expected_revenue_inr"] == 720.0
    assert biz["ingredient_cost_inr"] == 172.80
    assert biz["fixed_cost_per_hour_inr"] == 50.0
    assert biz["candidate_predicted_profit_inr"] == 497.20
    assert biz["travel_fraction"] == 0.70
    assert abs(biz["time_cost_inr"] - 149.16) < 0.10
    assert abs(biz["relocation_adjusted_realized_profit_inr"] - 324.40) < 0.10
    assert biz["relocation_decision"] == "STAY_PUT"
    assert biz["realized_net_uplift_inr"] < 0.0



