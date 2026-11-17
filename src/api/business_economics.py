"""
GeoDemand AI — Business Economics & Financial Conversion Engine

Converts raw ML demand predictions into commercial unit economics:
  - Average Order Value (AOV) & Average Selling Price (ASP)
  - Revenue calculation with inventory capacity limits
  - Cost of Goods Sold (COGS / Stock cost) & perishable spoilage
  - Hourly fixed operating expenses (cart rent, license, utilities)
  - Travel fuel friction with traffic congestion penalty
  - Time opportunity cost during transit (reduced trading window)
  - Stationary predicted profit vs relocation-adjusted realized profit
  - Decision intelligence policy (RECOMMENDED_MOVE, CONSIDER_MOVE, STAY_PUT)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config.product_pricing import get_category_pricing
from config.settings import (
    CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR,
    DEFAULT_AOV_INR,
    DEFAULT_FIXED_COST_PER_DAY_INR,
    DEFAULT_OPERATING_HOURS_PER_DAY,
    DEFAULT_VARIABLE_COST_RATE,
    FUEL_EFFICIENCY_KM_PER_L,
    FUEL_PRICE_INR_PER_LITRE,
    MAX_RECOMMENDED_TRAVEL_KM,
    MAX_RECOMMENDED_TRAVEL_MIN,
    STAY_THRESHOLD_INR_PER_HOUR,
)


@dataclass
class BusinessMetrics:
    """Comprehensive financial breakdown for an hourly location candidate."""
    # Footfall & Capacity
    expected_customers: int
    inventory_capacity: Optional[int]
    effective_customers: int
    unserved_customers: int
    stockout_loss_revenue_inr: float

    # Commercial Pricing
    average_order_value_inr: float
    average_spend_inr: float
    expected_revenue_inr: float

    # Direct Product & Inventory Costs
    variable_cost_rate: float
    ingredient_cost_inr: float
    spoilage_rate: float
    spoilage_cost_inr: float
    total_stock_cost_inr: float

    # Operating Overhead
    fixed_cost_per_day_inr: float
    operating_hours_per_day: float
    fixed_cost_per_hour_inr: float

    # Travel & Time Friction
    distance_km: float
    fuel_cost_inr: float
    travel_time_minutes: float
    travel_fraction: float
    time_cost_inr: float

    # Aggregated Costs & Profits
    base_operating_cost_inr: float
    expected_costs_inr: float
    expected_operating_cost_inr: float
    expected_profit_inr: float
    candidate_predicted_profit_inr: float
    relocation_adjusted_realized_profit_inr: float
    realized_next_hour_profit_inr: float

    # Decision Policy (when comparing to origin)
    current_profit_inr: Optional[float] = None
    realized_net_uplift_inr: Optional[float] = None
    relocation_decision: Optional[str] = None
    decision_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Individual Economic Step Functions
# ---------------------------------------------------------------------------

def calculate_effective_demand(
    expected_customers: int,
    inventory_capacity: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Apply vendor physical inventory / throughput capacity limit to raw demand.
    Returns: (effective_customers_served, unserved_customers_stockout).
    """
    raw = max(0, int(expected_customers))
    if inventory_capacity is not None and inventory_capacity > 0:
        cap = int(inventory_capacity)
        effective = min(raw, cap)
        unserved = max(0, raw - cap)
        return effective, unserved
    return raw, 0


def calculate_revenue(
    customers: int,
    aov: float,
) -> float:
    """Expected Revenue = Effective Customers * Average Order Value."""
    return round(max(0, int(customers)) * max(0.0, float(aov)), 2)


def calculate_stock_cost(
    revenue: float,
    variable_cost_rate: float,
    spoilage_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Calculate Cost of Goods Sold (COGS) and perishable spoilage.
    Returns: (ingredient_cost, spoilage_cost, total_stock_cost).
    """
    rev = max(0.0, float(revenue))
    cogs_rate = max(0.0, float(variable_cost_rate))
    spoil_rate = max(0.0, float(spoilage_rate))

    ingredient_cost = round(rev * cogs_rate, 2)
    spoilage_cost = round(ingredient_cost * spoil_rate, 2)
    total_stock_cost = round(ingredient_cost + spoilage_cost, 2)
    return ingredient_cost, spoilage_cost, total_stock_cost


def calculate_hourly_fixed_cost(
    fixed_cost_per_day: float,
    hours_per_day: float,
) -> float:
    """Amortize daily fixed operating overhead into an hourly cost."""
    daily = max(0.0, float(fixed_cost_per_day))
    hours = max(1.0, float(hours_per_day))
    return round(daily / hours, 2)


def calculate_fuel_cost(
    distance_km: float,
    fuel_price_inr_per_l: float = FUEL_PRICE_INR_PER_LITRE,
    fuel_efficiency_km_per_l: float = FUEL_EFFICIENCY_KM_PER_L,
    congestion_ratio: float = 0.0,
) -> float:
    """
    Calculate direct driving fuel cost with optional traffic congestion multiplier.
    """
    dist = max(0.0, float(distance_km))
    if dist <= 0.0:
        return 0.0

    efficiency = max(1.0, float(fuel_efficiency_km_per_l))
    price = max(0.0, float(fuel_price_inr_per_l))
    base_fuel = (dist / efficiency) * price

    congestion = max(0.0, float(congestion_ratio))
    congestion_penalty = 1.0 + (congestion * 0.35)
    return round(base_fuel * congestion_penalty, 2)


def calculate_travel_opportunity_cost(
    candidate_predicted_profit: float,
    travel_time_minutes: float,
) -> Tuple[float, float]:
    """
    Calculate the trading window reduction and lost opportunity profit during transit.
    Returns: (travel_fraction, opportunity_time_cost_inr).
    """
    t_min = max(0.0, float(travel_time_minutes))
    # Fraction of next hour available for active selling
    travel_fraction = round(max(0.0, (60.0 - t_min) / 60.0), 4)
    lost_fraction = 1.0 - travel_fraction
    time_cost = round(max(0.0, candidate_predicted_profit) * lost_fraction, 2)
    return travel_fraction, time_cost


def evaluate_relocation_decision(
    candidate_realized_profit: float,
    current_profit: float,
    travel_time_minutes: float = 0.0,
    distance_km: float = 0.0,
    is_current_cell: bool = False,
) -> Tuple[str, float, str]:
    """
    Determine strategic relocation recommendation based on net realized hourly uplift.
    Policies:
      - RECOMMENDED_MOVE: Net uplift >= STAY_THRESHOLD (e.g. ₹50/hr), travel <= 12m, dist <= 3km
      - CONSIDER_MOVE: Net uplift between -₹10 and +₹50/hr, or high profit exceeding travel limits
      - STAY_PUT: Relocation friction exceeds gains or candidate is origin
    Returns: (decision_code, net_uplift_inr, explanation)
    """
    if is_current_cell:
        return "STAY_PUT", 0.0, "Current baseline location — no relocation required."

    net_uplift = round(candidate_realized_profit - current_profit, 2)
    dist = max(0.0, float(distance_km))
    t_min = max(0.0, float(travel_time_minutes))

    # Condition 1: High net profit uplift within comfortable moving radius
    if net_uplift >= STAY_THRESHOLD_INR_PER_HOUR:
        if dist <= MAX_RECOMMENDED_TRAVEL_KM and t_min <= MAX_RECOMMENDED_TRAVEL_MIN:
            return (
                "RECOMMENDED_MOVE",
                net_uplift,
                f"High net profit gain of +₹{net_uplift:.0f}/hr with quick {t_min:.0f}m ({dist:.1f}km) move.",
            )
        else:
            return (
                "CONSIDER_MOVE",
                net_uplift,
                f"Substantial profit uplift (+₹{net_uplift:.0f}/hr), but travel ({dist:.1f}km, {t_min:.0f}m) requires commitment.",
            )

    # Condition 2: Moderate or borderline gain
    if -10.0 <= net_uplift < STAY_THRESHOLD_INR_PER_HOUR:
        return (
            "CONSIDER_MOVE",
            net_uplift,
            f"Marginal change ({net_uplift:+.0f} ₹/hr). Moving may be beneficial if footfall momentum increases.",
        )

    # Condition 3: Loss or negligible return after travel costs
    return (
        "STAY_PUT",
        net_uplift,
        f"Staying put is financially optimal; relocation friction causes a net loss of {net_uplift:.0f} ₹/hr.",
    )


# ---------------------------------------------------------------------------
# Master Business Economics Pipeline
# ---------------------------------------------------------------------------

def calculate_business_metrics(
    expected_customers: int,
    aov: Optional[float] = None,
    variable_cost_rate: Optional[float] = None,
    fixed_cost_per_day: Optional[float] = None,
    hours_per_day: Optional[float] = None,
    fuel_cost: Optional[float] = None,
    travel_time_minutes: Optional[float] = None,
    distance_km: Optional[float] = None,
    inventory_capacity: Optional[int] = None,
    vendor_category: Optional[str] = None,
    spoilage_rate: Optional[float] = None,
    current_profit: Optional[float] = None,
    is_current_cell: bool = False,
    congestion_ratio: float = 0.0,
) -> dict:
    """
    Master business economics conversion for GeoDemand AI.
    Converts ML footfall demand into full P&L metrics with inventory limits,
    category-specific defaults, travel friction, and relocation decision logic.
    """
    # 1. Resolve category defaults
    pricing = get_category_pricing(vendor_category)
    if vendor_category is not None:
        resolved_aov = float(aov) if aov is not None else pricing.default_aov_inr
        resolved_cogs_rate = float(variable_cost_rate) if variable_cost_rate is not None else pricing.default_cogs_rate
        resolved_spoilage = float(spoilage_rate) if spoilage_rate is not None else pricing.spoilage_rate
    else:
        resolved_aov = float(aov) if aov is not None else DEFAULT_AOV_INR
        resolved_cogs_rate = float(variable_cost_rate) if variable_cost_rate is not None else DEFAULT_VARIABLE_COST_RATE
        resolved_spoilage = float(spoilage_rate) if spoilage_rate is not None else 0.0

    resolved_fixed_daily = float(fixed_cost_per_day) if fixed_cost_per_day is not None else DEFAULT_FIXED_COST_PER_DAY_INR
    resolved_hours = float(hours_per_day) if hours_per_day and hours_per_day > 0 else DEFAULT_OPERATING_HOURS_PER_DAY
    resolved_dist = max(0.0, float(distance_km)) if distance_km is not None else 0.0
    resolved_t_min = max(0.0, float(travel_time_minutes)) if travel_time_minutes is not None else 0.0

    # 2. Compute fuel cost if not pre-calculated
    if fuel_cost is not None:
        resolved_fuel = max(0.0, float(fuel_cost))
    else:
        resolved_fuel = calculate_fuel_cost(resolved_dist, congestion_ratio=congestion_ratio)

    # 3. Capacity constraints
    effective_cust, unserved_cust = calculate_effective_demand(expected_customers, inventory_capacity)
    stockout_loss_revenue = round(unserved_cust * resolved_aov, 2)

    # 4. Revenue & Stock Costs
    revenue = calculate_revenue(effective_cust, resolved_aov)
    ingredient_cost, spoilage_cost, total_stock_cost = calculate_stock_cost(
        revenue=revenue,
        variable_cost_rate=resolved_cogs_rate,
        spoilage_rate=resolved_spoilage,
    )

    # 5. Fixed Operating Overhead
    hourly_fixed = calculate_hourly_fixed_cost(resolved_fixed_daily, resolved_hours)

    # Base operating cost (without travel fuel)
    base_operating_cost = round(ingredient_cost + hourly_fixed, 2)
    # Total operating cost (including travel fuel and spoilage)
    total_operating_cost = round(base_operating_cost + resolved_fuel, 2)

    # 6. Stationary Candidate Predicted Profit
    # Candidate predicted profit if vendor had already been situated there all hour
    candidate_predicted_profit = round(revenue - base_operating_cost, 2)
    # Net profit if stationery or full-cost deduction
    net_predicted_profit = round(revenue - total_operating_cost, 2)

    # 7. Travel Fraction & Relocation-Adjusted Realized Profit
    if is_current_cell:
        travel_fraction = 1.0
        time_cost = 0.0
        relocation_adjusted_realized_profit = candidate_predicted_profit
    else:
        travel_fraction, time_cost = calculate_travel_opportunity_cost(candidate_predicted_profit, resolved_t_min)
        relocation_adjusted_realized_profit = round(candidate_predicted_profit * travel_fraction - resolved_fuel, 2)

    # 8. Decision Evaluation (if baseline current profit provided or is origin anchor)
    decision = None
    net_uplift = None
    decision_reason = None
    if current_profit is not None or is_current_cell:
        baseline = current_profit if current_profit is not None else relocation_adjusted_realized_profit
        decision, net_uplift, decision_reason = evaluate_relocation_decision(
            candidate_realized_profit=relocation_adjusted_realized_profit,
            current_profit=baseline,
            travel_time_minutes=resolved_t_min,
            distance_km=resolved_dist,
            is_current_cell=is_current_cell,
        )

    metrics = BusinessMetrics(
        expected_customers=int(expected_customers),
        inventory_capacity=int(inventory_capacity) if inventory_capacity is not None else None,
        effective_customers=effective_cust,
        unserved_customers=unserved_cust,
        stockout_loss_revenue_inr=stockout_loss_revenue,
        average_order_value_inr=resolved_aov,
        average_spend_inr=resolved_aov,
        expected_revenue_inr=revenue,
        variable_cost_rate=resolved_cogs_rate,
        ingredient_cost_inr=ingredient_cost,
        spoilage_rate=resolved_spoilage,
        spoilage_cost_inr=spoilage_cost,
        total_stock_cost_inr=total_stock_cost,
        fixed_cost_per_day_inr=resolved_fixed_daily,
        operating_hours_per_day=resolved_hours,
        fixed_cost_per_hour_inr=hourly_fixed,
        distance_km=resolved_dist,
        fuel_cost_inr=resolved_fuel,
        travel_time_minutes=resolved_t_min,
        travel_fraction=travel_fraction,
        time_cost_inr=time_cost,
        base_operating_cost_inr=base_operating_cost,
        expected_costs_inr=total_operating_cost,
        expected_operating_cost_inr=total_operating_cost,
        expected_profit_inr=net_predicted_profit,
        candidate_predicted_profit_inr=candidate_predicted_profit,
        relocation_adjusted_realized_profit_inr=relocation_adjusted_realized_profit,
        realized_next_hour_profit_inr=relocation_adjusted_realized_profit,
        current_profit_inr=current_profit,
        realized_net_uplift_inr=net_uplift,
        relocation_decision=decision,
        decision_reason=decision_reason,
    )

    return metrics.to_dict()
