"""
Vendor Profile Generator.

Runs BEFORE the Market Simulation Engine. Assigns each vendor a full
business profile — not just a category — so that two vendors in the
identical H3 cell at the identical hour produce DIFFERENT simulated
outcomes, driven by their own menu, pricing, capacity, and service speed.
This is what makes the dataset behave like real heterogeneous vendors
rather than "one demand curve per category."

Each vendor gets:
  - category (food/fruits/salon/repair)
  - product menu with per-item price and weather/time affinity
  - operating hours
  - inventory capacity (caps how many customers can be served per hour)
  - prep_speed_factor (affects effective capacity)
  - avg_service_time_min
  - fixed_daily_cost, variable_cost_pct (feeds profit derivation)
  - hidden_skill_factor (see market_simulator.py — never exposed to the model)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Category -> product menu. Each item has a base price and affinity
# multipliers so product-mix simulation responds to real conditions
# (this is what lets "rainy evening -> tea demand up" actually hold).
CATEGORY_MENUS = {
    "food": [
        {"item": "Tea",     "price": 15, "rain_affinity": 1.6, "cold_affinity": 1.4, "lunch_affinity": 0.6},
        {"item": "Coffee",  "price": 20, "rain_affinity": 1.3, "cold_affinity": 1.3, "lunch_affinity": 0.6},
        {"item": "Samosa",  "price": 20, "rain_affinity": 1.2, "cold_affinity": 1.0, "lunch_affinity": 1.1},
        {"item": "Meals",   "price": 90, "rain_affinity": 0.9, "cold_affinity": 1.0, "lunch_affinity": 1.8},
        {"item": "Bajji",   "price": 25, "rain_affinity": 1.4, "cold_affinity": 1.1, "lunch_affinity": 0.9},
        {"item": "Water",   "price": 20, "rain_affinity": 0.5, "cold_affinity": 0.6, "lunch_affinity": 1.0},
    ],
    "fruits": [
        {"item": "Mixed Fruit Cup", "price": 40, "rain_affinity": 0.7, "cold_affinity": 0.8, "lunch_affinity": 1.0},
        {"item": "Juice",           "price": 35, "rain_affinity": 0.6, "cold_affinity": 0.7, "lunch_affinity": 1.1},
        {"item": "Whole Fruit",     "price": 25, "rain_affinity": 0.9, "cold_affinity": 0.9, "lunch_affinity": 1.0},
    ],
    "salon": [
        {"item": "Haircut",     "price": 100, "rain_affinity": 0.8, "cold_affinity": 1.0, "lunch_affinity": 1.0},
        {"item": "Shave",       "price": 50,  "rain_affinity": 0.8, "cold_affinity": 1.0, "lunch_affinity": 1.0},
        {"item": "Face Clean",  "price": 150, "rain_affinity": 0.9, "cold_affinity": 1.0, "lunch_affinity": 1.0},
    ],
    "repair": [
        {"item": "Puncture Fix",   "price": 60,  "rain_affinity": 1.3, "cold_affinity": 1.0, "lunch_affinity": 1.0},
        {"item": "General Repair", "price": 150, "rain_affinity": 1.0, "cold_affinity": 1.0, "lunch_affinity": 1.0},
    ],
}

# Rough capacity/speed ranges per category (customers/hour a single vendor can realistically serve)
CATEGORY_CAPACITY_RANGE = {
    "food": (25, 70),
    "fruits": (15, 40),
    "salon": (3, 8),
    "repair": (2, 6),
}

CATEGORY_SERVICE_TIME_MIN = {
    "food": (1.5, 4.0),
    "fruits": (1.0, 3.0),
    "salon": (10.0, 25.0),
    "repair": (15.0, 40.0),
}

CATEGORY_FIXED_DAILY_COST = {
    "food": (150, 400),      # fuel, gas, ice
    "fruits": (80, 200),
    "salon": (50, 150),
    "repair": (60, 180),
}

CATEGORY_VARIABLE_COST_PCT = {
    # % of revenue that's cost of goods (ingredients, parts)
    "food": (0.35, 0.50),
    "fruits": (0.40, 0.55),
    "salon": (0.10, 0.20),
    "repair": (0.20, 0.35),
}


@dataclass
class VendorProfile:
    vendor_id: str
    category: str
    home_cell: str
    menu: list
    max_capacity_per_hour: int
    prep_speed_factor: float       # 0.7 (slow) to 1.3 (fast) — modulates effective capacity
    avg_service_time_min: float
    fixed_daily_cost: float
    variable_cost_pct: float
    hidden_skill_factor: float     # reputation/quality — NOT a model feature, see market_simulator.py


def generate_vendor_profiles(
    grid: pd.DataFrame,
    n_vendors: int = 150,
    seed: int = 2026,
) -> list[VendorProfile]:
    rng = np.random.default_rng(seed)
    categories = rng.choice(
        list(CATEGORY_MENUS.keys()), size=n_vendors, p=[0.45, 0.25, 0.15, 0.15]
    )
    home_cells = rng.choice(grid["h3_cell"].to_numpy(), size=n_vendors)

    profiles = []
    for i in range(n_vendors):
        cat = categories[i]
        cap_lo, cap_hi = CATEGORY_CAPACITY_RANGE[cat]
        svc_lo, svc_hi = CATEGORY_SERVICE_TIME_MIN[cat]
        fixed_lo, fixed_hi = CATEGORY_FIXED_DAILY_COST[cat]
        var_lo, var_hi = CATEGORY_VARIABLE_COST_PCT[cat]

        profiles.append(VendorProfile(
            vendor_id=f"V{i:04d}",
            category=cat,
            home_cell=home_cells[i],
            menu=CATEGORY_MENUS[cat],
            max_capacity_per_hour=int(rng.integers(cap_lo, cap_hi + 1)),
            prep_speed_factor=round(float(rng.uniform(0.7, 1.3)), 2),
            avg_service_time_min=round(float(rng.uniform(svc_lo, svc_hi)), 1),
            fixed_daily_cost=round(float(rng.uniform(fixed_lo, fixed_hi)), 2),
            variable_cost_pct=round(float(rng.uniform(var_lo, var_hi)), 3),
            hidden_skill_factor=float(np.clip(rng.normal(1.0, 0.18), 0.5, 1.6)),
        ))
    return profiles


def profiles_to_dataframe(profiles: list[VendorProfile]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        rows.append({
            "vendor_id": p.vendor_id,
            "category": p.category,
            "home_cell": p.home_cell,
            "max_capacity_per_hour": p.max_capacity_per_hour,
            "prep_speed_factor": p.prep_speed_factor,
            "avg_service_time_min": p.avg_service_time_min,
            "fixed_daily_cost": p.fixed_daily_cost,
            "variable_cost_pct": p.variable_cost_pct,
            "hidden_skill_factor": p.hidden_skill_factor,  # kept for audit; excluded from model features downstream
        })
    return pd.DataFrame(rows)


def simulate_product_mix(profile: VendorProfile, is_raining: bool, temp_c: float, hour: int, rng: np.random.Generator) -> dict:
    """
    Given real conditions, weight the vendor's menu items by their
    affinity to those conditions and return a probability distribution
    over items — this is the mechanism that makes "rain -> tea demand up"
    and "lunch -> meals demand up" actually hold in the generated data.
    """
    is_lunch = 12 <= hour <= 14
    is_cold = temp_c < 24

    weights = []
    for item in profile.menu:
        w = 1.0
        if is_raining:
            w *= item["rain_affinity"]
        if is_cold:
            w *= item["cold_affinity"]
        if is_lunch:
            w *= item["lunch_affinity"]
        weights.append(w)

    weights = np.array(weights)
    weights = weights / weights.sum()
    return {item["item"]: round(float(w), 3) for item, w in zip(profile.menu, weights)}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from h3_grid import generate_city_grid

    grid = generate_city_grid(16.5062, 80.6480, 8.0)
    profiles = generate_vendor_profiles(grid, n_vendors=150)
    df = profiles_to_dataframe(profiles)
    print(df.head(10))
    print(f"\n{len(profiles)} vendor profiles generated")
    print(df.groupby("category")[["max_capacity_per_hour", "fixed_daily_cost"]].mean())

    # quick product-mix sanity check
    rng = np.random.default_rng(1)
    food_vendor = next(p for p in profiles if p.category == "food")
    print(f"\nProduct mix, food vendor, RAINY evening (7pm, 20C):")
    print(simulate_product_mix(food_vendor, is_raining=True, temp_c=20, hour=19, rng=rng))
    print(f"\nProduct mix, same vendor, SUNNY lunch (1pm, 30C):")
    print(simulate_product_mix(food_vendor, is_raining=False, temp_c=30, hour=13, rng=rng))
