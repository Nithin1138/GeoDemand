"""
Dataset 9 — Market Simulation Output (historical_transactions.parquet)

This is where business outcomes are generated — the ONLY dataset in the
whole pipeline where the target variable comes from. Everything upstream
(h3_cells, static_features, weather_hourly, calendar, events, competition,
vendor_profiles, products) is real or hybrid-real and is CONSUMED here,
not regenerated. This module joins them together and applies the demand
model.

Anti-circularity design (unchanged from v1, see market_simulator.py for
full rationale): hidden per-vendor/per-cell latent factors (popularity_
score, quality_score, hidden_skill_factor — sourced from vendor_profiles.
parquet's simulation-only columns) drive real variance but are excluded
from the model's feature set at feature_store build time. Interaction
terms (rain x office density, weekend x mall density) and heteroscedastic
noise remain.

Multiplier decomposition (new in this version, per pipeline spec): rather
than one opaque expected_customer_count number, every transaction records
the individual weather_multiplier, holiday_multiplier, competition_
multiplier, event_multiplier, and noise_component that produced it — this
makes the simulation auditable row-by-row, not just in aggregate via the
qualitative validation suite.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from vendor_profile import simulate_product_mix, VendorProfile  # noqa: E402
from market_simulator import CATEGORY_BASE_DEMAND, CATEGORY_POI_AFFINITY  # noqa: E402

POI_NORM_CONST = {
    "office_count": 40, "college_count": 3, "restaurant_count": 25,
    "population_density": 25000, "hospital_count": 4, "park_count": 5,
    "mall_count": 3, "road_density": 1.0,
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _row_to_vendor_profile(row) -> VendorProfile:
    from vendor_profile import CATEGORY_MENUS
    return VendorProfile(
        vendor_id=row["vendor_id"], category=row["vendor_category"], home_cell=row["home_cell_id"],
        menu=CATEGORY_MENUS[row["vendor_category"]], max_capacity_per_hour=999,
        prep_speed_factor=1.0, avg_service_time_min=row["service_time"],
        fixed_daily_cost=row["fixed_cost_per_day"], variable_cost_pct=row["variable_cost_rate"],
        hidden_skill_factor=row["hidden_skill_factor"],
    )


def generate_historical_transactions(
    h3_cells: pd.DataFrame,
    static_features: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    calendar: pd.DataFrame,
    events: pd.DataFrame,
    competition: pd.DataFrame,
    vendor_profiles: pd.DataFrame,
    hours_per_day: tuple = (8, 21),
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Simulates hourly vendor transactions under a true NEXT-HOUR forecasting contract:
      - decision_timestamp (t): when the vendor requests the recommendation / decision.
      - target_timestamp (t + 1h): the start of the 1-hour operating window being predicted.
      - expected_customer_count: demand realized during [t + 1h, t + 2h).
    """
    rng = np.random.default_rng(seed)

    static_lookup = static_features.set_index("h3_cell_id")
    weather_lookup = weather_hourly.set_index("timestamp")
    calendar_lookup = calendar.set_index("timestamp")
    competition_lookup = competition.set_index(["date", "h3_cell_id"])
    events_lookup = events.groupby(
        [events["timestamp"].dt.floor("D"), "h3_cell_id"]
    )["event_importance"].max()

    # Hidden per-cell latent factor, independent of vendor-level hidden factors —
    # never exposed to the model, mirrors "local reputation / visibility".
    cell_reputation = pd.Series(
        rng.normal(1.0, 0.15, size=len(h3_cells)).clip(0.6, 1.5),
        index=h3_cells["h3_cell_id"],
    )

    start_h, end_h = hours_per_day
    # Decision hours: e.g. from 8:00 (for 9:00 target) to 20:00 (for 21:00 target)
    hours = list(range(start_h, end_h))
    dates = calendar["timestamp"].dt.floor("D").unique()

    rows = []
    for _, vrow in vendor_profiles.iterrows():
        cell = vrow["home_cell_id"]
        cat = vrow["vendor_category"]
        static = static_lookup.loc[cell]
        reputation = cell_reputation.loc[cell]
        affinity = CATEGORY_POI_AFFINITY[cat]
        vendor_profile_obj = _row_to_vendor_profile(vrow)

        poi_score = sum(
            w * (static[feat] / POI_NORM_CONST[feat]) for feat, w in affinity.items()
        )

        for d in dates:
            d_ts = pd.Timestamp(d)
            comp_row = competition_lookup.loc[(d_ts, cell)] if (d_ts, cell) in competition_lookup.index else None
            competition_score = comp_row["competition_score"] if comp_row is not None else 0.0
            event_importance = events_lookup.get((d_ts, cell), 0.0)

            for h in hours:
                # Decision time t (features known as of t)
                decision_ts = d_ts + pd.Timedelta(hours=h)
                # Target window t+1 (operating window [t+1, t+2))
                target_ts = decision_ts + pd.Timedelta(hours=1)
                h_target = target_ts.hour

                weather_row = weather_lookup.loc[decision_ts] if decision_ts in weather_lookup.index else None
                if isinstance(weather_row, pd.DataFrame):
                    weather_row = weather_row.iloc[0]
                cal_row = calendar_lookup.loc[decision_ts] if decision_ts in calendar_lookup.index else None
                if isinstance(cal_row, pd.DataFrame):
                    cal_row = cal_row.iloc[0]

                is_raining = bool(weather_row["weather_condition"] == "rain") if weather_row is not None else False
                temp_c = float(weather_row["temperature"]) if weather_row is not None else 27.0
                is_holiday = bool(cal_row["is_holiday"]) if cal_row is not None else False
                is_weekend = bool(cal_row["is_weekend"]) if cal_row is not None else False

                # --- base + hour curve for the target window ---
                base = CATEGORY_BASE_DEMAND[cat]
                hour_multiplier = round(float(
                    1.0 + 0.9 * np.exp(-((h_target - 13) ** 2) / 6) + 1.1 * np.exp(-((h_target - 19) ** 2) / 4)
                ), 4)

                # --- weather multiplier: category + target hour aware ---
                if is_raining:
                    if cat == "food" and 17 <= h_target <= 21:
                        weather_multiplier = 1.30 + 0.01 * (temp_c - 27)
                    else:
                        weather_multiplier = 0.65 + 0.01 * (temp_c - 27)
                else:
                    weather_multiplier = 1.0 + 0.01 * (temp_c - 27)
                weather_multiplier = round(float(max(0.3, weather_multiplier)), 4)

                # --- holiday multiplier ---
                holiday_multiplier = round(1.25 if is_holiday else (1.15 if is_weekend else 1.0), 4)

                # --- competition multiplier: more same-cell-area competitors -> demand split ---
                competition_multiplier = round(float(max(0.5, 1.0 - 0.4 * competition_score)), 4)

                # --- event multiplier ---
                event_multiplier = round(float(1.0 + 0.5 * event_importance), 4)

                # --- interaction terms (anti-circularity design) ---
                rain_office_interaction = 0.15 * float(is_raining) * (static["office_count"] / 40)
                weekend_mall_interaction = 0.12 * float(is_weekend) * (static["mall_count"] / 3)

                # --- hidden effects, never exposed to the model ---
                hidden_effect = (
                    0.3 * (reputation - 1.0)
                    + 0.3 * (vrow["hidden_skill_factor"] - 1.0)
                    + 0.15 * (vrow["popularity_score"] - 0.5)
                    + 0.10 * (vrow["quality_score"] - 0.6)
                )

                linear_score = (
                    base + poi_score + rain_office_interaction + weekend_mall_interaction + hidden_effect
                )
                base_demand = linear_score * hour_multiplier

                combined = (
                    base_demand * weather_multiplier * holiday_multiplier
                    * competition_multiplier * event_multiplier
                )
                expected = 80.0 * _sigmoid((combined - 3.0) * 1.0)

                noise_std = 0.15 * np.sqrt(max(expected, 0.1)) * 1.8
                noise_component = round(float(rng.normal(0, noise_std)), 3)
                customer_count = max(0, int(round(expected + noise_component)))

                # --- product mix + revenue/cost derivation ---
                mix = simulate_product_mix(vendor_profile_obj, is_raining, temp_c, h_target, rng)
                top_product = max(mix, key=mix.get) if mix else None
                aov = vrow["average_order_value"]
                revenue = round(customer_count * aov, 2)
                ingredient_cost = round(revenue * vrow["variable_cost_rate"], 2)
                fuel_cost = round(vrow["fixed_cost_per_day"] / len(hours), 2)
                operating_cost = round(ingredient_cost + fuel_cost, 2)
                profit = round(revenue - operating_cost, 2)

                rows.append({
                    "transaction_id": str(uuid.uuid4())[:10],
                    "decision_timestamp": decision_ts,
                    "target_timestamp": target_ts,
                    "vendor_id": vrow["vendor_id"],
                    "h3_cell_id": cell,
                    "expected_customer_count": customer_count,
                    "average_order_value": aov,
                    "revenue": revenue,
                    "ingredient_cost": ingredient_cost,
                    "fuel_cost": fuel_cost,
                    "operating_cost": operating_cost,
                    "profit": profit,
                    "top_selling_product": top_product,
                    "weather_multiplier": weather_multiplier,
                    "holiday_multiplier": holiday_multiplier,
                    "competition_multiplier": competition_multiplier,
                    "event_multiplier": event_multiplier,
                    "noise_component": noise_component,
                    "source": "simulated:market_outcomes",
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
    from h3_cells import build_h3_cells
    from static_features import build_static_features
    from weather import build_weather_hourly
    from calendar_dataset import build_calendar
    from events import build_events
    from competition import build_competition
    from vendor_profiles import build_vendor_profiles

    print("Building upstream datasets (smoke test: 20 vendors, 14 days)...")
    cells = build_h3_cells()
    static = build_static_features(cells)
    weather = build_weather_hourly(16.5062, 80.6480, "2025-01-01", "2025-01-14")
    cal = build_calendar("2025-01-01", "2025-01-14")
    vendors = build_vendor_profiles(cells["h3_cell_id"].tolist(), n_vendors=20)
    events_df = build_events(cells["h3_cell_id"].tolist(), "2025-01-01", "2025-01-14")
    comp = build_competition(cells["h3_cell_id"].tolist(), vendors, pd.date_range("2025-01-01", "2025-01-14"))

    t0 = time.time()
    tx = generate_historical_transactions(cells, static, weather, cal, events_df, comp, vendors)
    print(f"Generated {len(tx):,} transactions in {time.time()-t0:.1f}s")
    print(tx.head())
    print(f"\nexpected_customer_count stats:\n{tx['expected_customer_count'].describe()}")
    print(f"\nprofit stats:\n{tx['profit'].describe()}")
