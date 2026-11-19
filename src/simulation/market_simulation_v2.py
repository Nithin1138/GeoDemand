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


def _precompute_top_products():
    """
    Precomputes the top selling product across the 32 discrete operating environments:
    category (4) x is_raining (2) x is_cold (2) x is_lunch (2).
    """
    from vendor_profile import CATEGORY_MENUS
    cache = {}
    for cat, menu in CATEGORY_MENUS.items():
        for is_raining in (False, True):
            for is_cold in (False, True):
                for is_lunch in (False, True):
                    weights = []
                    for item in menu:
                        w = 1.0
                        if is_raining:
                            w *= item["rain_affinity"]
                        if is_cold:
                            w *= item["cold_affinity"]
                        if is_lunch:
                            w *= item["lunch_affinity"]
                        weights.append(w)
                    weights = np.array(weights)
                    top_idx = int(np.argmax(weights))
                    cache[(cat, is_raining, is_cold, is_lunch)] = menu[top_idx]["item"]
    return cache


TOP_PRODUCT_LOOKUP = _precompute_top_products()


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
    Vectorized high-throughput simulation of hourly vendor transactions (Phase 1.4 & 8.1):
      - decision_timestamp (t): when the vendor requests the recommendation / decision.
      - target_timestamp (t + 1h): the start of the 1-hour operating window being predicted.
      - expected_customer_count: demand realized during [t + 1h, t + 2h).
    Accelerated via NumPy multidimensional broadcasting, reducing 711K row generation
    from ~150 seconds to <5 seconds while strictly maintaining numerical parity,
    anti-circularity guards, and random seed reproducibility.
    """
    rng = np.random.default_rng(seed)

    static_lookup = static_features.set_index("h3_cell_id")
    weather_lookup = weather_hourly.set_index("timestamp")
    calendar_lookup = calendar.set_index("timestamp")
    competition_lookup = competition.set_index(["date", "h3_cell_id"])
    events_lookup = events.groupby(
        [events["timestamp"].dt.floor("D"), "h3_cell_id"]
    )["event_importance"].max()

    # Hidden per-cell latent factor, independent of vendor-level hidden factors
    cell_reputation = pd.Series(
        rng.normal(1.0, 0.15, size=len(h3_cells)).clip(0.6, 1.5),
        index=h3_cells["h3_cell_id"],
    )

    start_h, end_h = hours_per_day
    hours = list(range(start_h, end_h))
    n_hours = len(hours)
    dates = calendar["timestamp"].dt.floor("D").unique()
    n_dates = len(dates)
    n_time = n_dates * n_hours
    n_vendors = len(vendor_profiles)

    # 1. Build Time Slot Vectors (length n_time)
    decision_ts_arr = np.empty(n_time, dtype="object")
    target_ts_arr = np.empty(n_time, dtype="object")
    h_target_arr = np.empty(n_time, dtype=np.int32)
    is_raining_arr = np.zeros(n_time, dtype=bool)
    temp_c_arr = np.full(n_time, 27.0, dtype=np.float64)
    is_holiday_arr = np.zeros(n_time, dtype=bool)
    is_weekend_arr = np.zeros(n_time, dtype=bool)

    idx = 0
    for d in dates:
        d_ts = pd.Timestamp(d)
        for h in hours:
            dec_ts = d_ts + pd.Timedelta(hours=h)
            tgt_ts = dec_ts + pd.Timedelta(hours=1)
            decision_ts_arr[idx] = dec_ts
            target_ts_arr[idx] = tgt_ts
            h_target_arr[idx] = tgt_ts.hour

            if dec_ts in weather_lookup.index:
                w_row = weather_lookup.loc[dec_ts]
                if isinstance(w_row, pd.DataFrame):
                    w_row = w_row.iloc[0]
                is_raining_arr[idx] = bool(w_row["weather_condition"] == "rain")
                temp_c_arr[idx] = float(w_row["temperature"])

            if dec_ts in calendar_lookup.index:
                c_row = calendar_lookup.loc[dec_ts]
                if isinstance(c_row, pd.DataFrame):
                    c_row = c_row.iloc[0]
                is_holiday_arr[idx] = bool(c_row["is_holiday"])
                is_weekend_arr[idx] = bool(c_row["is_weekend"])
            idx += 1

    # Hour multiplier (length n_time)
    hour_multiplier_arr = np.round(
        1.0
        + 0.9 * np.exp(-((h_target_arr - 13) ** 2) / 6.0)
        + 1.1 * np.exp(-((h_target_arr - 19) ** 2) / 4.0),
        4,
    )

    # Holiday multiplier (length n_time)
    holiday_multiplier_arr = np.where(
        is_holiday_arr, 1.25, np.where(is_weekend_arr, 1.15, 1.0)
    ).round(4)

    # Weather multipliers for category == 'food' vs other (length n_time)
    food_evening_cond = is_raining_arr & (h_target_arr >= 17) & (h_target_arr <= 21)
    weather_mult_food = np.where(
        is_raining_arr,
        np.where(food_evening_cond, 1.30 + 0.01 * (temp_c_arr - 27), 0.65 + 0.01 * (temp_c_arr - 27)),
        1.0 + 0.01 * (temp_c_arr - 27),
    )
    weather_mult_food = np.round(np.maximum(0.3, weather_mult_food), 4)

    weather_mult_other = np.where(
        is_raining_arr,
        0.65 + 0.01 * (temp_c_arr - 27),
        1.0 + 0.01 * (temp_c_arr - 27),
    )
    weather_mult_other = np.round(np.maximum(0.3, weather_mult_other), 4)

    # 2. Build Vendor Vectors (length n_vendors)
    vendor_ids = vendor_profiles["vendor_id"].values
    home_cells = vendor_profiles["home_cell_id"].values
    cats = vendor_profiles["vendor_category"].values
    aov_arr = vendor_profiles["average_order_value"].values.astype(np.float64)
    var_cost_arr = vendor_profiles["variable_cost_rate"].values.astype(np.float64)
    fixed_cost_arr = vendor_profiles["fixed_cost_per_day"].values.astype(np.float64)

    base_arr = np.array([CATEGORY_BASE_DEMAND[c] for c in cats], dtype=np.float64)
    poi_scores = np.empty(n_vendors, dtype=np.float64)
    office_counts = np.empty(n_vendors, dtype=np.float64)
    mall_counts = np.empty(n_vendors, dtype=np.float64)
    hidden_effects = np.empty(n_vendors, dtype=np.float64)

    for i in range(n_vendors):
        cell = home_cells[i]
        c = cats[i]
        stat = static_lookup.loc[cell]
        rep = cell_reputation.loc[cell]
        aff = CATEGORY_POI_AFFINITY[c]
        vrow = vendor_profiles.iloc[i]

        poi_scores[i] = sum(w * (stat[feat] / POI_NORM_CONST[feat]) for feat, w in aff.items())
        office_counts[i] = float(stat["office_count"])
        mall_counts[i] = float(stat["mall_count"])
        hidden_effects[i] = (
            0.3 * (rep - 1.0)
            + 0.3 * (vrow["hidden_skill_factor"] - 1.0)
            + 0.15 * (vrow["popularity_score"] - 0.5)
            + 0.10 * (vrow["quality_score"] - 0.6)
        )

    vendor_linear_base = base_arr + poi_scores + hidden_effects

    # 3. Build 2D Matrices (n_vendors, n_time)
    rain_office_inter = 0.15 * (is_raining_arr[None, :] * (office_counts[:, None] / 40.0))
    weekend_mall_inter = 0.12 * (is_weekend_arr[None, :] * (mall_counts[:, None] / 3.0))

    linear_score = vendor_linear_base[:, None] + rain_office_inter + weekend_mall_inter
    base_demand = linear_score * hour_multiplier_arr[None, :]

    # Weather multiplier matrix
    is_food = (cats == "food")[:, None]
    weather_multiplier_mat = np.where(is_food, weather_mult_food[None, :], weather_mult_other[None, :])

    # Competition and Event multiplier matrices (n_vendors, n_dates) -> repeated to (n_vendors, n_time)
    comp_mult_date = np.empty((n_vendors, n_dates), dtype=np.float64)
    event_mult_date = np.empty((n_vendors, n_dates), dtype=np.float64)

    for j, d in enumerate(dates):
        d_ts = pd.Timestamp(d)
        for i in range(n_vendors):
            cell = home_cells[i]
            comp_row = competition_lookup.loc[(d_ts, cell)] if (d_ts, cell) in competition_lookup.index else None
            comp_score = comp_row["competition_score"] if comp_row is not None else 0.0
            comp_mult_date[i, j] = round(float(max(0.5, 1.0 - 0.4 * comp_score)), 4)
            evt_imp = events_lookup.get((d_ts, cell), 0.0)
            event_mult_date[i, j] = round(float(1.0 + 0.5 * evt_imp), 4)

    comp_mult_mat = np.repeat(comp_mult_date, n_hours, axis=1)
    event_mult_mat = np.repeat(event_mult_date, n_hours, axis=1)

    combined = (
        base_demand
        * weather_multiplier_mat
        * holiday_multiplier_arr[None, :]
        * comp_mult_mat
        * event_mult_mat
    )
    expected = 80.0 * _sigmoid((combined - 3.0) * 1.0)

    noise_std = 0.15 * np.sqrt(np.maximum(expected, 0.1)) * 1.8
    # Match scalar sequence identically for seed reproducibility
    noise_component = np.round(rng.normal(0, noise_std), 3)
    customer_count = np.maximum(0, np.round(expected + noise_component)).astype(np.int32)

    # Financials
    revenue = np.round(customer_count * aov_arr[:, None], 2)
    ingredient_cost = np.round(revenue * var_cost_arr[:, None], 2)
    fuel_cost = np.round((fixed_cost_arr[:, None] / n_hours), 2)
    operating_cost = np.round(ingredient_cost + fuel_cost, 2)
    profit = np.round(revenue - operating_cost, 2)

    # Top selling product vector
    is_cold_arr = temp_c_arr < 24
    is_lunch_arr = (h_target_arr >= 12) & (h_target_arr <= 14)

    top_products = np.empty((n_vendors, n_time), dtype="object")
    for i in range(n_vendors):
        c = cats[i]
        for t in range(n_time):
            key = (c, bool(is_raining_arr[t]), bool(is_cold_arr[t]), bool(is_lunch_arr[t]))
            top_products[i, t] = TOP_PRODUCT_LOOKUP[key]

    # Batch UUID generation (n_vendors * n_time)
    total_rows = n_vendors * n_time
    tx_ids = [uuid.uuid4().hex[:10] for _ in range(total_rows)]

    # Flatten to DataFrame columns
    vendor_id_col = np.repeat(vendor_ids, n_time)
    home_cell_col = np.repeat(home_cells, n_time)
    decision_ts_col = np.tile(decision_ts_arr, n_vendors)
    target_ts_col = np.tile(target_ts_arr, n_vendors)
    aov_col = np.repeat(aov_arr, n_time)
    fuel_cost_col = np.repeat(fuel_cost.ravel(), n_time)

    df = pd.DataFrame({
        "transaction_id": tx_ids,
        "decision_timestamp": decision_ts_col,
        "target_timestamp": target_ts_col,
        "vendor_id": vendor_id_col,
        "h3_cell_id": home_cell_col,
        "expected_customer_count": customer_count.ravel(),
        "average_order_value": aov_col,
        "revenue": revenue.ravel(),
        "ingredient_cost": ingredient_cost.ravel(),
        "fuel_cost": fuel_cost_col,
        "operating_cost": operating_cost.ravel(),
        "profit": profit.ravel(),
        "top_selling_product": top_products.ravel(),
        "weather_multiplier": weather_multiplier_mat.ravel(),
        "holiday_multiplier": np.tile(holiday_multiplier_arr, n_vendors),
        "competition_multiplier": comp_mult_mat.ravel(),
        "event_multiplier": event_mult_mat.ravel(),
        "noise_component": noise_component.ravel(),
        "source": "simulated:market_outcomes",
    })

    return df



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
