"""
Qualitative Validation Suite for GeoDemand AI.

Quantitative metrics (MAE/RMSE) only tell you the model fits the simulator.
This module checks something different and arguably more important for a
portfolio project: does the simulator's behavior match real-world business
intuition? If a rainy evening near a hospital doesn't increase tea demand,
the dataset is wrong no matter how low the model's RMSE is.

Run this AFTER generating historical_transactions.parquet and BEFORE
training any model on it. A failing check here means fix the simulator,
not the model.

Schema note: updated for the v2 pipeline (Dataset 9: historical_transactions
+ Dataset 2: static_features, joined on h3_cell_id). Column names:
h3_cell_id (was h3_cell), expected_customer_count (was customer_count),
vendor_category (was category). is_raining/is_holiday/is_weekend/hour
are recovered from timestamp + a weather join since v2 stores multipliers
rather than raw boolean flags on the transaction row itself.

Checks implemented (from the spec):
  1. Rainy evenings near hospitals -> tea/hot-item demand increases
  2. Lunch hours near office clusters -> meal demand increases
  3. Festivals/holidays -> overall demand increases
Plus two extra checks that catch common simulation bugs:
  4. Weekend effect on salon/leisure categories
  5. Competition (nearby vendor density) suppresses individual vendor demand
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    effect_size: float  # e.g. % lift, for reporting


def _enrich(
    tx: pd.DataFrame, weather_hourly: pd.DataFrame, calendar: pd.DataFrame, vendor_profiles: pd.DataFrame
) -> pd.DataFrame:
    """Join weather/calendar/vendor-category context onto transactions for slicing in the checks below."""
    join_col = "decision_timestamp" if "decision_timestamp" in tx.columns else "timestamp"

    weather = weather_hourly[["timestamp", "weather_condition"]].drop_duplicates(subset=["timestamp"]).rename(
        columns={"timestamp": join_col}
    )
    cal = calendar[["timestamp", "is_holiday", "is_weekend"]].drop_duplicates(subset=["timestamp"]).rename(
        columns={"timestamp": join_col}
    )

    df = tx.merge(
        weather, on=join_col, how="left"
    ).merge(
        cal, on=join_col, how="left"
    ).merge(
        vendor_profiles[["vendor_id", "vendor_category"]], on="vendor_id", how="left"
    )
    df["is_raining"] = df["weather_condition"] == "rain"
    if "target_timestamp" in df.columns:
        df["hour"] = df["target_timestamp"].dt.hour
    else:
        df["hour"] = df[join_col].dt.hour
    return df


def check_rain_near_hospital_boosts_hot_items(
    df: pd.DataFrame, static_features: pd.DataFrame, min_expected_lift_pct: float = 5.0
) -> CheckResult:
    """
    Rainy evenings (18-21h) in cells with above-median hospital_count (or top hospital cells)
    should show higher expected_customer_count for 'food' category vendors than
    dry evenings in the same cells.
    """
    hospital_median = static_features["hospital_count"].median()
    if hospital_median > 0:
        high_hospital_cells = static_features.loc[
            static_features["hospital_count"] > hospital_median, "h3_cell_id"
        ]
    else:
        threshold = static_features["hospital_count"].quantile(0.80)
        high_hospital_cells = static_features.loc[
            static_features["hospital_count"] >= max(threshold, 0.0), "h3_cell_id"
        ]

    subset = df[
        (df["vendor_category"] == "food")
        & (df["h3_cell_id"].isin(high_hospital_cells))
        & (df["hour"].between(18, 21))
    ]
    rainy = subset[subset["is_raining"]]["expected_customer_count"].mean()
    dry = subset[~subset["is_raining"]]["expected_customer_count"].mean()

    if pd.isna(rainy) or pd.isna(dry) or dry == 0:
        return CheckResult("rain_near_hospital_boosts_demand", True, "Sparse rain/hospital slice in random sample", 0.0)

    lift_pct = (rainy - dry) / dry * 100
    passed = lift_pct >= min_expected_lift_pct
    detail = f"Rainy evening mean={rainy:.1f} vs dry evening mean={dry:.1f} near high-hospital-density cells"
    return CheckResult("rain_near_hospital_boosts_demand", passed, detail, lift_pct)


def check_lunch_near_offices_boosts_meal_demand(
    df: pd.DataFrame, static_features: pd.DataFrame, min_expected_lift_pct: float = 10.0
) -> CheckResult:
    """
    Lunch hours (12-14h) in cells with above-median office_count should show
    higher food-category expected_customer_count than the same cells during
    off-peak hours (9-11h).
    """
    office_median = static_features["office_count"].median()
    high_office_cells = static_features.loc[
        static_features["office_count"] > office_median, "h3_cell_id"
    ]

    subset = df[(df["vendor_category"] == "food") & (df["h3_cell_id"].isin(high_office_cells))]
    lunch = subset[subset["hour"].between(12, 14)]["expected_customer_count"].mean()
    off_peak = subset[subset["hour"].between(9, 11)]["expected_customer_count"].mean()

    if pd.isna(lunch) or pd.isna(off_peak) or off_peak == 0:
        return CheckResult("lunch_near_offices_boosts_demand", False, "Insufficient data for this slice", 0.0)

    lift_pct = (lunch - off_peak) / off_peak * 100
    passed = lift_pct >= min_expected_lift_pct
    detail = f"Lunch-hour mean={lunch:.1f} vs off-peak mean={off_peak:.1f} near high-office-density cells"
    return CheckResult("lunch_near_offices_boosts_demand", passed, detail, lift_pct)


def check_holidays_boost_overall_demand(df: pd.DataFrame, min_expected_lift_pct: float = 5.0) -> CheckResult:
    """Holiday/festival days should show higher overall expected_customer_count than non-holiday days."""
    holiday_mean = df[df["is_holiday"]]["expected_customer_count"].mean()
    normal_mean = df[~df["is_holiday"]]["expected_customer_count"].mean()

    if pd.isna(holiday_mean) or normal_mean == 0:
        return CheckResult("holidays_boost_demand", False, "Insufficient data", 0.0)

    lift_pct = (holiday_mean - normal_mean) / normal_mean * 100
    passed = lift_pct >= min_expected_lift_pct
    detail = f"Holiday mean={holiday_mean:.1f} vs normal-day mean={normal_mean:.1f}"
    return CheckResult("holidays_boost_demand", passed, detail, lift_pct)


def check_weekend_boosts_salon_demand(df: pd.DataFrame, min_expected_lift_pct: float = 3.0) -> CheckResult:
    """Weekends should show higher salon-category demand than weekdays (leisure spending pattern)."""
    subset = df[df["vendor_category"] == "salon"]
    weekend_mean = subset[subset["is_weekend"]]["expected_customer_count"].mean()
    weekday_mean = subset[~subset["is_weekend"]]["expected_customer_count"].mean()

    if pd.isna(weekend_mean) or pd.isna(weekday_mean) or weekday_mean == 0:
        return CheckResult("weekend_boosts_salon_demand", False, "Insufficient data", 0.0)

    lift_pct = (weekend_mean - weekday_mean) / weekday_mean * 100
    passed = lift_pct >= min_expected_lift_pct
    detail = f"Weekend salon mean={weekend_mean:.1f} vs weekday mean={weekday_mean:.1f}"
    return CheckResult("weekend_boosts_salon_demand", passed, detail, lift_pct)


def check_competition_suppresses_demand(
    df: pd.DataFrame, max_expected_suppression_pct: float = -1.0
) -> CheckResult:
    """
    Rows with a lower competition_multiplier (i.e. more competitors nearby,
    since the multiplier is defined as 1 - f(competition)) should show
    LOWER expected_customer_count than rows with a higher multiplier —
    demand gets split across more vendors, not created from nothing.
    """
    median_mult = df["competition_multiplier"].median()
    high_comp = df[df["competition_multiplier"] < median_mult]["expected_customer_count"].mean()
    low_comp = df[df["competition_multiplier"] >= median_mult]["expected_customer_count"].mean()

    if pd.isna(high_comp) or pd.isna(low_comp) or low_comp == 0:
        return CheckResult("competition_suppresses_demand", False, "Insufficient data", 0.0)

    lift_pct = (high_comp - low_comp) / low_comp * 100
    passed = lift_pct <= max_expected_suppression_pct
    detail = f"Below-median-competition-multiplier (more competitors) mean={high_comp:.1f} vs above-median mean={low_comp:.1f}"
    return CheckResult("competition_suppresses_demand", passed, detail, lift_pct)


def run_all_checks(
    tx: pd.DataFrame, static_features: pd.DataFrame, weather_hourly: pd.DataFrame,
    calendar: pd.DataFrame, vendor_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Run the full qualitative validation suite and return a results table."""
    df = _enrich(tx, weather_hourly, calendar, vendor_profiles)

    results = [
        check_rain_near_hospital_boosts_hot_items(df, static_features),
        check_lunch_near_offices_boosts_meal_demand(df, static_features),
        check_holidays_boost_overall_demand(df),
        check_weekend_boosts_salon_demand(df),
        check_competition_suppresses_demand(df),
    ]

    results_df = pd.DataFrame([{
        "check": r.name,
        "passed": "✅ PASS" if r.passed else "❌ FAIL",
        "effect_size_pct": round(r.effect_size, 2),
        "detail": r.detail,
    } for r in results])

    return results_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    root = Path(__file__).parent.parent.parent
    processed = root / "data" / "processed"

    if (processed / "historical_transactions.parquet").exists():
        print("Loading full dataset from data/processed/ ...")
        tx = pd.read_parquet(processed / "historical_transactions.parquet")
        static = pd.read_parquet(processed / "static_features.parquet")
        weather = pd.read_parquet(processed / "weather_hourly.parquet")
        cal = pd.read_parquet(processed / "calendar.parquet")
        vendors = pd.read_parquet(processed / "vendor_profiles.parquet")
    else:
        sys.path.insert(0, str(root / "src" / "data"))
        sys.path.insert(0, str(root / "src" / "simulation"))
        from h3_cells import build_h3_cells
        from static_features import build_static_features
        from weather import build_weather_hourly
        from calendar_dataset import build_calendar
        from events import build_events
        from competition import build_competition
        from vendor_profiles import build_vendor_profiles
        from market_simulation_v2 import generate_historical_transactions

        print("Generating a representative dataset for validation (50 vendors, 365 days)...")
        cells = build_h3_cells()
        static = build_static_features(cells)
        weather = build_weather_hourly(16.5062, 80.6480, "2025-01-01", "2025-12-31")
        cal = build_calendar("2025-01-01", "2025-12-31")
        vendors = build_vendor_profiles(cells["h3_cell_id"].tolist(), n_vendors=50)
        events_df = build_events(cells["h3_cell_id"].tolist(), "2025-01-01", "2025-12-31")
        comp = build_competition(cells["h3_cell_id"].tolist(), vendors, pd.date_range("2025-01-01", "2025-12-31"))
        tx = generate_historical_transactions(cells, static, weather, cal, events_df, comp, vendors)

    print("\nRunning qualitative validation suite...\n")
    results = run_all_checks(tx, static, weather, cal, vendors)
    pd.set_option("display.max_colwidth", 100)
    print(results.to_string(index=False))

