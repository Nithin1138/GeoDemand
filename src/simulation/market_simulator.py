"""
Market Simulation Engine for GeoDemand AI.

Generates a full year of hourly (H3 cell x vendor) "Expected Customer Count"
observations for 100-200 simulated vendors.

DESIGN NOTE - anti-circularity (Phase 1 review fix #1):
The demand-generating process below is deliberately NOT a simple linear
combination of the exact features we expose to the model. It includes:
  - hidden/latent variables the model never sees directly (a per-cell
    "local reputation" factor, per-vendor "skill" factor)
  - interaction terms between observed features (rain x office density,
    weekend x mall density) rather than pure additive effects
  - a nonlinear demand curve (soft saturation, so demand doesn't grow
    unboundedly with POI counts)
  - heteroscedastic noise (variance grows with expected demand, like real
    count data)
This means a model trained on the exposed features has to actually learn
structure, not invert an equation it was handed. Reported metrics reflect
"can the model recover generative structure from noisy partial observation"
rather than "can XGBoost do algebra."

Every dataset produced here is versioned: a JSON sidecar records the
simulation parameters, seed, feature-source versions, and generation
timestamp, so any run is fully reproducible.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


VENDOR_CATEGORIES = ["food", "fruits", "salon", "repair"]

CATEGORY_BASE_DEMAND = {
    # relative baseline popularity, before any context
    "food": 1.0,
    "fruits": 0.6,
    "salon": 0.35,
    "repair": 0.25,
}

CATEGORY_POI_AFFINITY = {
    # which static POI features each category responds to most
    "food": {"office_count": 0.35, "college_count": 0.30, "restaurant_count": 0.15},
    "fruits": {"population_density": 0.30, "hospital_count": 0.15, "park_count": 0.10},
    "salon": {"population_density": 0.25, "mall_count": 0.20, "office_count": 0.10},
    "repair": {"population_density": 0.15, "office_count": 0.10, "road_density": 0.15},
}


@dataclass
class SimulationParams:
    n_vendors: int = 150
    start_date: str = "2025-01-01"
    n_days: int = 365
    hours_per_day: tuple = (8, 21)  # vendors operate 8am-9pm
    seed: int = 2026
    h3_resolution: int = 8
    center_lat: float = 16.5062
    center_lng: float = 80.6480
    radius_km: float = 8.0
    # Bump this whenever the demand-generating LOGIC changes (not just params),
    # so dataset_v<hash> actually changes when the simulator's behavior changes.
    # Without this, two runs with identical params but different simulator code
    # silently overwrite the same "version" — a real bug caught while wiring
    # up qualitative validation (see CHANGELOG note below).
    sim_logic_version: str = "1.1.0"  # 1.1.0: category+hour-aware rain effect for food

    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _softplus_saturate(x: np.ndarray, cap: float, offset: float = 2.5, slope: float = 1.0) -> np.ndarray:
    """
    Smooth saturation so demand doesn't grow unboundedly with POI density.
    `offset` centers the sigmoid on the typical linear_score value so the
    full 0-cap range is actually used instead of clustering near the cap.
    """
    return cap * _sigmoid((x - offset) * slope)


def assign_vendors(grid: pd.DataFrame, params: SimulationParams, rng: np.random.Generator) -> pd.DataFrame:
    """Assign each simulated vendor a category, a home cell, and a hidden 'skill' factor."""
    n = params.n_vendors
    categories = rng.choice(VENDOR_CATEGORIES, size=n, p=[0.45, 0.25, 0.15, 0.15])
    home_cells = rng.choice(grid["h3_cell"].to_numpy(), size=n)
    # Hidden variable: vendor skill/reputation, never exposed as a model feature.
    # Represents things like "friendliness", "food quality", "word of mouth" -
    # real-world factors that genuinely affect sales but aren't in any dataset.
    skill = rng.normal(loc=1.0, scale=0.18, size=n)
    skill = np.clip(skill, 0.5, 1.6)

    return pd.DataFrame({
        "vendor_id": [f"V{i:04d}" for i in range(n)],
        "category": categories,
        "home_cell": home_cells,
        "hidden_skill_factor": skill,  # kept in simulator output for audit, NOT a model feature
    })


def simulate_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Simple seasonal weather simulator: temp (C) and rain probability by day-of-year."""
    doy = dates.dayofyear.to_numpy()
    # India-ish seasonal pattern: hot summer (Mar-Jun), monsoon (Jul-Sep), mild winter
    temp = 27 + 8 * np.sin(2 * np.pi * (doy - 60) / 365) + rng.normal(0, 1.5, len(dates))
    monsoon = ((doy > 160) & (doy < 270)).astype(float)
    rain_prob = 0.08 + 0.45 * monsoon + rng.normal(0, 0.03, len(dates))
    rain_prob = np.clip(rain_prob, 0, 0.9)
    is_raining = rng.random(len(dates)) < rain_prob
    return pd.DataFrame({"date": dates, "temp_c": temp.round(1), "is_raining": is_raining})


def simulate_holidays(dates: pd.DatetimeIndex) -> pd.Series:
    """Flag a handful of fixed + a few floating holidays/festival days per year."""
    fixed_md = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}
    is_holiday = dates.map(lambda d: (d.month, d.day) in fixed_md)
    return pd.Series(is_holiday, index=dates)


def simulate_events(grid: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sparse random 'events' (college fest, wedding, marathon) at random cells/days.
    Represents the kind of signal a real event-detection layer would surface.
    """
    n_events = int(len(dates) * 0.15)  # events happen on ~15% of cell-days, sparsely
    event_days = rng.choice(dates, size=n_events)
    event_cells = rng.choice(grid["h3_cell"].to_numpy(), size=n_events)
    event_boost = rng.uniform(0.15, 0.6, size=n_events)
    return pd.DataFrame({"date": event_days, "h3_cell": event_cells, "event_boost": event_boost})


def generate_dataset(params: SimulationParams, grid: pd.DataFrame, static_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(params.seed)

    dates = pd.date_range(params.start_date, periods=params.n_days, freq="D")
    weather = simulate_weather(dates, rng)
    holidays = simulate_holidays(dates)
    vendors = assign_vendors(grid, params, rng)
    events = simulate_events(grid, dates, rng)

    static_lookup = static_features.set_index("h3_cell")
    event_lookup = events.groupby(["date", "h3_cell"])["event_boost"].max()

    # Hidden per-cell "local reputation" latent factor - never exposed to the model.
    # Represents unmeasured factors like street visibility, foot-traffic quality,
    # informal reputation that a pure POI-count feature can't capture.
    cell_reputation = pd.Series(
        rng.normal(1.0, 0.15, size=len(grid)).clip(0.6, 1.5),
        index=grid["h3_cell"],
    )

    start_h, end_h = params.hours_per_day
    hours = list(range(start_h, end_h))

    rows = []
    for _, vrow in vendors.iterrows():
        cell = vrow["home_cell"]
        cat = vrow["category"]
        skill = vrow["hidden_skill_factor"]
        static = static_lookup.loc[cell]
        reputation = cell_reputation.loc[cell]
        affinity = CATEGORY_POI_AFFINITY[cat]

        # Normalize POI features to 0-1 scale for use in the demand function
        poi_score = 0.0
        for feat, w in affinity.items():
            raw = static[feat]
            # rough per-feature normalization constants (see static_features.py ranges)
            norm_const = {
                "office_count": 40, "college_count": 3, "restaurant_count": 25,
                "population_density": 25000, "hospital_count": 4, "park_count": 5,
                "mall_count": 3, "road_density": 1.0,
            }[feat]
            poi_score += w * (raw / norm_const)

        for d in dates:
            day_weather = weather.loc[weather["date"] == d].iloc[0]
            is_holiday = bool(holidays.loc[d])
            is_weekend = d.weekday() >= 5
            event_boost = event_lookup.get((d, cell), 0.0)

            for h in hours:
                # --- base + observed-feature effects (what the model CAN see) ---
                base = CATEGORY_BASE_DEMAND[cat]

                # time-of-day curve: lunch (12-14) and evening (18-20) peaks
                hour_effect = (
                    0.9 * np.exp(-((h - 13) ** 2) / 6)
                    + 1.1 * np.exp(-((h - 19) ** 2) / 4)
                    + 0.15
                )

                weather_effect = (-0.35 if day_weather["is_raining"] else 0.0) + \
                                  0.01 * (day_weather["temp_c"] - 27)

                # Rain suppresses general outdoor footfall, but for FOOD vendors
                # specifically during evening hours, hot-item demand (tea/coffee/
                # bajji) partially offsets — and can net-exceed — that footfall
                # loss. This is what makes the "rainy evening -> tea demand up"
                # qualitative check hold at the category level, not just in the
                # product-mix weighting. Other categories keep the flat penalty.
                if cat == "food" and day_weather["is_raining"] and 17 <= h <= 21:
                    weather_effect = 0.30 + 0.01 * (day_weather["temp_c"] - 27)

                holiday_effect = 0.25 if is_holiday else 0.0
                weekend_effect = 0.15 if is_weekend else 0.0

                # --- interaction terms (NOT purely additive - review fix) ---
                rain_office_interaction = (
                    0.15 * float(day_weather["is_raining"]) * (static["office_count"] / 40)
                )
                weekend_mall_interaction = (
                    0.12 * float(is_weekend) * (static["mall_count"] / 3)
                )

                # --- hidden variables (NEVER exposed as model features) ---
                hidden_effect = 0.3 * (reputation - 1.0) + 0.3 * (skill - 1.0)

                # --- combine, nonlinear saturation, heteroscedastic noise ---
                linear_score = (
                    base
                    + poi_score
                    + hour_effect
                    + weather_effect
                    + holiday_effect
                    + weekend_effect
                    + rain_office_interaction
                    + weekend_mall_interaction
                    + event_boost
                    + hidden_effect
                )
                expected = _softplus_saturate(linear_score, cap=80.0, offset=2.5, slope=1.1)

                noise_std = 0.15 * np.sqrt(max(expected, 0.1)) * 1.8  # heteroscedastic
                noisy = expected + rng.normal(0, noise_std)
                customer_count = max(0, int(round(noisy)))

                rows.append({
                    "date": d,
                    "hour": h,
                    "vendor_id": vrow["vendor_id"],
                    "category": cat,
                    "h3_cell": cell,
                    "is_raining": bool(day_weather["is_raining"]),
                    "temp_c": day_weather["temp_c"],
                    "is_holiday": is_holiday,
                    "is_weekend": is_weekend,
                    "event_flag": event_boost > 0,
                    "event_boost_raw": round(float(event_boost), 3),
                    "customer_count": customer_count,
                })

    df = pd.DataFrame(rows)

    metadata = {
        "version_hash": params.content_hash(),
        "generated_at": datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "simulation_params": asdict(params),
        "n_rows": len(df),
        "n_vendors": params.n_vendors,
        "n_cells_covered": df["h3_cell"].nunique(),
        "date_range": [str(dates.min().date()), str(dates.max().date())],
        "notes": (
            "Synthetic target (customer_count). Contextual features (weather, "
            "calendar) are simulated but structurally realistic. Demand-generating "
            "process includes hidden variables (vendor skill, cell reputation) and "
            "feature interactions NOT exposed to downstream models, to avoid "
            "trivial formula-inversion during training. See module docstring."
        ),
    }

    return df, metadata


def save_versioned_dataset(df: pd.DataFrame, metadata: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = metadata["version_hash"]

    parquet_path = out_dir / f"dataset_v{version}.parquet"
    meta_path = out_dir / f"dataset_v{version}.meta.json"

    df.to_parquet(parquet_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved {len(df):,} rows -> {parquet_path}")
    print(f"Metadata -> {meta_path}")
    return parquet_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from h3_grid import generate_city_grid
    from static_features_synth import generate_synthetic_static_features

    params = SimulationParams(n_vendors=150, n_days=365)
    grid = generate_city_grid(params.center_lat, params.center_lng, params.radius_km, params.h3_resolution)
    static = generate_synthetic_static_features(grid, seed=params.seed)

    df, meta = generate_dataset(params, grid, static)
    print(df.head())
    print(f"\nTotal rows: {len(df):,}")
    print(f"customer_count stats:\n{df['customer_count'].describe()}")

    save_versioned_dataset(df, meta, out_dir=Path(__file__).parent.parent.parent / "data" / "simulated")
