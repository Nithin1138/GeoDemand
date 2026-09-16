"""
Dataset 7 — Vendor Profiles (vendor_profiles.parquet)

Single responsibility: one row per simulated vendor's business attributes.
Built on top of the original Vendor Profile Generator (src/simulation/
vendor_profile.py), extended with the fuller schema from the pipeline
spec: vendor_name, menu_type, and three additional simulation-only latent
factors (popularity_score, quality_score, repeat_customer_rate) alongside
hidden_skill_factor — all four are marked simulation-only and excluded
from the model's feature set at feature_store build time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
from vendor_profile import generate_vendor_profiles, CATEGORY_MENUS  # noqa: E402

VENDOR_NAME_PREFIXES = {
    "food": ["Sri", "Annapurna", "Royal", "Krishna", "Balaji", "Amma's", "Fresh"],
    "fruits": ["Fresh", "Nature's", "Daily", "Green", "Farm"],
    "salon": ["Style", "Classic", "Modern", "Prince", "Quick"],
    "repair": ["Speed", "Reliable", "24x7", "Master", "City"],
}
VENDOR_NAME_SUFFIXES = {
    "food": ["Tiffins", "Corner", "Snacks", "Point", "Kitchen"],
    "fruits": ["Fruits", "Juice Center", "Stall"],
    "salon": ["Salon", "Cuts", "Grooming"],
    "repair": ["Repairs", "Service", "Fix-It"],
}


def build_vendor_profiles(h3_cell_ids: list[str], n_vendors: int = 150, seed: int = 2026) -> pd.DataFrame:
    grid = pd.DataFrame({"h3_cell": h3_cell_ids})
    profiles = generate_vendor_profiles(grid, n_vendors=n_vendors, seed=seed)
    rng = np.random.default_rng(seed + 2)

    rows = []
    for p in profiles:
        prefix = rng.choice(VENDOR_NAME_PREFIXES[p.category])
        suffix = rng.choice(VENDOR_NAME_SUFFIXES[p.category])
        rows.append({
            "vendor_id": p.vendor_id,
            "vendor_category": p.category,
            "vendor_name": f"{prefix} {suffix}",
            "menu_type": ", ".join(item["item"] for item in p.menu[:3]) + ("..." if len(p.menu) > 3 else ""),
            "average_order_value": round(float(np.mean([item["price"] for item in p.menu])), 2),
            "preparation_time": round(p.avg_service_time_min * 0.6, 1),
            "service_time": p.avg_service_time_min,
            "inventory_capacity": p.max_capacity_per_hour * 4,  # rough per-shift stock estimate
            "operating_start": 8,
            "operating_end": 21,
            "fixed_cost_per_day": p.fixed_daily_cost,
            "variable_cost_rate": p.variable_cost_pct,
            "home_cell_id": p.home_cell,
            "hidden_skill_factor": p.hidden_skill_factor,          # simulation only
            "popularity_score": round(float(np.clip(rng.normal(0.5, 0.15), 0, 1)), 3),   # simulation only
            "quality_score": round(float(np.clip(rng.normal(0.6, 0.12), 0, 1)), 3),       # simulation only
            "repeat_customer_rate": round(float(np.clip(rng.normal(0.3, 0.1), 0, 0.8)), 3),  # simulation only
            "source": "simulated:business_profiles",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from h3_cells import build_h3_cells

    cells = build_h3_cells()
    df = build_vendor_profiles(cells["h3_cell_id"].tolist(), n_vendors=150)
    out = Path(__file__).parent.parent.parent / "data" / "processed" / "vendor_profiles.parquet"
    df.to_parquet(out, index=False)
    print(f"vendor_profiles.parquet: {len(df)} vendors -> {out}")
    print(df[["vendor_id", "vendor_name", "vendor_category", "average_order_value", "popularity_score"]].head(8))
