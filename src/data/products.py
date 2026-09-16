"""
Dataset 8 — Product Catalog (products.parquet)

Single responsibility: one row per menu item across all vendor categories.
Extracted from the CATEGORY_MENUS definitions in vendor_profile.py so
there's one canonical product list instead of prices scattered inline.
cost_price is derived from each category's typical variable_cost_rate
(see vendor_profiles.py) rather than invented per-item.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
from vendor_profile import CATEGORY_MENUS, CATEGORY_VARIABLE_COST_PCT  # noqa: E402

PREP_TIME_MIN = {
    "food": 3, "fruits": 2, "salon": 15, "repair": 20,
}
SHELF_LIFE_HOURS = {
    "food": 6, "fruits": 24, "salon": None, "repair": None,  # services have no shelf life
}


def build_products() -> pd.DataFrame:
    rows = []
    for category, menu in CATEGORY_MENUS.items():
        var_lo, var_hi = CATEGORY_VARIABLE_COST_PCT[category]
        avg_cost_rate = (var_lo + var_hi) / 2
        for item in menu:
            rows.append({
                "product_id": str(uuid.uuid4())[:8],
                "vendor_category": category,
                "product_name": item["item"],
                "selling_price": item["price"],
                "cost_price": round(item["price"] * avg_cost_rate, 2),
                "prep_time": PREP_TIME_MIN[category],
                "shelf_life": SHELF_LIFE_HOURS[category],
                "rain_affinity": item["rain_affinity"],
                "cold_affinity": item["cold_affinity"],
                "lunch_affinity": item["lunch_affinity"],
                "source": "simulated:product_catalog",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_products()
    out = Path(__file__).parent.parent.parent / "data" / "processed" / "products.parquet"
    df.to_parquet(out, index=False)
    print(f"products.parquet: {len(df)} products -> {out}")
    print(df.to_string(index=False))
