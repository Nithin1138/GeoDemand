"""
Dataset 6 — Competition (competition.parquet)

Single responsibility: how many competing vendors (by category) are near
each cell, at daily grain (vendor home-cell assignment doesn't change
hour to hour in this v1 model — that's a v2 extension once vendors can
actually relocate based on recommendations).

Hybrid by design: informal-vendor presence isn't in OSM (street vendors
aren't mapped), so competitor counts are derived from our own simulated
vendor_profiles positions using REAL H3 spatial adjacency (grid_disk) —
not a made-up number. This matches the spec's "🟡 Hybrid (OSM + simulation)"
classification: the counting geometry is real, the entities being counted
are simulated because no public directory of informal vendors exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd


def build_competition(
    h3_cell_ids: list[str],
    vendor_profiles: pd.DataFrame,
    dates: pd.DatetimeIndex,
    neighbor_rings: int = 1,
) -> pd.DataFrame:
    """
    vendor_profiles: needs vendor_category, home_cell_id columns.
    For each h3_cell, counts same-cell + neighbor-ring vendors by category
    using h3.grid_disk (real spatial adjacency), then computes a single
    competition_score. Vendor positions are static across the whole period
    in v1 (no daily relocation yet), so this is repeated per date only to
    make the join key consistent with the rest of the pipeline — cheap
    since the underlying counts don't change.
    """
    categories = vendor_profiles["vendor_category"].unique().tolist()
    cat_col_map = {
        "food": "food_truck_count", "fruits": "fruit_vendor_count",
        "salon": "salon_van_count", "repair": "repair_van_count",
    }
    # tea/snack counts are a food sub-split — approximate as half of food_truck_count
    # for schema completeness without inventing a separate simulation dimension.

    counts_by_cell = {}
    for cell in h3_cell_ids:
        neighborhood = h3.grid_disk(cell, neighbor_rings)
        local_vendors = vendor_profiles[vendor_profiles["home_cell_id"].isin(neighborhood)]
        cat_counts = local_vendors["vendor_category"].value_counts().to_dict()
        counts_by_cell[cell] = cat_counts

    rows = []
    for d in dates:
        for cell in h3_cell_ids:
            cat_counts = counts_by_cell[cell]
            food_ct = cat_counts.get("food", 0)
            row = {
                "date": d,
                "h3_cell_id": cell,
                "tea_vendor_count": food_ct // 2,
                "snack_vendor_count": food_ct - (food_ct // 2),
                "fruit_vendor_count": cat_counts.get("fruits", 0),
                "food_truck_count": food_ct,
                "repair_van_count": cat_counts.get("repair", 0),
                "salon_van_count": cat_counts.get("salon", 0),
            }
            total_competitors = sum(cat_counts.values())
            row["competition_score"] = round(min(1.0, total_competitors / 8.0), 3)  # normalized, cap at 8 vendors = saturated
            row["source"] = "hybrid:h3_spatial_adjacency"
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from h3_cells import build_h3_cells
    from vendor_profiles import build_vendor_profiles

    cells = build_h3_cells()
    vendors = build_vendor_profiles(cells["h3_cell_id"].tolist(), n_vendors=150)
    dates = pd.date_range("2025-01-01", "2025-01-07", freq="D")  # smoke test: 1 week

    df = build_competition(cells["h3_cell_id"].tolist(), vendors, dates)
    print(f"competition.parquet (smoke test, 1 week): {len(df):,} rows")
    print(df.describe()[["competition_score", "food_truck_count"]])
    print(df.sort_values("competition_score", ascending=False).head())
