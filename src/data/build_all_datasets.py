"""
Master dataset build script for GeoDemand AI (Pipeline Schema v2.1).

Builds all datasets in proper dependency order and saves to data/processed/:
  1. h3_cells.parquet
  2. static_features.parquet
  3. weather_hourly.parquet
  4. calendar.parquet
  5. events.parquet
  6. vendor_profiles.parquet
  7. competition.parquet
  8. products.parquet
  9. historical_transactions.parquet
 10. feature_store.parquet
 11. pipeline_manifest_v<hash>.json
"""

from __future__ import annotations

import time
import json
from pathlib import Path
import pandas as pd

from h3_cells import build_h3_cells
from static_features import build_static_features
from weather import build_weather_hourly
from calendar_dataset import build_calendar
from events import build_events
from vendor_profiles import build_vendor_profiles
from competition import build_competition
from products import build_products
from pipeline_version import snapshot_pipeline_version

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
from market_simulation_v2 import generate_historical_transactions

sys.path.insert(0, str(Path(__file__).parent.parent / "features"))
from feature_store import build_feature_store


def build_all(
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    n_vendors: int = 150,
    center_lat: float = 16.5062,
    center_lng: float = 80.6480,
    radius_km: float = 8.0,
    seed: int = 2026,
    processed_dir: Path | None = None,
) -> dict:
    if processed_dir is None:
        processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GeoDemand AI — Building Dataset Pipeline v2.1 (Next-Hour Forecast)")
    print("=" * 70)

    # 1. H3 Cells
    t0 = time.time()
    print("\n1. Building H3 spatial grid...")
    cells = build_h3_cells(center_lat=center_lat, center_lng=center_lng, radius_km=radius_km)
    cells.to_parquet(processed_dir / "h3_cells.parquet", index=False)
    print(f"   -> {len(cells):,} cells saved in {time.time()-t0:.2f}s")

    # 2. Static Features
    t0 = time.time()
    print("\n2. Building static spatial features...")
    static = build_static_features(cells, seed=seed)
    static.to_parquet(processed_dir / "static_features.parquet", index=False)
    print(f"   -> {len(static):,} cells with {static.shape[1]} features in {time.time()-t0:.2f}s")

    # 3. Weather Hourly
    t0 = time.time()
    print("\n3. Building weather hourly dataset...")
    weather = build_weather_hourly(center_lat, center_lng, start_date, end_date, seed=seed)
    weather.to_parquet(processed_dir / "weather_hourly.parquet", index=False)
    print(f"   -> {len(weather):,} hourly records in {time.time()-t0:.2f}s")

    # 4. Calendar
    t0 = time.time()
    print("\n4. Building calendar & holidays dataset...")
    cal = build_calendar(start_date, end_date)
    cal.to_parquet(processed_dir / "calendar.parquet", index=False)
    print(f"   -> {len(cal):,} calendar records in {time.time()-t0:.2f}s")

    # 5. Events
    t0 = time.time()
    print("\n5. Building sparse local events...")
    events = build_events(cells["h3_cell_id"].tolist(), start_date, end_date, seed=seed)
    events.to_parquet(processed_dir / "events.parquet", index=False)
    print(f"   -> {len(events):,} local events in {time.time()-t0:.2f}s")

    # 6. Vendor Profiles
    t0 = time.time()
    print("\n6. Building vendor profiles...")
    vendors = build_vendor_profiles(cells["h3_cell_id"].tolist(), n_vendors=n_vendors, seed=seed)
    vendors.to_parquet(processed_dir / "vendor_profiles.parquet", index=False)
    print(f"   -> {len(vendors):,} vendor profiles in {time.time()-t0:.2f}s")

    # 7. Competition
    t0 = time.time()
    print("\n7. Building competition density dataset...")
    dates = pd.date_range(start_date, end_date, freq="D")
    comp = build_competition(cells["h3_cell_id"].tolist(), vendors, dates)
    comp.to_parquet(processed_dir / "competition.parquet", index=False)
    print(f"   -> {len(comp):,} competition records in {time.time()-t0:.2f}s")

    # 8. Products
    t0 = time.time()
    print("\n8. Building product catalogue...")
    prods = build_products()
    prods.to_parquet(processed_dir / "products.parquet", index=False)
    print(f"   -> {len(prods):,} menu products in {time.time()-t0:.2f}s")

    # 9. Historical Transactions (Next-Hour Contract)
    t0 = time.time()
    print("\n9. Simulating historical transactions (next-hour demand contract)...")
    tx = generate_historical_transactions(cells, static, weather, cal, events, comp, vendors, seed=seed)
    tx.to_parquet(processed_dir / "historical_transactions.parquet", index=False)
    print(f"   -> {len(tx):,} transaction records in {time.time()-t0:.2f}s")
    print(f"   -> Columns: {list(tx.columns)}")

    # 10. Feature Store
    t0 = time.time()
    print("\n10. Building Unified Feature Store...")
    fs = build_feature_store(tx, static, weather, cal, comp, vendors, events)
    fs.to_parquet(processed_dir / "feature_store.parquet", index=False)
    print(f"   -> {len(fs):,} rows x {fs.shape[1]} columns in {time.time()-t0:.2f}s")

    # 11. Pipeline Manifest
    print("\n11. Generating versioned pipeline manifest...")
    manifest = snapshot_pipeline_version(processed_dir)
    manifest_path = processed_dir / f"pipeline_manifest_v{manifest['pipeline_version_hash']}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   -> Version: v{manifest['pipeline_version_hash']} saved to {manifest_path.name}")

    print("\n" + "=" * 70)
    print("Pipeline build successfully completed!")
    print("=" * 70)
    return manifest


if __name__ == "__main__":
    build_all()
