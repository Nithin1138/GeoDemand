"""
Pipeline-level dataset versioning.

Every dataset in the 12-dataset pipeline has its OWN generation code and
can be regenerated independently, but for training you need to know that
a given feature_store.parquet was built from a mutually consistent set of
upstream datasets — not e.g. historical_transactions.parquet from one run
joined against weather_hourly.parquet from a different run with different
random weather (see the reproducibility bug fixed in weather_client.py,
Section 6.4 of the master doc — two separate script invocations produced
different synthetic rain days because Python's str hash() is randomized
per-process; this was the actual root cause of two qualitative-check
"failures" that looked like simulator bugs but were a mismatched dataset
bug instead).

Run this after generating all datasets in `data/processed/` from a single
consistent pipeline run to snapshot their versions together.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PIPELINE_FILES = [
    "h3_cells.parquet", "static_features.parquet", "weather_hourly.parquet",
    "calendar.parquet", "events.parquet", "competition.parquet",
    "vendor_profiles.parquet", "historical_transactions.parquet", "feature_store.parquet",
]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


DATASET_PROVENANCE = {
    "h3_cells.parquet": {
        "classification": "real:spatial_indexing",
        "description": "Deterministic H3 hexagonal indexing (Uber H3, Res 8) over Vijayawada geographic boundary",
    },
    "static_features.parquet": {
        "classification": "real_periodic:openstreetmap_overpass",
        "description": "Spatial POI counts and land-use attributes from OpenStreetMap Overpass (population density via OSM building proxy)",
    },
    "weather_hourly.parquet": {
        "classification": "real_periodic:open_meteo_archive",
        "description": "Hourly weather metrics from Open-Meteo archive (with deterministic diurnal simulation fallback)",
    },
    "calendar.parquet": {
        "classification": "real:holidays_lib",
        "description": "Temporal and national/state holiday calendar flags from Python holidays library",
    },
    "events.parquet": {
        "classification": "simulated:sparse_stochastic",
        "description": "Sparse stochastic festival, wedding, and college fest local event spikes",
    },
    "competition.parquet": {
        "classification": "hybrid:h3_spatial_adjacency",
        "description": "Real H3 grid_disk geometric counting applied over simulated vendor category profiles",
    },
    "vendor_profiles.parquet": {
        "classification": "simulated:business_profiles",
        "description": "Vendor business characteristics with latent simulation factors excluded from model training",
    },
    "historical_transactions.parquet": {
        "classification": "simulated:market_outcomes",
        "description": "Simulated hourly transactions with decision_timestamp (t) -> target_timestamp (t+1h) demand contract",
    },
    "feature_store.parquet": {
        "classification": "hybrid:joined_feature_store",
        "description": "Unified ML training store: features joined strictly at decision_timestamp t to predict target at t+1h",
    },
    "products.parquet": {
        "classification": "simulated:product_catalog",
        "description": "Canonical menu item catalogue and variable cost affinities for downstream revenue engine",
    },
}


def snapshot_pipeline_version(processed_dir: str | Path) -> dict:
    processed_dir = Path(processed_dir)
    manifest = {
        "pipeline_schema_version": "v2.1-next-hour-forecast",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    for fname in PIPELINE_FILES:
        fpath = processed_dir / fname
        if not fpath.exists():
            manifest["files"][fname] = {"status": "MISSING"}
            continue
        df = pd.read_parquet(fpath)
        prov = DATASET_PROVENANCE.get(fname, {"classification": "unknown", "description": ""})
        
        # Dynamically use the dataset's actual provenance tag if present
        provenance_str = prov["classification"]
        if "source" in df.columns and not df.empty:
            actual_source = str(df["source"].iloc[0])
            if actual_source:
                provenance_str = actual_source
        elif "source_type" in df.columns and not df.empty:
            actual_source_type = str(df["source_type"].iloc[0])
            if actual_source_type:
                provenance_str = actual_source_type

        manifest["files"][fname] = {
            "sha256_16": file_hash(fpath),
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "size_bytes": fpath.stat().st_size,
            "provenance": provenance_str,
            "description": prov["description"],
            "columns": list(df.columns),
        }

    combined = json.dumps(manifest["files"], sort_keys=True).encode()
    manifest["pipeline_version_hash"] = hashlib.sha256(combined).hexdigest()[:12]
    return manifest


if __name__ == "__main__":
    processed = Path(__file__).parent.parent.parent / "data" / "processed"
    manifest = snapshot_pipeline_version(processed)

    out = processed / f"pipeline_manifest_v{manifest['pipeline_version_hash']}.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Pipeline version: {manifest['pipeline_version_hash']}")
    for fname, info in manifest["files"].items():
        status = info.get("status", f"{info.get('n_rows', '?'):,} rows")
        print(f"  {fname}: {status}")
    print(f"\nManifest saved -> {out}")
