# GeoDemand AI — Data Directory

This directory contains raw, simulated, and processed datasets powering the GeoDemand AI demand intelligence platform.

> [!IMPORTANT]
> **Authoritative Dataset Catalog**: For complete column dictionaries, schemas, provenance definitions, and data contracts, see [docs/DATASETS_CATALOG.md](file:///Users/nithin/Projects/MOVIGO-ENG/docs/DATASETS_CATALOG.md).

---

## Directory Layout

```text
data/
├── processed/                           # Canonical v2.1 ML Pipeline Datasets
│   ├── h3_cells.parquet                 # Spatial Grid (817 H3 Res-8 cells)
│   ├── static_features.parquet          # POI and Land-Use features per cell
│   ├── weather_hourly.parquet           # City-grain hourly meteorological data
│   ├── calendar.parquet                 # Hourly temporal and holiday flags
│   ├── events.parquet                   # Local festival and gathering flags
│   ├── competition.parquet              # Hyperlocal competitor density
│   ├── vendor_profiles.parquet          # Vendor capacity and business features
│   ├── products.parquet                 # Canonical menu items and margin data
│   ├── historical_transactions.parquet  # Hourly transactions (decision_timestamp -> target_timestamp)
│   ├── feature_store.parquet            # CANONICAL TRAINING DATASET (711,750 x 45)
│   └── pipeline_manifest_v*.json        # Versioned cryptographic manifest
├── raw/                                 # API response caches (OSM Overpass, Open-Meteo)
│   ├── osm/
│   └── weather/
└── simulated/                           # Legacy v1 Datasets (DO NOT USE FOR TRAINING)
    └── dataset_v2bc7185bc94d.parquet    # Monolithic legacy table (superseded by v2.1)
```

---

## Active ML Training Source

* **Training Dataset**: `data/processed/feature_store.parquet`
* **Prediction Contract**: Features evaluated at `decision_timestamp` ($t$) to predict `expected_customer_count` at `target_timestamp` ($t+1\text{h}$).
* **Pipeline Schema Version**: `v2.1-next-hour-forecast`
