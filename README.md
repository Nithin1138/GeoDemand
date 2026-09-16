# 📍 GeoDemand AI — Geospatial Decision Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-ff69b4.svg)](https://lightgbm.readthedocs.io/)
[![Uber H3](https://img.shields.io/badge/Spatial-Uber%20H3%20Res8-orange.svg)](https://h3geo.org/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **GeoDemand AI** is a production-shaped geospatial decision intelligence platform designed for mobile vendors (food trucks, street carts, mobile retail). It combines real-time environmental context, OpenStreetMap building and POI infrastructure, and a gradient-boosted demand forecasting engine with quantile uncertainty bounds, road navigation, and travel economics to answer the fundamental operational question:
>
> **"From where I am operating right now, what is the predicted demand and net profit at nearby locations over the next hour, and does relocating justify the travel time, fuel expense, and operating friction versus staying put?"**

---

## 🌟 Key Features

- **🌐 Uber H3 Hexagonal Indexing (Resolution 8):** Uniform equidistant spatial discretization ($\sim 461\,\text{m}$ edge length, $\sim 0.74\,\text{km}^2$ area) with automated water-body polygon exclusion.
- **🛰️ Hybrid Real-Time Data Ingestion:**
  - **Live Weather:** Real-time ambient readings and next-hour precipitation forecasts via Open-Meteo.
  - **Real Infrastructure:** 35,794 real OpenStreetMap POIs (offices, schools, malls, transport hubs) extracted via Overpass QL.
  - **Contextual Signals:** Road traffic speeds (HERE API / diurnal proxy), category competition (Google Places / OSM), and public event footprints.
- **⚡ Next-Hour Machine Learning Demand Forecast:** LightGBM regressor trained on 711K observations under a strict anti-circularity temporal contract ($t \to t+1$).
- **📊 Quantile Uncertainty Intervals ($P_{10}$–$P_{90}$):** Dual quantile heads estimating conservative floor ($P_{10}$), median ($P_{50}$), and optimistic ceiling ($P_{90}$) demand bounds without Gaussian assumptions.
- **💡 Sub-15ms TreeSHAP Explainability:** Exact Shapley value attributions revealing the top positive and negative drivers behind every recommendation.
- **🚗 Travel Time Friction & Realized Net Profit:** Models operating minutes lost during transit ($\frac{60 - T_{\text{travel}}}{60}$) and fuel costs via OSRM street network routing.
- **🎯 Multi-Factor STAY vs. MOVE Decision Engine:** Categorizes recommendations into **`STAY HERE`**, **`CONSIDER`**, or **`RECOMMENDED MOVE`** with plain-English business rationales.
- **🗺️ Interactive Modern Dashboard:** Built with Leaflet, CARTO Voyager basemap, dynamic H3 demand heatmaps, Google Maps-style route casings, and data freshness tracking.

---

## 🏗️ System Architecture

```text
                    REAL-WORLD CONTEXTUAL SIGNALS
    ┌──────────────────────┬──────────────────────┬──────────────────────┐
    │     Live GPS Point   │   Open-Meteo Weather │ OSM Infrastructure   │
    │      [real_live]     │      [real_live]     │ 35.8K POIs [periodic]│
    └──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
               │                      │                      │
               ▼                      ▼                      ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                     FASTAPI BACKEND MICROSERVICE                   │
    │  • H3 Res-8 Hexagon Grid Resolution & Spatial Polygon Filtering    │
    │  • Candidate Generation (≤ Search Radius km, Water Pruning)        │
    │  • 40-Feature Real-Time Assembly & Strict Schema Validation        │
    │  • LightGBM ML Inference (Point Estimate + P10/P90 Quantiles)      │
    │  • TreeSHAP Exact Feature Attribution Decomposition                │
    │  • OSRM Road Network Distance & Driving Duration                   │
    │  • Margin Economics & Next-Hour Realized Uplift Modeling           │
    │  • Strategic STAY vs. MOVE Decision Classification                 │
    └─────────────────────────────────┬──────────────────────────────────┘
                                      │
                                      ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                      LEAFLET DECISION DASHBOARD                    │
    │  • Strategic Guidance Action Banner & Plain-English Rationale      │
    │  • CARTO Voyager Street Map + OSRM Driving Path & Pins             │
    │  • Cellular H3 Demand Heatmap & Provenance Freshness Table         │
    │  • Recommendation Cards with TreeSHAP Insights & Uncertainty Bounds│
    └────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
MOVIGO-ENG/
├── config/
│   └── settings.py              # Central configuration & policy thresholds
├── dashboard/
│   ├── index.html               # Frontend dashboard layout
│   ├── style.css                # Modern UI styles & decision banners
│   └── app.js                   # Map rendering, OSRM routing & API integration
├── data/
│   ├── raw/                     # Raw OSM nodes and weather archives
│   ├── interim/                 # Intermediary spatial joins
│   └── processed/
│       ├── static_features.parquet # 35K OSM POIs mapped to 817 H3 cells
│       └── feature_store.parquet   # 711,750 historical spatial observations
├── docs/                        # Architecture & engineering specifications
├── models/
│   └── demand_model.pkl         # Trained LightGBM bundle (Point + P10/P90 + TreeSHAP)
├── src/
│   ├── api/
│   │   ├── app.py               # FastAPI entrypoint & recommendation routes
│   │   ├── feature_assembler.py # 40-feature vector assembly pipeline
│   │   ├── location.py          # Coordinate to H3 cell resolution
│   │   ├── recommender.py       # ML wrapper, Quantiles, SHAP, and Economics
│   │   ├── spatial_filter.py    # Bounding box & water body exclusion
│   │   └── providers/           # Live providers (Weather, Traffic, Places, Events)
│   ├── data/
│   │   ├── build_all_datasets.py# End-to-end dataset builder
│   │   ├── osm_ingestion.py     # OpenStreetMap Overpass extraction
│   │   ├── static_features.py   # Spatial aggregation into H3 cells
│   │   └── pipeline_version.py  # Provenance manifest tracking
│   └── models/
│       └── train_demand_model.py# LightGBM & Quantile model training pipeline
├── tests/
│   ├── test_api.py              # 11 FastAPI & Recommendation unit/contract tests
│   └── test_data_pipeline.py    # 5 Data contract, temporal, & business logic tests
├── .env.example                 # Environment configuration template
├── Dockerfile                   # Production Docker container definition
├── docker-compose.yml           # Multi-container orchestration
├── README.md                    # Project overview & quickstart
├── RESUME.md                    # In-depth interview mastery guide & 25+ Q&A
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11 or higher
- `pip` package manager
- (Optional) Docker & Docker-Compose

### 2. Clone & Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd MOVIGO-ENG

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional API keys like HERE, Google Places, or Eventbrite can be added to `.env`. If omitted, the system seamlessly operates in zero-key live mode using Open-Meteo, OpenStreetMap, and validated diurnal proxies).*

### 4. Build Datasets & Train Models
```bash
# Step 1: Ingest OSM POIs and build unified 711K-row feature store
python3 src/data/build_all_datasets.py

# Step 2: Train Champion LightGBM Regressor and P10/P90 Quantile Heads
python3 src/models/train_demand_model.py
```

### 5. Launch Application
```bash
# Start FastAPI backend server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to:
- **Interactive Dashboard:** `http://localhost:8000/`
- **Swagger Interactive API Docs:** `http://localhost:8000/docs`

---

## 🧪 Running Automated Tests

Run the complete 16-suite test pipeline covering API endpoints, spatial water exclusion, temporal anti-circularity, and qualitative business logic:

```bash
pytest tests/ -v
```

**Expected Output:**
```text
============================== 16 passed in 1.45s ==============================
```

---

## 📡 REST API Reference

### `POST /v1/recommendations/live`
Generate real-time demand, profit, and strategic STAY/MOVE recommendations for a given GPS coordinate.

**Request:**
```json
{
  "latitude": 16.5062,
  "longitude": 80.6480,
  "vendor_category": "food",
  "search_radius_km": 2.0,
  "top_n": 3
}
```

**Response Summary:**
```json
{
  "request_id": "rec_20260903160000",
  "decision": {
    "verdict": "CONSIDER_MOVE",
    "action": "CONSIDER MOVE",
    "headline": "Consider Relocating to #1 (0.838 km, ~3 min)",
    "decision_reason": "Moderate profit increase (+₹121/hr, +5.2%) available within 3 min drive."
  },
  "current_estimated_demand": 38,
  "current_expected_profit_inr": 2316.39,
  "recommendations": [
    {
      "rank": 1,
      "h3_cell": "88619aa6c3fffff",
      "distance_km": 0.838,
      "estimated_travel_time_min": 3,
      "expected_customers": 40,
      "prediction_interval": {
        "p10": 26,
        "p50": 36,
        "p90": 42,
        "uncertainty_level": "MODERATE"
      },
      "expected_profit_inr": 2436.93,
      "realized_next_hour_profit_inr": 2074.87,
      "profit_improvement_inr": 120.54,
      "decision_action": "CONSIDER",
      "top_drivers": [
        "Office & commercial density (+16.5 cust/hr)",
        "Daily temporal cycle (+6.9 cust/hr)"
      ]
    }
  ]
}
```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
# GeoDemand
