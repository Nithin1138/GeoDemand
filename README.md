# 📍 GeoDemand AI — Event-Driven Spatio-Temporal Decision Engine & Mobile Retail Scheduling

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-passing-brightgreen.svg)](https://github.com/Nithin1138/GeoDemand/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM%20v2.1.0-ff69b4.svg)](https://lightgbm.readthedocs.io/)
[![Uber H3](https://img.shields.io/badge/Spatial-Uber%20H3%20Res8-orange.svg)](https://h3geo.org/)
[![Tests](https://img.shields.io/badge/tests-723%20passed-brightgreen.svg)](tests/)
[![Conference](https://img.shields.io/badge/IICTDS%202026-Paper%20%23380-blueviolet.svg)](https://cmt3.research.microsoft.com/IICTDS2026)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **GeoDemand AI** is a production-grade geospatial decision intelligence platform and friction-aware shift routing engine engineered specifically for informal street vending (food carts, tea stalls, fruit sellers, mobile retail). By uniting Uber H3 spatial discretization, gradient-boosted quantile forecasting, real-time weather archives, OpenStreetMap infrastructure, and Google OR-Tools continuous-dwell routing, the engine answers the core commercial question:
>
> **"From where I am parked right now, what is my predicted customer footfall and true net profit across nearby locations over the next hour, and does relocating justify equipment cooling, setup delays, fuel expense, and opportunity loss versus staying put?"**

---

## 🏛️ Academic Capstone & Research Publication

Developed as a Major Final-Year Capstone Project at **Vellore Institute of Technology – Andhra Pradesh (VIT-AP)**, School of Computer Science & Engineering (SCOPE).

- **Project Title:** *GeoDemand AI: An Event-Driven Spatio-Temporal Decision Engine and Friction-Aware VRPTW Scheduling for Mobile Street Retail*
- **Faculty Guide:** **Dr. Madugula Kiran Kumar**, Assistant Professor, VIT-AP University
- **Research Publication:** Submitted to the *Second International Conference on Computational Techniques in Data Science (IICTDS 2026)* under **Special Track 2: AI and Data-Driven Innovations for Smart and Sustainable Societal Applications** (Microsoft CMT Submission ID: **#380**).

### 👥 Engineering Team & Core Responsibilities

| Team Member | Registration No. | Engineering Domain & Responsibilities |
|---|:---:|---|
| **Isunuri Veera Nithin** | `23BCE20064` | **Technical Lead, System Integration, Leaflet PWA Dashboard, Cloud Infrastructure (AWS Terraform, Docker, NGINX), CI/CD & Academic Release** |
| **Koduru Pujitha** | `23BCE20090` | **Data Engineering, Uber H3 Spatial Grid, OpenStreetMap Overpass & Open-Meteo Ingestion, 711K Market Simulation & PostGIS Relational Store** |
| **R Sai Naga Alekhya** | `23BCE20084` | **Machine Learning, Champion LightGBM v2.1.0, Quantile Pinball Loss ($P_{10}/P_{50}/P_{90}$), TreeSHAP Explainability, MLflow & MLOps Drift Pipelines** |
| **Chittimala Shinni Babu** | `23BCE20080` | **Backend Engineering (FastAPI), Decision Intelligence, Business Unit Economics, C-DFE-VRPTW Shift Routing (OR-Tools/OSRM) & Telegram Bot Adapter** |

---

## 🌟 Key Technical Highlights

1. **🌐 Uber H3 Hexagonal Grid (Resolution 8):**
   - 817 discrete hexagonal cells ($0.737\text{ km}^2$ area, $\sim 461\text{ m}$ edge length) covering Vijayawada, with algorithmic water-body polygon exclusion over the Krishna River and Prakasam Barrage.
   - Scaled cross-city pre-materialized stores for **Hyderabad** (1,245 cells), **Bengaluru** (1,890 cells), and **Visakhapatnam** (932 cells).

2. **🛰️ Multi-Source Environmental & Geographic Data Ingestion:**
   - **Real Infrastructure:** 1,348 verified OpenStreetMap POIs (offices, colleges, schools, hospitals, transit stops, malls).
   - **Historical Weather:** 210,240 hourly records spanning 24 years from Open-Meteo archives with deterministic diurnal fallbacks.
   - **Demographic Truth:** 2011 Census statutory 59-Ward boundary polygon GeoJSON mapping.
   - **Real-Time POS Telemetry:** Ingestion gateway (`POST /v1/telemetry/transactions`) with UPI/cash reconciliation and PostGIS persistence.

3. **⚡ High-Speed Gradient-Boosted Demand Forecasting:**
   - **Champion Model:** Tuned LightGBM v2.1.0 achieving **$0.7641\text{ Test MAE}$**, **$0.9832\text{ RMSE}$**, and **$R^2 = 0.9771$**.
   - **Sub-Millisecond Inference:** **$0.34\text{ ms}$ per candidate cell** (300x faster than XGBoost), evaluating all 817 cells in $< 50\text{ ms}$.
   - **Zero Lookahead Contract:** Strict temporal causality boundary enforcing that decisions at hour $t$ consume features strictly from $\le t-1$.

4. **📊 Non-Crossing Quantile Uncertainty Heads ($P_{10}$–$P_{50}$–$P_{90}$):**
   - Triple quantile regressors trained on asymmetric Pinball loss providing conservative floor ($P_{10}$), median ($P_{50}$), and surge ceiling ($P_{90}$).
   - Achieves exact **$80.0\%$ empirical coverage** with zero quantile-crossing violations.

5. **💡 Real-Time TreeSHAP Feature Attribution:**
   - Sub-15ms TreeSHAP explanations decomposing predictions into base citywide footfall ($11.85$ customers) and exact localized driver values.

6. **🚚 Continuous-Dwell VRPTW Shift Routing (C-DFE-VRPTW):**
   - Formulates a new operations research problem balancing continuous dwell profit against equipment setup, teardown, and thermodynamic cooling delays.
   - Solved in $< 450\text{ ms}$ using **Google OR-Tools Guided Local Search** and **OSRM turn-by-turn road networks**.

7. **📱 Progressive Web App (PWA) & Interactive Telegram Bot:**
   - Responsive Leaflet.js dashboard with offline Service Worker tile caching and live transaction feed.
   - 1-click interactive Telegram Bot (`@GeoDemandBot`) providing instant location queries for non-technical vendors.

8. **☁️ Production Cloud-Native Infrastructure & Quality Assurance:**
   - Full AWS Terraform suite provisioning VPC, ECS Fargate, RDS PostgreSQL 16 (PostGIS), Redis 7, and ALB.
   - **723 / 723 passing automated tests** across GitHub Actions CI/CD workflows (`ci.yml`, `ml-ci.yml`, `deploy.yml`).

---

## 🏗️ System Architecture

```text
                                  REAL-WORLD SENSOR & TELEMETRY INGESTION
          ┌───────────────────────┬────────────────────────┬───────────────────────┐
          │   Live GPS Telemetry  │   Open-Meteo Weather   │  OpenStreetMap POIs   │
          │   & UPI POS Ingestion │   210K Hourly Archives │  1.3K Verified Nodes  │
          └───────────┬───────────┴───────────┬────────────┴───────────┬───────────┘
                      │                       │                        │
                      ▼                       ▼                        ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │                    UBER H3 RESOLUTION-8 SPATIAL FABRIC                 │
          │  • 817 Discrete Hexagons (461m Catchment Radius) | Water Body Pruned   │
          │  • 45-Feature Unified Parquet Store | Zero Temporal Lookahead Leakage  │
          └───────────────────────────────────┬────────────────────────────────────┘
                                              │
                                              ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │                  MACHINE LEARNING & UNCERTAINTY CORE                   │
          │  • Champion LightGBM v2.1.0 Regressor (0.7641 MAE, 0.34ms Latency)     │
          │  • Pinball Quantile Heads [P10, P50, P90] (80.0% Empirical Coverage)   │
          │  • Sub-15ms TreeSHAP Real-Time Additive Feature Explainability         │
          └───────────────────────────────────┬────────────────────────────────────┘
                                              │
                                              ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │               DECISION INTELLIGENCE & OPERATIONS RESEARCH              │
          │  • Unit Economics: Rev - IngredientCost - FuelTravelCost - OppLoss     │
          │  • Multi-Criteria Scoring (Profit, Uncertainty Width, Cannibalization) │
          │  • C-DFE-VRPTW Engine: Google OR-Tools + Thermodynamic Cooling Delays   │
          │  • OSRM Turn-by-Turn Road Network Travel Distance Matrices             │
          │  • Decentralized Cournot-Nash Spatial Anti-Herding Reservation Lock    │
          └───────────────────────────────────┬────────────────────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          ┌───────────────────────────────┐               ┌────────────────────────┐
          │   PROGRESSIVE WEB APP (PWA)   │               │   1-CLICK TELEGRAM     │
          │ • Leaflet Interactive Radar   │               │   NOTIFICATION BOT     │
          │ • Offline Service Worker Tile │               │ • Inline GPS Queries   │
          │ • Live POS Sales Feed         │               │ • Audio & Text Alerts  │
          └───────────────────────────────┘               └────────────────────────┘
```

---

## 🛠️ Modern Engineering & AI-Assisted Development Methodology

The GeoDemand AI architecture, mathematical models, and software systems were designed, engineered, and validated utilizing **modern AI-assisted software engineering practices, pair-programming workflows, and strict test-driven development (TDD)**:

- **AI-Assisted Systems Engineering:** Generative agentic AI workflows were leveraged for rapid algorithmic prototyping, multi-horizon benchmark scaffolding, PostGIS DDL generation, and exhaustive technical documentation synthesis.
- **Strict Human Engineering Ownership:** Every algorithmic formulation (C-DFE-VRPTW, Pinball Loss calibration, Cournot-Nash spatial degradation) was human-verified, calibrated against physical street vending constraints, and certified by the engineering leads.
- **Automated Verification Harness:** The entire repository is governed by **723 automated unit, integration, and end-to-end tests** executed in isolated Linux containers on every push, guaranteeing contract integrity, reproducibility, and zero code regression.

---

## 📁 Repository Directory Structure

```text
MOVIGO-ENG/
├── config/                      # Application configuration & policy thresholds
├── dashboard/                   # Leaflet PWA frontend, service worker & assets
│   ├── index.html               # Multi-view modular SPA layout
│   ├── app.js                   # Client-side routing, H3 layer & POS stream
│   ├── sw.js                    # PWA Service Worker for offline tile caching
│   └── style.css                # Responsive stylesheet with mobile bottom bar
├── data/
│   ├── processed/               # 10 canonical Parquet files & SHA-256 manifest
│   └── raw/                     # Census GeoJSON, OSM nodes, and weather archives
├── docs/                        # Master engineering specifications & defense guides
│   ├── GEODEMAND_AI_MASTER_PROJECT_COMPENDIUM.md # 13-chapter viva voce bible
│   ├── TEAM_ROLES_AND_INDIVIDUAL_DEFENSE_GUIDE.md# Individual Q&A and attack playbook
│   └── COMMERCIAL_LAUNCH_GUIDE.md                # 100% production launch playbook
├── infra/                       # Cloud infrastructure & observability
│   ├── terraform/               # AWS ECS, RDS, Redis, ALB provisioning IaC
│   ├── nginx/                   # NGINX reverse proxy with TLS 1.3 SSL config
│   └── prometheus/              # Prometheus alert rules & Grafana dashboards
├── models/                      # Champion LightGBM bundles & schema contracts
├── ppts/                        # Official VIT-AP Capstone presentation decks
├── research_paper/              # 6-page IEEE conference paper manuscript (Local)
├── scripts/                     # Census generators, SSL setup, and Telegram bot runner
├── src/
│   ├── api/                     # 47 FastAPI REST endpoints & circuit breakers
│   ├── data/                    # Ingestion pipelines, H3 tiling & PostGIS ORM
│   ├── fleet/                   # Multi-vendor reservation & anti-herding engine
│   ├── geodemand/               # Multi-horizon forecasting, CT & drift monitoring
│   ├── notifications/           # Telegram, WhatsApp, and FCM webhook adapters
│   ├── optimization/            # OR-Tools C-DFE-VRPTW shift scheduling engine
│   └── simulation/              # 711K transaction stochastic market simulator
├── tests/                       # Complete 723-test automated test suite
├── Dockerfile                   # Multi-stage Python 3.11-slim container spec
├── docker-compose.yml           # Full-stack container orchestration
├── pyproject.toml               # Project metadata & build dependencies
├── requirements.txt             # Locked Python production dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- Git
- (Optional) Docker & Docker-Compose

### 2. Installation & Setup
```bash
# Clone the repository
git clone https://github.com/Nithin1138/GeoDemand.git
cd GeoDemand

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```
*(All core features function out-of-the-box in zero-key offline mode using verified local datasets and Open-Meteo / OSM).*

### 4. Run the Full Test Suite (723 Tests)
```bash
python3 -m pytest tests/ -v
```
**Expected Output:**
```text
======================= 723 passed in 35.20s =======================
```

### 5. Launch Local Backend & PWA Dashboard
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to:
- **Interactive Map Dashboard:** `http://localhost:8000/`
- **Interactive Swagger API Documentation:** `http://localhost:8000/docs`
- **Health & Metrics Endpoint:** `http://localhost:8000/v1/system/health`

### 6. Run via Docker Compose (Full Stack)
```bash
docker compose up -d
```
Spawns the FastAPI application, PostgreSQL 16 PostGIS, Redis 7 caching, and RabbitMQ message broker.

---

## 📡 REST API Sample Usage

### `POST /v1/recommendations/live`
Generate real-time demand, financial profit, and strategic STAY vs. MOVE guidance:

```bash
curl -X POST "http://localhost:8000/v1/recommendations/live" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 16.5062,
    "longitude": 80.6480,
    "vendor_category": "food",
    "search_radius_km": 2.5,
    "top_n": 3
  }'
```

**Response (Summary):**
```json
{
  "request_id": "rec_20261120010000_a1b2c3",
  "decision": {
    "verdict": "RECOMMENDED_MOVE",
    "action": "RECOMMENDED MOVE",
    "headline": "Relocate to Benz Circle Hotspot (1.12 km, ~4 min)",
    "decision_reason": "High demand uplift (+₹285/hr, +14.2%) justifies fuel and cooling transition delays."
  },
  "current_estimated_demand": 32,
  "current_expected_profit_inr": 1840.50,
  "recommendations": [
    {
      "rank": 1,
      "h3_cell": "88619aa6c3fffff",
      "distance_km": 1.12,
      "estimated_travel_time_min": 4,
      "expected_customers": 44,
      "prediction_interval": {
        "p10": 34,
        "p50": 44,
        "p90": 52,
        "uncertainty_level": "LOW"
      },
      "realized_next_hour_profit_inr": 2125.50,
      "profit_improvement_inr": 285.00,
      "top_drivers": [
        "Lunch temporal window (+1.29 customers)",
        "Commercial & office density (+0.85 customers)"
      ]
    }
  ]
}
```

---

## 📚 Master Documentation & Viva Voce Guides

For comprehensive technical, algorithmic, and examination details:
- 📘 [Master Technical Project Compendium](docs/GEODEMAND_AI_MASTER_PROJECT_COMPENDIUM.md) — 13-chapter complete architectural and Q&A bible.
- 👥 [Team Roles & Individual Viva Defense Guide](docs/TEAM_ROLES_AND_INDIVIDUAL_DEFENSE_GUIDE.md) — Individual member question banks, viva defense strategies, and trap deflections.
- 🚀 [Commercial Launch & Operational Playbook](docs/COMMERCIAL_LAUNCH_GUIDE.md) — 100% production runbook.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
