# GeoDemand AI
### A Production-Grade Decision Intelligence Platform for Mobile Vendor Demand Forecasting & Location Optimization

*(Internally powers "MOVIGO Demand Intelligence" — built as a standalone, reusable portfolio project not tied to any single startup)*

---

## 0. Document Purpose

This is the single source of truth for the project: the problem, the architecture, every design decision and the trade-off behind it, the data strategy, the ML system, the engineering stack, the roadmap, and how to talk about it in an interview. Everything below reflects the **frozen v1.1 architecture** — the point past which further changes trade implementation time for marginal design gains, not a point where the design is unfinished.

---

## 1. Problem Statement

Mobile vendors (food trucks, fruit sellers, salon vans, repair trucks) decide where to operate using guesswork, habit, or word-of-mouth. This causes:

- Fuel wasted driving to low-demand areas
- Unsold inventory / direct economic loss
- Accidental clustering (3–5 vendors converging on the same spot)
- No way to anticipate demand spikes (events, weather, holidays) before they happen

**Core question the system answers:**
> Given a vendor's category, current location, and inventory, where should they go in the next hour to maximize expected profit — and why?

This is a genuine ML + optimization problem, not one where ML is forced in: demand is a function of many weak, nonlinear, interacting signals (time, weather, POIs, events, competition) that no lookup table or static rule can capture well.

---

## 2. Business Value

| Stakeholder | Value |
|---|---|
| Vendor | Higher revenue per shift, less wasted fuel, data-backed confidence in location decisions |
| Platform (MOVIGO) | Core differentiator vs. static-listing competitors (Google Maps) and delivery apps (Zomato/Swiggy) that don't do real-time mobile-vendor routing |
| Aggregated data buyers | Anonymized hyperlocal demand patterns are sellable to FMCG, event planners, smart-city initiatives |

---

## 3. Design Principles (non-negotiable, drive every downstream decision)

1. **One predictive ML model only.** Everything else derived downstream.
2. **Prediction and decision-making are separate systems.** The model predicts a measurable quantity; business logic converts it to money and rank.
3. **Hybrid data**: real contextual data where it's free/reliable, clearly-labeled simulation where no real target exists.
4. **Modular, production-shaped architecture** — not a notebook.
5. **Every design choice must be defensible in an interview**, including its trade-offs.

---

## 4. Architecture (v2.1 — Live Recommendation System)

```
[LIVE REQUEST] GPS (Lat, Lng) + Vendor Profile + Radius
        │
        ▼
[LOCATION CONTEXT LAYER]
  GPS → H3 Cell (Res 8) → H3 k-ring candidate expansion (30–50 candidate cells)
        │
        ▼
[REAL-TIME DATA PROVIDER LAYER]
  ├── Weather: Open-Meteo current + hourly forecast (real_live / fallback: synthetic_fallback)
  ├── Static POIs: OpenStreetMap via Overpass / static_features.parquet (real_periodic)
  ├── Competition: OSM POI density / competition.parquet (real_periodic / simulated)
  ├── Traffic: HERE Traffic API / time-of-day proxy (real_live / synthetic_fallback)
  ├── Events: Stochastic event model / Eventbrite adapter (simulated / real_live)
  ├── Calendar/Holidays: Python holidays AP subdiv (real_offline)
  └── Fuel: Configured INR/Litre & km/L (real_periodic)
        │
        ▼
[FEATURE ASSEMBLER]
  Constructs 40-feature inference vector matching feature_store.parquet schema
  Generates data_freshness dictionary with explicit provenance per signal
        │
        ▼
[DEMAND PREDICTION MODEL]
  LightGBM Regressor (Test MAE: 0.91 customers/hr)
  Trained on next-hour temporal contract (features at t → target at t+1)
  Outputs expected_customer_count + uncertainty interval (p10, p50, p90)
        │
        ▼
[BUSINESS METRICS LAYER]
  Revenue = customers × AOV
  COGS = Revenue × variable_cost_rate
  Operating Cost = COGS + (fixed_cost / 13) + fuel_cost
  Profit = Revenue - Operating Cost
        │
        ▼
[RANKING & EXPLAINABILITY ENGINE]
  Score = 0.60×profit_norm + 0.25×demand_norm - 0.10×distance_norm - 0.05×competition_norm
  Generates top_drivers + human-readable rationale
        │
        ▼
[FASTAPI SERVICE & DASHBOARD]
  POST /v1/location/context
  POST /v1/recommendations/live
  GET  /v1/system/data-health
  Interactive Leaflet Map + Top-5 Cards + Provenance Badges
```

---

## 5. Design Decisions & Trade-offs (the part interviewers actually probe)

### 5.1 Predict "Expected Customer Count," not "Demand Score" or Revenue directly
- **Why**: gives a concrete, interpretable regression target with standard metrics (MAE/RMSE/R²). Business metrics (demand score, revenue, profit) are *derived* afterward via a documented formula, not learned — this mirrors how real production systems separate prediction from business logic.
- **Trade-off**: the model doesn't directly optimize for profit; if margin varies wildly by item, the derived-profit step needs its own validation. Accepted for v1 — profit derivation is a simple, auditable multiplication, not a hidden model.

### 5.2 H3 hexagonal spatial indexing over raw lat/lon
- **Why**: consistent spatial aggregation, efficient neighbor queries, industry-standard (Uber built it, Grab/Swiggy-style platforms use equivalent grids). Raw lat/lon produces irregular, unaggregatable buckets.
- **Trade-off**: resolution choice (8) is a tuning knob — too coarse loses granularity, too fine sparsifies data per cell. Resolution 8 (~0.7 km²) chosen as a "walkable neighborhood" scale appropriate for foot-traffic-driven vendor demand.

### 5.3 Fixed 1-hour prediction window (True Next-Hour Forecasting, not Nowcasting)
- **Why**: the system supports a forward-looking dispatch / relocation *decision*. A vendor deciding at `10:00` needs to know demand for the `11:00–12:00` window, not the demand occurring during the current hour.
- **Contract Enforcement**: Every training and simulation record explicitly defines:
  - `decision_timestamp`: time $t$ when context (weather forecast, calendar, static POIs, competition) is evaluated.
  - `target_timestamp`: time $t+1\text{h}$ when the 1-hour operating window begins.
  - `expected_customer_count`: footfall occurring during $[t+1\text{h}, t+2\text{h})$.
- **Trainable Matrix**: All IDs (`vendor_id`, `h3_cell_id`) and timestamps (`decision_timestamp`, `target_timestamp`) are strictly excluded from trainable features to guarantee pure generalization across POI and environmental attributes. Current vendor location and travel time are handled exclusively by the downstream dispatch optimizer.

### 5.4 Static / Dynamic feature split
- **Static** (rarely change): population density, road density, office/school/hospital/mall/restaurant/park/bus-stop counts — per H3 cell.
- **Dynamic** (change hourly/daily): weather, holidays, events, time, competitor activity, traffic proxy, fuel-price proxy.
- **Why**: mirrors how production feature stores are actually organized (different refresh cadences, different validation rules, different storage/versioning strategy).

### 5.5 Market Simulation Engine — anti-circularity is the most important decision in the project
- **The trap**: if the simulator generates `customer_count` as a simple linear formula of the exact features fed to the model (`0.4*weather + 0.3*office_count + ...`), the model just learns to invert the formula. Reported accuracy would reflect "can XGBoost do algebra," not "can it model demand."
- **The fix, actually implemented**:
  - **Hidden/latent variables never exposed to the model**: per-vendor `hidden_skill_factor` (reputation, friendliness, word-of-mouth), per-cell `cell_reputation` (visibility, informal foot-traffic quality). These genuinely drive simulated demand but the model has to work around not seeing them.
  - **Interaction terms, not pure additive effects**: e.g. `rain × office_density`, `weekend × mall_density` — the model has to discover these, not just read off exposed features.
  - **Nonlinear saturation** (sigmoid-shaped demand curve): demand doesn't grow unboundedly with POI counts.
  - **Heteroscedastic noise**: variance scales with expected demand (`noise_std ∝ √expected`), matching how real count data behaves, rather than constant-variance Gaussian noise.
- **Honesty in the README**: explicitly states model performance here validates the *pipeline*, not real-world accuracy, since ground truth is simulated. This is disclosed, not hidden — a reviewer who understands the limitation will trust the project more, not less, for saying so.

### 5.6 Optimization Engine — scoped correctly as ranking, not solved as full optimization
- **What v1 actually is**: `Score = w1·Demand + w2·Profit − w3·TravelCost − w4·Competition + w5·VendorPreference + w6·Confidence`, weights configurable, applied to rank candidate H3 cells. This is ranking, not constrained optimization.
- **What "Recommended Route" would require**: true multi-stop routing under constraints (fuel budget, travel time, working hours) — a genuine OR problem, not a ranking problem.
- **Decision**: v1 removes "Recommended Route" from scope and ships Top-5 ranked locations only. v2 adds OR-Tools for actual route optimization over the top-5. This keeps v1 scope honest and shippable.

### 5.7 Confidence — statistically grounded, not a magic number
- **The trap**: showing "Confidence: 92%" with no defined origin isn't defensible.
- **The fix**: quantile regression (P10/P50/P90 predictions via LightGBM/CatBoost quantile loss). Confidence is derived from prediction-interval width — narrower interval, higher stated confidence. Documented, reproducible, explainable in an interview.

### 5.8 Traffic and fuel price — simulated proxies, not live paid APIs
- **Why**: live traffic APIs (Google/Mapbox) are paid and rate-limited — a common trap where students design systems around dependencies they can't actually run. Traffic is simulated as a function of time-of-day + road density (a static feature) + weekday/weekend. Fuel price is a static/periodically-updated value.
- **Trade-off**: less "real-time," but the project stays fully reproducible and free to run indefinitely — the right trade for a portfolio system.

### 5.9 Explainability generated from SHAP, not hand-written templates
- **Why**: a hardcoded "high office density" reason string is not defensible as "explainable AI." SHAP top-contributor values are extracted per prediction and templated into a sentence (e.g. *"recommended because office density and rain increase expected demand, while competition remains low"*) — the explanation is tied directly to what the model actually weighted.

### 5.10 Authentication kept intentionally lightweight
- **Why**: full OAuth/refresh-token/RBAC is weeks of work that doesn't demonstrate ML/data engineering skill — it demonstrates auth-framework plumbing. API-key auth is used, documented as "production-pattern, scoped for portfolio," so reviewers understand the trade-off was made deliberately, not out of ignorance.

### 5.11 Time-based cross-validation, not random splits
- **Why**: random train/test splits leak autocorrelated demand patterns across the split and inflate reported accuracy — a well-known forecasting-project mistake. Uses forward-chaining validation (e.g., Jan–Sep train / Oct validate / Nov test) or rolling-window validation instead, which reflects how the model would actually perform in production.

### 5.12 Explicit temporal-leakage guard
- **Why**: it's easy to accidentally let a "dynamic" feature (e.g. competitor count) reflect information from *after* the prediction window rather than before it. Every feature carries an explicit `decision_timestamp`, and the feature-store validation step asserts no target-time or future data is used in feature calculation. This is enforced in code, not just documented as a rule.

---

## 6. Data Strategy

### 6.1 Real data sources & Provenance Classification
| Source | Data | Provenance Tag | Notes |
|---|---|---|---|
| OpenStreetMap / Overpass API | POI counts per H3 cell | `real:osm` / `real-proxy:osm_synthetic_fallback` | Free, live queryable with graceful fallback |
| Open-Meteo | Historical + forecast weather | `real:open-meteo` / `real-proxy:synthetic_fallback` | Free archive API with MD5-seeded diurnal fallback |
| Public holiday calendar | Fixed + regional holidays | `real:holidays_lib` | Python holidays library (IN) |
| Local Events | Hyperlocal demand spikes | `simulated:sparse_stochastic` | Sparse stochastic festival, wedding, fest flags |
| Competition | Hyperlocal competitor density | `hybrid:h3_spatial_adjacency` | Real H3 grid_disk geometry over vendor profiles |

### 6.2 Simulated data (clearly labeled)
- **Vendor sales / customer counts**: no public dataset exists for informal mobile-vendor sales. Generated via the Market Simulation Engine (Section 5.5), grounded in real contextual features, with the anti-circularity safeguards documented above.
- Every simulated dataset ships with a JSON metadata sidecar recording: version hash, generation timestamp, full simulation parameters, seed, row/vendor/cell counts, and explicit provenance tags.

### 6.3 Dataset versioning
```
dataset_v<hash>.parquet
dataset_v<hash>.meta.json
```
Metadata includes simulation parameters, seed, source versions, generation timestamp — every run is fully reproducible from its metadata alone.

## 6.6 Pipeline Restructure — 12 Independent Datasets, Not One Monolithic Table

The original v1 simulator (`market_simulator.py`) wrote one flat table. That doesn't scale and can't be independently versioned, tested, or reused. Restructured into single-responsibility datasets, each with its own build script under `src/data/`, joined only at the end:

| # | Dataset | File | Type | Provenance Tag |
|---|---|---|---|---|
| 1 | H3 Spatial Grid | `h3_cells.parquet` | 🟢 Real | `real:spatial_indexing` |
| 2 | Static Spatial Features | `static_features.parquet` | 🟢 Real (OSM) / proxy | `real-proxy:osm_synthetic_fallback` |
| 3 | Weather Hourly | `weather_hourly.parquet` | 🟢 Real (Open-Meteo) / proxy | `real-proxy:open_meteo_synthetic_fallback` |
| 4 | Calendar | `calendar.parquet` | 🟢 Real (`holidays` lib) | `real:holidays_lib` |
| 5 | Events | `events.parquet` | 🟡 Simulated, sparse | `simulated:sparse_stochastic` |
| 6 | Competition | `competition.parquet` | 🟡 Hybrid H3 geometry | `hybrid:h3_spatial_adjacency` |
| 7 | Vendor Profiles | `vendor_profiles.parquet` | 🟡 Simulated | `simulated:business_profiles` |
| 8 | Product Catalog | `products.parquet` | 🟡 Simulated | `simulated:product_catalog` |
| 9 | Historical Transactions | `historical_transactions.parquet` | 🟡 Next-Hour Target ($t \to t+1$) | `simulated:market_outcomes` |
| 10 | Feature Store | `feature_store.parquet` | 🟢 Unified ML Train Table | `hybrid:joined_feature_store` |

Datasets 11 (predictions) and 12 (recommendations) are produced later, by the model-serving and optimization layers — not part of the data pipeline itself.

**Feature store join contract, enforced in code, not just documented:**
- `decision_timestamp` ($t$) is the anchor for joining contextual features; `target_timestamp` ($t+1\text{h}$) is the prediction horizon for `expected_customer_count`.
- `LEAKAGE_COLUMNS` (hidden_skill_factor, popularity_score, quality_score, repeat_customer_rate, cell_reputation) are dropped by construction and the join raises `ValueError` if any leak through.
- A null-guard raises if any join produces unmatched rows, catching silent key mismatches instead of training on NaN-filled features.
- `historical_transactions.parquet` decomposes the target into an auditable multiplier chain (`weather_multiplier`, `holiday_multiplier`, `competition_multiplier`, `event_multiplier`, `noise_component`).

## 6.7 A Real Bug Found and Fixed While Wiring This Up (worth knowing for interviews)

While re-running the qualitative validation suite against the restructured pipeline, the "rain near hospital" check started failing/flipping sign inconsistently across runs of *the same code*. Root cause: `weather_client.py`'s synthetic-weather fallback seeded its RNG with Python's built-in `hash(start_date)` — which is **randomized per-process** by default (`PYTHONHASHSEED`), not stable across script invocations. Two separate runs of the identical pipeline were silently generating *different* synthetic rain days, so a diagnostic comparing `historical_transactions.parquet` from one run against `weather_hourly.parquet` from another was comparing mismatched data — it looked exactly like a demand-model bug and wasted real debugging time before the actual cause surfaced.

**Fix**: switched the seed derivation to `hashlib.md5(start_date.encode())`, which is stable across processes and Python versions. Also separately recalibrated the demand-saturation curve (a real, distinct bug: the sigmoid's offset/scale compressed nearly all evening-hour scores into the plateau near the cap, hiding the rain effect the multiplier chain was supposed to produce even once weather was consistent).

After both fixes, on a full-year, single-consistent-run dataset (150 vendors, 711,750 transactions, 92 real rain days included via the monsoon window):

| Check | Result | Effect Size |
|---|---|---|
| Rainy evenings near hospitals → food demand up | ✅ PASS | +43.8% lift |
| Lunch near offices → meal demand up | ✅ PASS | +79.7% lift |
| Holidays → overall demand up | ✅ PASS | +27.2% lift |
| Weekends → salon demand up | ✅ PASS | +24.3% lift |
| More competitors → lower per-vendor demand | ✅ PASS | -18.4% suppression |

## 6.8 Pipeline Versioning & Manifest

Beyond each individual dataset's own version tag, `src/data/pipeline_version.py` snapshots a **pipeline-level manifest** (`pipeline_manifest_v<hash>.json`) — recording schema version (`v2.1-next-hour-forecast`), every file's SHA-256 hash, row count, column list, and provenance classification. This guarantees full data lineage and reproducibility for model training.

---

## 7. Feature Store Design

**Static features** (per H3 cell, rarely change): population_density, office_count, college_count, school_count, hospital_count, mall_count, restaurant_count, park_count, bus_stop_count, road_density.

**Dynamic features** (per H3 cell × timestamp, change hourly/daily): is_raining, temp_c, is_holiday, is_weekend, event_flag, event_boost_raw, competitor_count (simulated proxy), traffic_proxy (time-of-day × road_density), fuel_price_proxy.

Every feature row carries: `source` (real/synthetic), `as_of_timestamp`, `feature_version`. This is what prevents both silent temporal leakage and silent provenance confusion later.

---

## 8. Market Simulation Engine — Technical Summary

- **Scale**: 150 simulated vendors, 4 categories (food/fruits/salon/repair) with category-specific base demand and POI affinity weights, across 133–817 H3 cells, 365 days, operating hours 8am–9pm → **711,750 hourly observations**.
- **Generative structure**: `customer_count = saturate(base + POI_affinity_score + hour_curve + weather_effect + holiday_effect + weekend_effect + rain×office_interaction + weekend×mall_interaction + event_boost + hidden_skill + hidden_reputation) + heteroscedastic_noise`
- **Validated output distribution**: 4–62 customers/hour, mean ~24.7, clear lunch (12–14h) and evening (18–20h) peaks, no saturation clustering.

---

## 9. Machine Learning

**Models compared**: Linear Regression (baseline), Random Forest, XGBoost, LightGBM, CatBoost.
**Evaluation**: MAE, RMSE, R², evaluated under time-based CV (Section 5.11).
**Tuning**: Optuna, Bayesian hyperparameter search.
**Explainability**: SHAP values per prediction, feeding both the reason-generation layer (5.9) and general feature-importance analysis.
**Uncertainty**: quantile regression (P10/P50/P90) for the confidence layer (5.7).
**Experiment tracking**: MLflow — every run logs params, metrics, SHAP artifacts, and the dataset version hash it trained on.
**Model registry**: MLflow Model Registry, staged (Staging → Production) promotion.

---

## 10. Business Metrics Layer (derived, not trained)

```
Expected Customer Count (model output)
        │
        ▼
Demand Score = normalize(Expected Customer Count)
        │
        ▼
Revenue = Expected Customer Count × Avg Order Value (per category)
        │
        ▼
Profit = Revenue − (Ingredient/Stock Cost + Fuel Cost + Time Cost)
        │
        ▼
Recommendation Score = weighted combination (Section 5.6)
```

---

## 11. Optimization / Decision Intelligence Engine

- **v1**: configurable weighted ranking over candidate H3 cells → Top-5 output.
- **v2 (future)**: OR-Tools constrained route optimization over the Top-5, respecting fuel budget / travel time / working hours.
- **Scenario Simulator**: "what if it rains / a competitor appears / fuel prices rise / I sell tea instead of juice / I start an hour later" — recomputes recommendations instantly by re-running the pipeline with modified feature inputs. No retraining required; this is a pure inference-time re-query.

**Recommendation output** — a ranked table, never bare coordinates:

| Rank | H3 Cell | Expected Customers | Revenue | Profit | Score | Reason |
|---|---|---|---|---|---|---|
| 1 | 886189... | 42 | ₹4,080 | ₹2,350 | 96.2 | High office density, light rain, low competition |

---

## 12. Backend / Serving

- **FastAPI** service with:
  - `POST /predict/demand` — score for a given H3 cell + time window
  - `POST /recommend/zones` — Top-5 ranked recommendations for a vendor category + current position
  - `POST /scenario/simulate` — re-run recommendations under modified conditions
- API-key authentication, response caching for repeated cell/time queries, background job support for batch scoring, versioned API (`/v1/...`).

---

## 13. MLOps & Infrastructure

- **MLflow**: experiment tracking + model registry.
- **Docker + docker-compose**: `api`, `mlflow`, `postgres` (feature store backend) services.
- **CI/CD**: GitHub Actions — lint/test → build image → push → deploy on merge to main.
- **Cloud**: AWS — S3 (data lake for versioned datasets), ECS Fargate or EC2 (API hosting), RDS or S3-backed MLflow tracking store.
- **Monitoring**: Evidently AI for data drift and prediction drift; drift beyond threshold triggers a retraining job. API latency and system logs tracked alongside model metrics.
- **Retraining strategy**: scheduled + drift-triggered retraining, new model versions promoted through the MLflow registry only after passing validation-set performance gates.

---

## 14. Operations Dashboard (Streamlit)

Built as an **operations console**, not a static report:
- Interactive map with live demand heatmap
- Top-5 recommended locations with reasons
- Expected customers / revenue / profit / recommendation score per location
- Scenario simulator controls
- System health panel (API latency, last retrain time, drift status)

---

## 15. Software Engineering Practices

Modular architecture (`simulation/`, `features/`, `models/`, `api/`, `optimization/`, `dashboard/`, `tests/`), type hints throughout, structured logging, environment-variable configuration (no hardcoded secrets/paths), unit + integration tests, explicit error handling, data validation at ingestion boundaries.

---

## 16. Repository Structure

```
geodemand-ai/
├── data/
│   ├── raw/            # real API pulls (OSM, weather, holidays)
│   ├── processed/       # cleaned/joined feature tables
│   └── simulated/       # versioned simulated datasets (.parquet + .meta.json)
├── src/
│   ├── simulation/      # h3_grid.py, static_features.py, market_simulator.py
│   ├── features/        # dynamic feature builders, osm_client.py, weather_client.py
│   ├── models/           # training, tuning, SHAP, quantile regression
│   ├── optimization/     # ranking engine, scenario simulator, (v2) OR-Tools routing
│   └── api/               # FastAPI app, endpoints, auth, caching
├── dashboard/            # Streamlit operations console
├── tests/                # unit + integration tests
├── config/               # environment configs
├── notebooks/            # EDA only — not where production logic lives
├── docs/                 # this file + architecture diagrams, API spec, etc.
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 17. Project Portfolio Context

| Project | Focus |
|---|---|
| 1. Samsung Regret Intelligence System | Analytics, DWDM, Data Warehouse, NLP, ML, BI |
| 2. Atlikon Data Engineering Platform | Databricks, PySpark, Delta Lake, Medallion Architecture, Incremental ETL |
| 3. **GeoDemand AI** (this project) | H3 Geospatial Analytics, Hybrid Data, Simulation Engineering, ML, Optimization, Decision Intelligence, FastAPI, MLOps, Dashboard |

Together: analytics & warehousing → enterprise data engineering → production-grade ML/decision-intelligence. A coherent three-project story, not three unrelated ones.

---

## 18. Status — What's Actually Built vs. Planned

**Built and tested:**
- Full 10-dataset pipeline (Section 6.6): h3_cells, static_features, weather_hourly, calendar, events, competition, vendor_profiles, products, historical_transactions, feature_store — each independently generatable, versioned, and joined only at the end
- Real weather client (Open-Meteo, no key) with synthetic fallback, source-tagged, **reproducibility bug fixed** (hashlib seed instead of randomized `hash()`)
- Real holiday/festival client (`holidays` library — correctly handles lunar-calendar festivals, no hardcoded dates) — 17 correctly-dated 2025 India holidays, fully real, ran end-to-end
- Real OSM/Overpass POI client built (works outside this sandbox; sandbox blocks the host)
- Vendor Profile Generator with product-mix simulation, validated: tea share 10%→28.5% on rainy evenings, meals share 11%→30% at sunny lunch
- Competition dataset using real H3 spatial adjacency (`h3.grid_disk`) over simulated vendor positions
- Market Simulation Engine v2: full anti-circularity design (hidden variables, interactions, nonlinear saturation, heteroscedastic noise), auditable multiplier decomposition per transaction, category+hour-aware weather effects
- **Feature Store join with enforced leakage guard** (verified by deliberate injection test) and null guard
- **Qualitative Validation Suite** — all 5 checks passing with strong, real effect sizes on the full 711,750-row consistent dataset (Section 6.7)
- Pipeline-level version manifest (`pipeline_version.py`) — hashes every dataset file together so a `feature_store.parquet` can be verified as coming from one consistent run

**Not yet built:**
- Full-grid real OSM data collection run (client works, hasn't been run against the whole 817-cell grid — needs real network access)
- Model training pipeline (LightGBM/XGBoost/CatBoost + quantile regression + SHAP + MLflow)
- Optimization/ranking engine + scenario simulator
- FastAPI service
- Docker/CI-CD/AWS deployment
- Evidently monitoring + retraining trigger
- Streamlit dashboard

---

## 6.4 Real Data Clients — What's Real, What Needs a Key, What Runs Where

| Source | File | Key needed? | Runs in this sandbox? |
|---|---|---|---|
| Open-Meteo (weather) | `src/features/weather_client.py` | **No** | No — sandbox blocks `api.open-meteo.com`; falls back to tagged synthetic |
| OpenStreetMap Overpass (POIs) | `src/features/osm_client.py` | **No** (rate-limited, be polite) | No — sandbox blocks `overpass-api.de`; raises a clear error |
| `holidays` library (festivals) | `src/features/holiday_client.py` | **No** — offline computed | **Yes** — this ran for real, 17 correctly-dated 2025 India holidays including lunar festivals |
| OpenWeatherMap (optional alt.) | `weather_client.py::OpenWeatherClient` | **Yes** — `OPENWEATHER_API_KEY` env var | N/A |
| Geoapify (optional alt.) | `osm_client.py::GeoapifyClient` | **Yes** — `GEOAPIFY_API_KEY` env var | N/A |

To run the real clients: clone the repo to a machine/environment with normal outbound internet access, set any optional keys as environment variables (never hardcode them), and run e.g. `python3 src/features/osm_client.py`. The clients cache results to `data/raw/`, so a full-grid POI pull only needs to run once.

**Population/Census — flagged as an open limitation.** Fine-grained free real-time population APIs don't really exist for India at H3-cell granularity. The honest options are: (a) download India Census 2011 ward-level shapefiles and spatially join to H3 cells — accurate but a real chunk of GIS work, or (b) use OSM building density as a population proxy, tagged `source: real-proxy` rather than `source: real`. Not yet implemented — call this out explicitly if asked, rather than quietly treating a proxy as ground truth.

## 6.5 Qualitative Validation Suite

Location: `src/validation/qualitative_checks.py`. Run after generating any dataset, before training on it — a failing check means fix the simulator, not the model.

| Check | Result (last run) |
|---|---|
| Rainy evenings near hospitals → food-category demand up | ✅ +24.9% |
| Lunch hours near office clusters → food-category demand up | ✅ +44.8% |
| Holidays/festivals → overall demand up | ✅ +20.8% |
| Weekends → salon-category demand up | ✅ +20.3% |
| More same-category competitors in a cell → lower per-vendor demand | ✅ -18.3% |

**Worth being honest about in an interview**: the first check initially *failed* (-9.5%) because the simulator applied a flat rain penalty to all categories regardless of item mix — footfall loss wasn't being offset by hot-item demand at the category-total level, even though the product-mix weighting was already directionally correct. Fixed by making the weather effect category- and hour-aware (food vendors get a net-positive rain effect during 17–21h, reflecting tea/hot-snack demand). This is a real example of qualitative validation catching a bug quantitative metrics alone would have missed — MAE/RMSE would have looked fine either way, since both versions are internally consistent with themselves.

---

## 19. Week-by-Week Roadmap

| Week | Focus |
|---|---|
| 1 | Data: real OSM/weather/holiday clients, H3 grid, simulation engine, data validation |
| 2 | Feature store: static/dynamic joins, as-of timestamping, EDA + heatmaps |
| 3 | Modeling: baseline → LightGBM/XGBoost/CatBoost, time-based CV, Optuna tuning, quantile regression, SHAP, MLflow tracking + registry |
| 4 | FastAPI serving: predict/recommend/scenario endpoints, auth, caching, unit tests, Docker |
| 5 | Optimization + decision layer: ranking engine, scenario simulator, Streamlit dashboard |
| 6 | CI/CD + AWS deployment: GitHub Actions, ECS/Fargate, logging, config management |
| 7 | Monitoring + retraining: Evidently drift detection, retrain trigger, polish, full documentation set |

---

## 20. Resume-Ready Description (draft)

> **GeoDemand AI** — Designed and built a production-shaped geospatial decision-intelligence system that predicts hourly customer demand for mobile vendors across H3-indexed city grids and converts predictions into ranked, explainable location recommendations. Built a custom market simulation engine with hidden-variable and interaction-term design to avoid target leakage/circularity in synthetic-data training. Implemented quantile-regression uncertainty estimation, SHAP-driven natural-language explanations, MLflow experiment tracking, FastAPI serving, and time-based validation to prevent temporal leakage. Stack: Python, H3, LightGBM/XGBoost, SHAP, MLflow, FastAPI, Docker, AWS, Streamlit.

---

## 21. Anticipated Interview Questions & How to Answer Them

**Q: Isn't your data fake? How is this valid?**
A: The *contextual* features (weather, POIs, holidays) are real or structurally realistic; the *target* (customer count) is simulated because no public dataset for informal mobile-vendor sales exists. I explicitly designed the simulator to avoid circularity — hidden variables the model never sees, interaction terms, nonlinear saturation, and heteroscedastic noise — so the model has to learn generative structure from noisy partial observation, not invert a known formula. I document this limitation directly in the README rather than hiding it.

**Q: Why predict customer count instead of revenue directly?**
A: Separation of concerns — the ML model should predict a stable, well-defined physical quantity; business metrics (which change with pricing, margins, promotions) are derived downstream without retraining the model.

**Q: How do you know your "confidence" score means anything?**
A: It's derived from quantile regression prediction intervals (P10/P90 width), not an arbitrary heuristic.

**Q: Why not build the full route optimizer?**
A: Scoped deliberately — v1 ranks candidate locations (a legitimate, shippable ML+business-logic system); true multi-stop routing under constraints is a distinct OR problem, planned as v2 with OR-Tools, not conflated with v1's ranking logic.

**Q: How did you validate the model without real-world ground truth?**
A: Time-based cross-validation against the simulated dataset, explicitly framed as validating pipeline correctness (can the model recover known generative structure under noise) rather than claiming real-world predictive accuracy.

---

*End of master document — this reflects the frozen v1.1 architecture as of the current build state (Section 18).*
