# GeoDemand AI — Machine Learning Model Architecture Specification

> **Document Status:** Authoritative Architectural Specification  
> **Target Version:** GeoDemand AI v2.0  
> **Primary Owner:** Member 2 (Machine Learning Engineering + MLOps)  
> **Production Model Artifact:** `models/demand_model.pkl` (`v2.1.0`)

---

## 1. Architectural Philosophy & Non-Negotiable Contracts

GeoDemand AI adopts a decoupled, production-grade intelligence architecture built upon five governing principles:

1. **One Predictive Target Only**: The ML engine predicts **physical customer count in the next 1-hour window** ($t \to t+1$). It **never** directly predicts revenue, profit, demand scores, or relocation decisions. Business economics and multi-objective ranking are derived downstream in the business layer (`src/api/recommender.py`).
2. **Zero Future-Data Leakage (Anti-Circularity)**: The temporal contract strictly requires that features observed at decision timestamp $t$ forecast demand at $t+1\text{h}$. All preprocessing, categorical encodings, scaling, and validation splits are strictly fit on chronological training masks ($t < T_{\text{cutoff}}$). Random shuffling is prohibited.
3. **Decoupled Point & Quantile Estimation**: Predictions output the median forecast alongside calibrated lower ($P_{10}$) and upper ($P_{90}$) prediction intervals to quantify spatial-temporal risk.
4. **Sub-3ms Latency Budget**: Inference must execute within $< 3\,\text{ms}$ per candidate location to allow the live recommender API to evaluate dozens of neighboring H3 cells in $< 50\,\text{ms}$.
5. **Native TreeSHAP Explainability**: Local attributions must compute within the request lifecycle to output human-interpretable demand drivers to mobile vendors.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CONTEXTUAL SIGNALS AT TIMESTAMP t                    │
│   • Uber H3 Spatial Grid (Res 8)    • Open-Meteo Hourly Forecast       │
│   • OSM POI Footprint Densities     • Calendar & Regional Holidays     │
│   • Dynamic Vendor Characteristics  • Cyclical Temporal Harmonics     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Strict Schema: 45 Features)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              PRODUCTION CHAMPION ENGINE (LightGBM v2.1.0)              │
│                                                                        │
│   ┌────────────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│   │ P10 Lower Bound Model  │  │ Point / Median │  │ P90 Upper Bound│   │
│   │ (Quantile alpha=0.10)  │  │ (L1 Objective) │  │ (Quantile 0.9) │   │
│   └───────────┬────────────┘  └───────┬────────┘  └────────┬───────┘   │
│               │                       │                    │           │
│               ▼                       ▼                    ▼           │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Monotonicity Enforcement: 0 <= P10 <= P50 (Point) <= P90       │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │                                    │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Native Fast TreeSHAP Engine (< 1.5ms per candidate vector)     │   │
│   │ • Positive Driver Attribution   • Negative Barrier Attribution │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
         Expected Customer Count + Quantile Interval + Driver Attributions
```

---

## 2. Input Feature Schema (45 Canonical Features)

The canonical feature store (`data/processed/feature_store.parquet`) supplies 45 strictly ordered features to the model:

| Feature Name | Type | Domain / Interpretation |
| :--- | :--- | :--- |
| `average_order_value` | Float | Vendor average order value in INR |
| `building_density` | Float | Number of building footprints per H3 cell |
| `bus_stop_count` | Float | Count of public transit bus stops |
| `cloud_cover` | Float | Cloud cover percentage ($0 - 100$) |
| `college_count` | Float | Universities and colleges in proximity |
| `commercial_ratio` | Float | Ratio of commercial zoning in H3 cell |
| `competition_score` | Float | Local density of competing vendors in category |
| `day_cos`, `day_sin` | Float | Cyclical harmonics for day-of-year |
| `event_importance` | Float | Nearby active event magnitude score ($0.0 - 1.0$) |
| `hospital_count` | Float | Hospitals and medical centers in cell |
| `hour` | Integer | Hour of day ($0 - 23$) |
| `hour_cos`, `hour_sin` | Float | Daily cyclical harmonics ($\cos(2\pi h/24), \sin(2\pi h/24)$) |
| `humidity` | Float | Relative humidity ($0 - 100\%$) |
| `industrial_ratio` | Float | Industrial land-use fraction |
| `inventory_capacity` | Float | Vendor daily max serving capacity |
| `is_holiday` | Float/Int | Regional Andhra Pradesh / National holiday indicator |
| `is_weekend` | Float/Int | Saturday / Sunday indicator |
| `mall_count` | Float | Shopping malls and commercial retail centers |
| `metro_station_count` | Float | Metro / rapid transit hubs |
| `month`, `month_cos`, `month_sin` | Float | Seasonal month indicators and harmonics |
| `office_count` | Float | Commercial office buildings (IT / government) |
| `park_count` | Float | Public parks and recreation zones |
| `parking_count` | Float | Formal vehicle parking structures |
| `population_density` | Float | Resident and footfall population per $\text{km}^2$ |
| `preparation_time` | Float | Vendor order turnaround time (minutes) |
| `pressure` | Float | Atmospheric pressure in hPa |
| `quarter` | Integer | Business quarter ($1 - 4$) |
| `railway_station_count` | Float | Intercity railway terminals |
| `rainfall` | Float | Precipitation in mm/hr |
| `residential_ratio` | Float | Residential zoning fraction |
| `restaurant_count` | Float | Dining and competitor food establishments |
| `road_density` | Float | Total road segment length within H3 cell |
| `school_count` | Float | Primary and secondary educational institutions |
| `school_vacation` | Float/Int | School break indicator |
| `season` | Categorical | Seasonal indicator (monsoon, summer, winter) |
| `temperature` | Float | Ambient temperature in degrees Celsius |
| `uv_index` | Float | Solar UV index |
| `vendor_category` | Categorical | Business type (`food`, `chai`, `snacks`, `fruits`) |
| `visibility` | Float | Atmospheric visibility in meters |
| `weekday` | Categorical | Day of week (`Monday` to `Sunday`) |
| `wind_speed` | Float | Wind velocity in km/h |

---

## 3. Production Champion Specification

The production model bundle is serialized as `models/demand_model.pkl` and contains:
1. **Primary Regressor (`model`)**: LightGBM `regression_l1` (MAE-optimizing objective).
2. **Quantile Regressor Heads**: Dedicated LightGBM regressors with `objective="quantile"` for $\alpha=0.10$ (`p10_model`) and $\alpha=0.90$ (`p90_model`).
3. **Encoders**: Deterministic `LabelEncoder` objects fit strictly on training mask for `vendor_category`, `weekday`, and `season`.
4. **Metadata Sidecar**: Strict schema, hash, metrics, and training timestamps (`models/demand_model_meta.json`).

### Hyperparameters (Champion `v2.1.0`)
```python
CHAMPION_HYPERPARAMETERS = {
    "objective": "regression_l1",
    "n_estimators": 450,
    "learning_rate": 0.05,
    "num_leaves": 95,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "random_state": 42,
    "n_jobs": -1,
}
```

---

## 4. Serving & Inference Pipeline

Serving is abstracted through `DemandModelPredictor` in [`src/geodemand/models/inference.py`](file:///c:/Users/Lenovo/projects/GeoDemand/src/geodemand/models/inference.py) and wrapped by `DemandModel` in [`src/api/recommender.py`](file:///c:/Users/Lenovo/projects/GeoDemand/src/api/recommender.py):
* **Batch Scoring**: Validates schema, aligns missing columns to default $0.0$, encodes categoricals with unseen-class fallback, predicts vector, and clips outputs to non-negative bounds: $\hat{y} = \max(0, \text{round}(y))$.
* **Uncertainty Quantification**: Extracts $P_{10}$, $P_{50}$, and $P_{90}$, and strictly enforces non-crossing:
  $$P_{10} = \max(0, \text{round}(\hat{q}_{10}))$$
  $$P_{50} = \max(P_{10}, \text{round}(\hat{q}_{50}))$$
  $$P_{90} = \max(P_{50}, \text{round}(\hat{q}_{90}))$$
* **Latency Profile**:
  - Raw single-sample latency: **1.65 ms** (p50), **1.75 ms** (p95).
  - End-to-end latency with TreeSHAP: **2.08 ms** (p50), **2.18 ms** (p95).
