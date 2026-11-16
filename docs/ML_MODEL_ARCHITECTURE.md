# GeoDemand AI — Machine Learning Model Architecture Specification

> **Document Status:** Authoritative Architectural Specification (Phase 20.2.1)  
> **Target Version:** GeoDemand AI v2.1.0-prod  
> **Primary Owner:** Member 2 (Machine Learning Engineering + MLOps)  
> **Production Model Artifact:** `models/demand_model.pkl` (SHA-256: `c75ac3b4...`)  
> **Approved Baseline Artifact:** `models/champion_backup/demand_model.pkl`  

---

## 1. Architectural Philosophy & Non-Negotiable Contracts

GeoDemand AI adopts a decoupled, production-grade intelligence architecture built upon five governing principles:

1. **One Predictive Target Only**: The ML engine predicts **physical customer count in the next forward prediction window** ($t \to t+H$, where $H \in \{1, 2, 4, 8\}$ hours, with $H=1$ as primary operational target). It **never** directly predicts revenue, profit, demand scores, or relocation decisions. Business economics and multi-objective ranking are derived downstream in the business layer (`src/api/recommender.py`).
2. **Zero Future-Data Leakage (Anti-Circularity)**: The temporal contract strictly requires that features observed at decision timestamp $t$ forecast demand at $t+H$. All preprocessing, categorical encodings, scaling, and validation splits are strictly fit on chronological training masks ($t < T_{\text{cutoff}}$). Random shuffling and standard k-fold cross-validation are strictly prohibited.
3. **Decoupled Point & Quantile Estimation**: Predictions output the median forecast alongside calibrated lower ($P_{10}$) and upper ($P_{90}$) prediction intervals to quantify spatial-temporal risk without assuming Gaussian normality.
4. **Sub-3ms Latency Budget**: Inference must execute within $< 3\,\text{ms}$ per candidate location to allow the live recommender API to evaluate dozens of neighboring H3 cells in $< 50\,\text{ms}$.
5. **Native TreeSHAP Explainability**: Local attributions must compute within the request lifecycle ($< 1.5\,\text{ms}$) to output human-interpretable demand drivers to mobile vendors.

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
│   │ Native Fast TreeSHAP Engine (< 0.5ms per candidate vector)     │   │
│   │ • Positive Driver Attribution   • Negative Barrier Attribution │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
         Expected Customer Count + Quantile Interval + Driver Attributions
```

---

## 2. Input Feature Schema (45 Canonical Features)

The canonical feature store (`data/processed/feature_store.parquet`) supplies 45 strictly ordered, typed, and bounded features to the model, frozen in `models/frozen_feature_schema.json`:

| Feature Name | Type | Physical Bounds | Domain / Interpretation |
| :--- | :--- | :---: | :--- |
| `average_order_value` | Float | $[10.0, 500.0]$ | Vendor average order value in INR |
| `building_density` | Float | $[0.0, \infty)$ | Number of building footprints per H3 cell |
| `bus_stop_count` | Float | $[0.0, \infty)$ | Count of public transit bus stops |
| `cloud_cover` | Float | $[0.0, 100.0]$ | Cloud cover percentage |
| `college_count` | Float | $[0.0, \infty)$ | Universities and colleges in proximity |
| `commercial_ratio` | Float | $[0.0, 1.0]$ | Ratio of commercial zoning in H3 cell |
| `competition_score` | Float | $[0.0, 100.0]$ | Local density of competing vendors in category |
| `day_cos`, `day_sin` | Float | $[-1.0, 1.0]$ | Cyclical harmonics for day-of-year |
| `event_importance` | Float | $[0.0, 1.0]$ | Nearby active event magnitude score |
| `hospital_count` | Float | $[0.0, \infty)$ | Hospitals and medical centers in cell |
| `hour` | Integer | $[0, 23]$ | Hour of day |
| `hour_cos`, `hour_sin` | Float | $[-1.0, 1.0]$ | Daily cyclical harmonics ($\cos(2\pi h/24), \sin(2\pi h/24)$) |
| `humidity` | Float | $[0.0, 100.0]$ | Relative humidity percentage |
| `industrial_ratio` | Float | $[0.0, 1.0]$ | Industrial land-use fraction |
| `inventory_capacity` | Float | $[1.0, 1000.0]$ | Vendor daily max serving capacity |
| `is_holiday` | Float/Int | $\{0, 1\}$ | Regional Andhra Pradesh / National holiday indicator |
| `is_weekend` | Float/Int | $\{0, 1\}$ | Saturday / Sunday indicator |
| `mall_count` | Float | $[0.0, \infty)$ | Shopping malls and commercial retail centers |
| `metro_station_count` | Float | $[0.0, \infty)$ | Metro / rapid transit hubs (0 in Tier-2 cities) |
| `month`, `month_cos`, `month_sin` | Float | Various | Seasonal month indicators and harmonics |
| `office_count` | Float | $[0.0, \infty)$ | Commercial office buildings (IT / corporate) |
| `park_count` | Float | $[0.0, \infty)$ | Public parks and recreation zones |
| `parking_count` | Float | $[0.0, \infty)$ | Formal vehicle parking structures |
| `population_density` | Float | $[0.0, \infty)$ | Resident and footfall population per $\text{km}^2$ |
| `preparation_time` | Float | $[1.0, 60.0]$ | Vendor order turnaround time (minutes) |
| `pressure` | Float | $[800.0, 1100.0]$| Atmospheric pressure in hPa |
| `quarter` | Integer | $[1, 4]$ | Business quarter |
| `railway_station_count` | Float | $[0.0, \infty)$ | Intercity railway terminals |
| `rainfall` | Float | $[0.0, 200.0]$ | Precipitation in mm/hr |
| `residential_ratio` | Float | $[0.0, 1.0]$ | Residential zoning fraction |
| `restaurant_count` | Float | $[0.0, \infty)$ | Dining and competitor food establishments |
| `road_density` | Float | $[0.0, \infty)$ | Total road segment length within H3 cell |
| `school_count` | Float | $[0.0, \infty)$ | Primary and secondary educational institutions |
| `school_vacation` | Float/Int | $\{0, 1\}$ | School break indicator |
| `season` | Categorical | Strings | Seasonal indicator (`monsoon`, `summer`, `winter`) |
| `temperature` | Float | $[-10.0, 55.0]$ | Ambient temperature in degrees Celsius |
| `uv_index` | Float | $[0.0, 15.0]$ | Solar UV index |
| `vendor_category` | Categorical | Strings | Business type (`food`, `chai`, `snacks`, `fruits`) |
| `visibility` | Float | $[0.0, 50000.0]$| Atmospheric visibility in meters |
| `weekday` | Categorical | Strings | Day of week (`Monday` to `Sunday`) |
| `wind_speed` | Float | $[0.0, 150.0]$ | Wind velocity in km/h |

---

## 3. Comprehensive Model Architectures Evaluated

During Phases 2.2 and 13.2, eight distinct model architectures spanning classical tabular baselines, gradient boosting machines, autoregressive setups, and deep neural time-series transformers were implemented, calibrated, and evaluated.

### 3.1 Tabular Baseline Regressors

1. **Ordinary Least Squares (Linear Regression)**:
   - Form: $\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j X_j$.
   - Strengths: Fast ($< 0.01\text{ ms}$), interpretable coefficients.
   - Weaknesses: Unable to capture complex spatial interactions, nonlinear weather-demand thresholds, or asymmetric quantile risks. Test MAE: $1.471$.
2. **Random Forest Regressor**:
   - Form: Ensemble of 100 bagged de-correlated decision trees with max depth 12.
   - Strengths: Non-parametric, robust to outliers.
   - Weaknesses: High RAM footprint ($48.2\text{ MB}$), sluggish inference ($8.40\text{ ms}$ p95), unable to natively compute fast TreeSHAP during live scoring. Test MAE: $0.905$.
3. **CatBoost Regressor**:
   - Form: Oblivious decision trees with ordered target statistics for categorical splits.
   - Strengths: Native categorical handling, low overfitting risk.
   - Weaknesses: Latency ($3.10\text{ ms}$ p95) marginally exceeds the sub-3ms SLA when combined with explainability. Test MAE: $0.772$.
4. **XGBoost Regressor**:
   - Form: Depth-wise gradient boosted trees with histogram-based binning (`hist`).
   - Strengths: High predictive accuracy (Test MAE $0.763$).
   - Weaknesses: Memory consumption ($10.2\text{ MB}$) and inference latency ($2.84\text{ ms}$) higher than LightGBM.

### 3.2 Production Champion: LightGBM Regressor (`regression_l1`, v2.1.0)

- **Architecture**: Leaf-wise tree growth with Maximum Depth constraint, Histogram-based feature binning, Gradient-based One-Side Sampling (GOSS), and Exclusive Feature Bundling (EFB).
- **Objective**: Mean Absolute Error (MAE / $L_1$ loss), aligned directly with the operational cost function of mobile vendors.
- **Quantile Extensions**: Three synchronized models: Point/Median ($\alpha=0.50$), Lower Bound ($\alpha=0.10$), and Upper Bound ($\alpha=0.90$).
- **Explainability**: Compiled C++ TreeSHAP hook (`predict(pred_contrib=True)`).
- **Production Performance**:
  - Test MAE ($H=1$): **$0.7565$** (Vijayawada test split) / **$3.6672$** (canonical multi-horizon benchmark).
  - Single-sample inference latency: **$0.008\text{ ms}$** (vectorized) / **$1.75\text{ ms}$** (full API path).
  - Artifact size: **$3.8\text{ MB}$**.
  - Quantile coverage: **$82.4\%$** (within $[70\%, 95\%]$ target range).

```python
# Champion Hyperparameters
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

### 3.3 Advanced Multi-Horizon Architectures (Phase 13.2)

1. **Recursive Autoregressive LightGBM**:
   - **Formulation**: Iterative one-step-ahead forecasting where $\hat{y}_{t+h}$ is fed back as an autoregressive lag feature $y_{t+h-1}$ for step $h+1$.
   - **Strengths**: Shared model weights across all horizons; flexible sequence generation.
   - **Weaknesses**: Compounding prediction error across long horizons. For $H=1$, MAE matches Champion ($3.6672$), but degrades sharply to $5.8846$ at $H=2$, $8.6567$ at $H=4$, and $10.1549$ at $H=8$.
   - **Latency**: $0.16\text{ ms}$ per sample (recursive loop overhead).

2. **Temporal Fusion Transformer (TFT)**:
   - **Formulation**: Attention-based deep architecture designed specifically for multi-horizon time-series forecasting. Incorporates Variable Selection Networks (VSN) for feature importance, Gated Residual Networks (GRN) for nonlinear processing, and Multi-Head Self-Attention for long-range temporal dependencies.
   - **Quantiles**: Directly optimizes multi-quantile pinball loss ($0.10, 0.50, 0.90$) in a single forward pass.
   - **Results**:
     - $H=1$ MAE: $4.9479$
     - $H=2$ MAE: $5.1497$
     - $H=4$ MAE: $5.5559$
     - $H=8$ MAE: $6.3724$
     - Coverage: $25.0\%$
     - Mean interval width: $4.91$
     - Latency: $0.007\text{ ms}$ per sample; Peak RAM: $0.66\text{ MB}$.
   - **Trade-off**: Requires extensive sequence histories (sliding windows $\ge 24\text{h}$) and shows lower point accuracy on static spatial POI features compared to gradient boosted trees.

3. **N-HiTS (Neural Hierarchical Interpolation for Time Series)**:
   - **Formulation**: Multi-rate hierarchical decomposition architecture. Operates through backward and forward residual blocks with multi-rate sampling of inputs and multi-frequency synthesis of predictions.
   - **Results**:
     - $H=1$ MAE: $4.8359$
     - $H=2$ MAE: $5.0293$
     - $H=4$ MAE: $5.4162$
     - $H=8$ MAE: $6.1900$
     - Coverage: $40.0\%$
     - Mean interval width: $9.43$
     - Latency: $0.006\text{ ms}$ per sample; Peak RAM: $0.52\text{ MB}$.
   - **Trade-off**: Excellent long-horizon stability, but underperforms LightGBM on immediate $H=1$ and $H=2$ operational horizons.

4. **PatchTST (Patch Time Series Transformer)**:
   - **Formulation**: Channel-independent patch-based Transformer. Segments continuous 1D time-series into sub-series patches that serve as input tokens to a Transformer encoder. Preserves local semantic information and reduces attention complexity from $O(L^2)$ to $O((L/P)^2)$.
   - **Results**:
     - $H=1$ MAE: $4.6188$
     - $H=2$ MAE: $4.7908$
     - $H=4$ MAE: $5.1512$
     - $H=8$ MAE: $5.8781$
     - Coverage: $60.0\%$
     - Mean interval width: $10.67$
     - Latency: $0.006\text{ ms}$ per sample; Peak RAM: $0.27\text{ MB}$.
   - **Trade-off**: Superior quantile coverage among deep architectures, but larger interval widths and higher operational pipeline complexity compared to LightGBM.

---

## 4. Multi-Horizon Architecture Benchmark Summary

The empirical benchmark executed across all 5 advanced models under identical chronological splits confirms the production champion selection:

| Model Architecture | H=1 MAE | H=2 MAE | H=4 MAE | H=8 MAE | 80% Coverage | Interval Width | Latency (ms) | Peak RAM | Decision Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Approved LightGBM (v2.1.0)** | **3.6672** | **3.8789** | **4.3250** | **5.2931** | **82.4%** | 3.60 | **0.02 ms** | 1.47 MB | **PRODUCTION CHAMPION** |
| Recursive Autoregressive LightGBM | **3.6672** | 5.8846 | 8.6567 | 10.1549 | 21.0% | 3.60 | 0.16 ms | 1.47 MB | Rejected (Error accumulation) |
| Temporal Fusion Transformer (TFT) | 4.9479 | 5.1497 | 5.5559 | 6.3724 | 25.0% | 4.91 | 0.01 ms | 0.66 MB | Research / Long-horizon |
| N-HiTS Hierarchical Synthesis | 4.8359 | 5.0293 | 5.4162 | 6.1900 | 40.0% | 9.43 | 0.01 ms | 0.52 MB | Research / Macro-planning |
| PatchTST Patch Transformer | 4.6188 | 4.7908 | 5.1512 | 5.8781 | 60.0% | 10.67 | 0.01 ms | 0.27 MB | Research / High-volatility |

### Champion Selection Criteria:
1. **Primary Operational Target ($H=1$)**: LightGBM achieves $3.6672$ MAE, representing a **$20.6\%$ lower error** than PatchTST ($4.6188$) and **$25.8\%$ lower error** than TFT ($4.9479$).
2. **Serving SLA Compliance**: Raw scoring is completed in $0.008 - 0.02\text{ ms}$, well beneath the $3.0\text{ ms}$ budget.
3. **Explainability**: Only LightGBM provides native sub-millisecond TreeSHAP driver attributions without sampling approximations.
4. **Operational Simplicity**: No GPU hardware required; negligible cold-start latency ($< 15\text{ ms}$).

---

## 5. Serving Pipeline & Non-Crossing Monotonicity

Serving is encapsulated through `DemandModelPredictor` in [`src/geodemand/models/inference.py`](file:///c:/Users/Lenovo/projects/GeoDemand/src/geodemand/models/inference.py) and called by `DemandModel` in [`src/api/recommender.py`](file:///c:/Users/Lenovo/projects/GeoDemand/src/api/recommender.py):

* **Validation & Encoding**: Input dictionaries are validated against `models/frozen_feature_schema.json`. Categoricals are transformed using deterministic label encoders fit strictly on historical training sets.
* **Non-Crossing Monotonicity Enforcement**:
  Because quantile heads are trained independently, quantile crossing ($P_{10} > P_{50}$ or $P_{50} > P_{90}$) can mathematically occur in under-sampled regions of feature space. The predictor strictly enforces non-crossing monotonicity via:
  $$P_{10} = \max\left(0, \text{round}(\hat{q}_{10})\right)$$
  $$P_{50} = \max\left(P_{10}, \text{round}(\hat{q}_{50})\right)$$
  $$P_{90} = \max\left(P_{50}, \text{round}(\hat{q}_{90})\right)$$
* **Guaranteed Properties**:
  1. Non-negativity: $\forall q, P_q \ge 0$.
  2. Ordering: $0 \le P_{10} \le P_{50} \le P_{90}$ with strictly zero violations.
  3. Discrete Customer Unit: Outputs are rounded to integer customer counts.
