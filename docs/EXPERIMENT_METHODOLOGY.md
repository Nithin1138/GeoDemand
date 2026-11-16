# GeoDemand AI — Experimentation, Training & Evaluation Methodology

> **Document Status:** Authoritative Experimentation Methodology Specification (Phase 20.2.2)  
> **Target Version:** GeoDemand AI v2.1.0-prod  
> **Primary Owner:** Member 2 (Machine Learning Engineering + MLOps)  
> **Tracking Hub:** MLflow Local Experiment Registry (`mlruns/`)  

---

## 1. Experimental Philosophy: Zero Future-Data Leakage

In spatial-temporal demand forecasting for mobile micro-retailers, standard cross-validation strategies (such as random K-Fold or stratified splitting) are catastrophically flawed. Randomly partitioning time-series observations leaks future temporal signals, weather patterns, and local event shocks into past training partitions, yielding deceptively optimistic validation scores that fail in live production.

GeoDemand AI enforces the **Temporal Integrity Axiom**:
$$\mathcal{D}_{\text{train}} = \{ (x_t, y_{t+H}) \mid t \le T_{\text{train\_cutoff}} \}$$
$$\mathcal{D}_{\text{val}} = \{ (x_t, y_{t+H}) \mid T_{\text{train\_cutoff}} < t \le T_{\text{val\_cutoff}} \}$$
$$\mathcal{D}_{\text{test}} = \{ (x_t, y_{t+H}) \mid t > T_{\text{val\_cutoff}} \}$$

```
Chronological Timeline
├──────────────────────────────────────┬──────────────────┬──────────────────┤
│             TRAIN (70%)              │  VALIDATION (15%)│    TEST (15%)    │
│    Earliest Observed Historical      │ Hyperparameter   │ Out-of-Sample    │
│    Context & Encoders Fit Here       │ Tuning & Pruning │ Final Evaluation │
└──────────────────────────────────────┴──────────────────┴──────────────────┘
0                                  T_train            T_val              T_end
```

### Strict Anti-Leakage Contracts:
1. **Encoders and Scalers**: Categorical label encoders (`vendor_category`, `weekday`, `season`) and any feature normalization parameters are fit exclusively on $\mathcal{D}_{\text{train}}$. Unseen categories in validation or test sets are handled via explicit fallback handlers (`"unknown"` / `-1`), never by fitting on combined sets.
2. **Feature Observation Timestamp**: Every feature in vector $x_t$ must be strictly observable at or before timestamp $t$. Features that aggregate future intervals (e.g. daily total rainfall, next-hour actual events) are strictly forbidden.
3. **Target Alignment**: The prediction target $y_{t+H}$ is the real physical customer count observed in the forward interval $[t+H-1, t+H]$.

---

## 2. Feature Engineering & Preprocessing Pipeline

The tabular feature pipeline transforms heterogeneous spatial, meteorological, and calendar signals into 45 canonical features:

### 2.1 Cyclical Temporal Encodings
Standard integer representations of hours ($0 - 23$), days ($0 - 364$), and months ($1 - 12$) exhibit artificial discontinuities between the maximum value and zero (e.g. hour 23 to hour 0). We transform these into continuous 2D trigonometric harmonic pairs:
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\text{day\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_year}}{365.25}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_year}}{365.25}\right)$$
$$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

This preserves the Euclidean proximity between 23:59 and 00:01 in tree-based and distance-based architectures.

### 2.2 Spatial & POI Density Features
Using Uber's H3 spatial index at Resolution 8 ($\approx 0.737\,\text{km}^2$ cell area, $\approx 461\,\text{m}$ edge length):
- OpenStreetMap POI counts within each cell are aggregated into distinct functional categories: `office_count`, `bus_stop_count`, `metro_station_count`, `railway_station_count`, `college_count`, `school_count`, `hospital_count`, `mall_count`, `park_count`, `parking_count`, and `restaurant_count`.
- Land-use zoning ratios are normalized to $[0.0, 1.0]$: `commercial_ratio`, `residential_ratio`, and `industrial_ratio`.
- Infrastructure metrics: `building_density` and `road_density` (total linear meters of road network within cell).

### 2.3 Meteorological Features
Hourly weather signals ingested from Open-Meteo:
- Continuous physical signals: `temperature` (°C), `humidity` (%), `pressure` (hPa), `wind_speed` (km/h), `cloud_cover` (%), `visibility` (m), `uv_index`, and `rainfall` (mm/h).
- Non-linear effects: High temperature ($> 38^\circ\text{C}$) and precipitation ($> 5\text{ mm/h}$) suppress open-air cart demand while driving indoor mall and delivery footfall.

---

## 3. Bayesian Hyperparameter Optimization (Optuna)

Model tuning was performed using **Optuna** with the Tree-structured Parzen Estimator (TPE) algorithm.

### 3.1 Objective Function
The optimization objective directly minimizes validation Mean Absolute Error (MAE):
$$\min_{\theta \in \Theta} \frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{i \in \mathcal{D}_{\text{val}}} \left| y_i - f(x_i; \theta) \right|$$

MAE is selected over Mean Squared Error (MSE) because:
1. Mobile street vendor earnings scale linearly with customer counts, not quadratically.
2. MAE provides robustness against rare footfall spikes (e.g. unexpected flash mob or parade) that would otherwise distort model weights under quadratic penalties.

### 3.2 Search Space & Optimal Parameters

| Hyperparameter | Distribution | Search Bounds | Champion Value (`v2.1.0`) | Architectural Role |
| :--- | :--- | :---: | :---: | :--- |
| `n_estimators` | Integer | $[100, 800]$ | **450** | Number of boosting rounds |
| `learning_rate` | Log-uniform | $[0.01, 0.20]$ | **0.05** | Shrinkage step size |
| `num_leaves` | Integer | $[31, 150]$ | **95** | Maximum tree complexity |
| `min_child_samples`| Integer | $[10, 100]$ | **25** | Minimum observations per leaf |
| `subsample` | Uniform | $[0.60, 1.00]$ | **0.80** | Row subsampling ratio (bagging) |
| `colsample_bytree` | Uniform | $[0.60, 1.00]$ | **0.80** | Column subsampling ratio |
| `reg_alpha` ($L_1$) | Log-uniform | $[10^{-3}, 10.0]$ | **0.10** | Feature sparsity regularization |
| `reg_lambda` ($L_2$)| Log-uniform | $[10^{-3}, 10.0]$ | **0.50** | Weight shrinkage regularization |

### 3.3 Pruning and Early Stopping
- **Trial Budget**: 100 trials with `MedianPruner(n_startup_trials=10, n_warmup_steps=20)`.
- **Early Stopping**: Early stopping patience set to 30 boosting rounds based on validation MAE.
- All trials were tracked in MLflow under experiment `geodemand_optuna_tuning`.

---

## 4. Multi-Horizon Sequence Modeling Formulation

For Phase 13.2, the modeling paradigm was extended from single-step ($H=1$) to multi-horizon forecasting ($H \in \{1, 2, 4, 8\}$):

### 4.1 Sliding-Window Preparation
Given chronological sequence $(x_1, y_1), (x_2, y_2), \dots, (x_T, y_T)$:
- **Input Window (Lookback $L$)**: Context history $[t - L + 1, t]$ with $L = 24$ hours.
- **Forecast Window (Horizon $H$)**: Targets $[y_{t+1}, y_{t+2}, \dots, y_{t+H}]$.
- **Feature Alignment**: Static features (vendor capacity, POI counts) are broadcast across sequence steps; dynamic features (hourly weather, cyclic harmonics) vary per step.

### 4.2 Training Protocols by Architecture
1. **Direct Horizon Boosting (LightGBM Champion)**:
   Independent or multi-target boosting models trained for specific horizon targets $H \in \{1, 2, 4, 8\}$. Avoids error feedback loops.
2. **Recursive Autoregressive (LightGBM AR)**:
   A single model $f(x_t, y_{t-1})$ trained on $H=1$, recursively predicting:
   $$\hat{y}_{t+h} = f(\tilde{x}_{t+h}, \hat{y}_{t+h-1})$$
   Evaluated to test recursive stability against error compounding.
3. **Neural Deep Sequences (TFT, N-HiTS, PatchTST)**:
   Trained via AdamW optimizer with cosine annealing learning rate schedules, batch size 64, and dropout $0.10$. Multi-quantile output heads simultaneously optimize Pinball loss across $\{0.10, 0.50, 0.90\}$.

---

## 5. Experiment Tracking & Artifact Management

All model runs, metrics, parameters, and serialized bundles are logged through **MLflow**:
- **Artifacts Saved**:
  - `model.pkl`: Model regressor bundle.
  - `feature_schema.json`: Strict input feature names and column order.
  - `label_encoders.pkl`: Fitted scikit-learn LabelEncoders.
  - `metrics.json`: Validation and test MAE, RMSE, $R^2$, Quantile Coverage, and Latency.
- **Reproducibility Guarantee**:
  - Global random seeds pinned to `42` across Python `random`, NumPy, and LightGBM.
  - Every released model artifact is verified via SHA-256 cryptographic digest before promotion.
