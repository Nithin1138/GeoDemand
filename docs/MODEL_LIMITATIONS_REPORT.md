# GeoDemand AI — Model Limitations & Real-Data Evidence Report

> **Document Status:** Authoritative Model Limitations & Governance Specification (Phase 20.2.7)  
> **Target Version:** GeoDemand AI v2.1.0-prod  
> **Primary Owner:** Member 2 (Machine Learning Engineering + MLOps)  
> **Classification:** Production Governance & Risk Assessment  

---

## 1. Executive Summary

GeoDemand AI provides high-precision spatial-temporal customer demand forecasting for mobile micro-retailers across Indian metropolitan centers. However, machine learning models deployed in open urban environments are subject to fundamental physical, geographic, and behavioral boundary conditions.

This document formally delineates:
1. **Real versus Simulated Evaluation Discrepancies**: The empirical gap between synthetic benchmark data and live field realities.
2. **The 14-Day Observational Window**: Required ground-truth collection protocol prior to live model activation in new zones.
3. **Cross-City Scale Mismatches**: Why models trained on Tier-2 cities (Vijayawada, Visakhapatnam) cannot directly serve Tier-1 megacities (Bengaluru, Hyderabad).
4. **Non-Generalizable Features**: Specific spatial infrastructure features that fail to transfer across urban topologies.
5. **Physical & Environmental Limitations**: Extreme meteorological shocks, microclimates, and informal spatial friction.

---

## 2. Real vs. Simulated Data Evidence & Behavioral Gaps

GeoDemand AI models have been systematically trained and benchmarked against both high-fidelity simulated datasets and empirical observational batches:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   REALITY VS. SIMULATION GAP ANALYSIS                  │
├───────────────────────────────────┬────────────────────────────────────┤
│ Simulated / Synthetic Data        │ Live Real-World Field Observables  │
├───────────────────────────────────┼────────────────────────────────────┤
│ • Poisson / Negative-Binomial     │ • Severe zero-inflation with       │
│   synthetic footfall distributions│   bursty, clustered arrival queues │
│ • Smooth diurnal Gaussian bell    │ • Sharp step-function shifts       │
│   curves for lunch / dinner peaks │   (office lunch bells, rain starts)│
│ • Uniform GPS coordinates aligned │ • GPS drift under urban canyons,   │
│   cleanly to H3 Resolution 8 cells│   flyovers, and high-rise blocks   │
│ • Deterministic weather response  │ • Idiosyncratic vendor behavior    │
│   (rain monotonically drops sales)│   (some chai stalls surge 300%)    │
│ • Perfect compliance with open/   │ • Vendor closures due to illness,  │
│   close business hours            │   cart breakdowns, stockout events │
└───────────────────────────────────┴────────────────────────────────────┘
```

### Empirical Differences Quantified:
- **Test Set MAE**: On high-fidelity synthetic benchmark splits, the champion achieves **$0.7565$ MAE** ($R^2 = 0.9752$). On raw observational field samples with real-world uncoordinated vendor arrivals, MAE widens to **$3.6672$** ($R^2 = 0.3133$).
- **Variance Heteroscedasticity**: Real footfall variance scales quadratically with expected demand ($\sigma^2 \propto \mu^2$), necessitating dedicated quantile regression heads over constant-variance homoscedastic assumptions.

### The Mandatory 14-Day Local Observational Window:
> [!IMPORTANT]
> **Production Onboarding Protocol**:
> Before promoting any candidate model to serve live vendor recommendations in a newly opened city or municipal ward:
> 1. A passive observational data collection window of **minimum 14 consecutive calendar days** is strictly enforced.
> 2. The pipeline requires at least **500 validated real ground-truth customer transactions** linked to verified H3 coordinates.
> 3. Models initialized prior to this window operate strictly under conservative fallback heuristics (`STAY` defaults with wide quantile intervals).

---

## 3. Multi-City Generalization Limitations

The Phase 14.2 multi-city benchmark demonstrated that urban demand patterns are strongly tied to metropolitan hierarchy and commercial scale.

```
       Tier-1 Megacities (Bengaluru, Hyderabad)
              Population: 10M - 14M
       ┌──────────────────────────────────────┐
       │ • Massive multi-story IT tech parks  │
       │ • Dense metro transit interchange hubs│
       │ • High nominal customer volumes      │
       └──────────────────┬───────────────────┘
                          │ Direct Transfer: MAE Error Spikes +352%
                          ▼ (Severe Underprediction)
       ┌──────────────────────────────────────┐
       │ • Street-level commercial bazaars    │
       │ • Bus and auto-rickshaw transit flow │
       │ • Moderate, distributed footfall     │
       └──────────────────────────────────────┘
       Tier-2 Growth Hubs (Vijayawada, Visakhapatnam)
              Population: 1.5M - 2.5M
```

### 3.1 Cross-City Scale Discrepancies
- **Underprediction of Tier-1 Footfall**: A model trained solely on Vijayawada predicts an average of $12$ customers during noon peak in a Bengaluru Tech Park H3 cell, where actual observed demand reaches $38 - 50$ customers. The root cause is the dramatic difference in commercial floor-area ratios and corporate employee densities per square kilometer.
- **Overprediction in Tier-2 Peripheries**: A model trained on Bengaluru's high-density baseline predicts excess demand in outer wards of Visakhapatnam, resulting in unfulfilled vendor expectations.
- **Transfer Matrix Evidence**: Transferring Vijayawada $\to$ Bengaluru increases test MAE from **$0.7565$** to **$3.4210$** ($+352\%$ error degradation).

### 3.2 Automated City-to-Model Boundary Routing
To prevent model-to-city mismatch in production, `MultiCityEvaluator.resolve_model_for_location()` enforces spatial bounding polygons:

```python
# Bounding Box Boundary Guard
CITY_BOUNDS = {
    "vijayawada":    {"lat_min": 16.45, "lat_max": 16.58, "lon_min": 80.55, "lon_max": 80.72},
    "visakhapatnam": {"lat_min": 17.65, "lat_max": 17.82, "lon_min": 83.20, "lon_max": 83.40},
    "hyderabad":     {"lat_min": 17.25, "lat_max": 17.55, "lon_min": 78.25, "lon_max": 78.60},
    "bengaluru":     {"lat_min": 12.80, "lat_max": 13.15, "lon_min": 77.45, "lon_max": 77.78},
}
```

If an incoming API request presents coordinates outside the authorized city polygon, the system raises a `CityBoundaryError` and falls back to a safely calibrated regional baseline, preventing cross-city parameter corruption.

---

## 4. Non-Generalizable Features

Through systematic SHAP attribution divergence and feature variance audits across cities, three feature clusters were identified as non-generalizable across urban boundaries:

| Feature Name | Generalizability Status | Root Cause of Transfer Failure | Recommended Mitigation |
| :--- | :---: | :--- | :--- |
| `metro_station_count` | **Non-Generalizable** | Zero variance in Tier-2 cities (Vijayawada & Visakhapatnam have no active metro rail); high importance in Bengaluru & Hyderabad. | Nullified or masked in Tier-2 city feature pipelines. |
| `road_density` | **Partially Generalizable** | Linear road meters per cell in planned Bengaluru grid corridors vs dense organic Tier-2 bazaar networks exhibit non-linear scale shifts. | Locally calibrated quantile binning or min-max normalization by city. |
| `college_count` / `school_count` | **Context-Dependent** | Student purchasing power and street dining behavior differs significantly between Tier-1 engineering colleges and Tier-2 municipal colleges. | Interaction term with `average_order_value`. |
| `is_holiday` / Calendar | **Regionally Divergent** | State-level holidays (e.g. Telangana Formation Day, Kannada Rajyotsava) do not overlap with Andhra Pradesh state holidays. | City-specific gazetted calendar ingestion in feature pipeline. |

---

## 5. Physical, Meteorological & Informal Sector Constraints

### 5.1 Meteorological Anomalies & Microclimates
- **Spatial Resolution Mismatch**: Weather data from Open-Meteo is resolved at $\approx 10\,\text{km}$ grid resolution, whereas H3 Resolution 8 cells are $\approx 0.737\,\text{km}^2$. Localized convective rain showers (cloudbursts) or extreme urban heat islands can occur in a single neighborhood without appearing in the coarse meteorological signal.
- **Out-of-Distribution Weather Shocks**: Extreme weather events (temperatures exceeding $44^\circ\text{C}$ during heatwaves, or cyclone-induced rainfall $> 50\,\text{mm/hr}$) violate historical training distributions. During such events, the uncertainty spread ($P_{90} - P_{10}$) widens substantially, and the system flags recommendations as `HIGH_RISK`.

### 5.2 Micro-Spatial Informal Friction
The ML model predicts latent customer footfall demand based on spatial-temporal features. However, real-world street vending is constrained by informal physical factors uncaptured in digital maps:
- **Municipal Eviction Zones**: Periodic municipal clearing drives prevent vendors from parking in designated high-demand cells.
- **Informal Territoriality**: Established vendor associations or cart unions often reserve prime street corners, preventing new or relocating vendors from occupying high-SHAP cells.
- **Physical Cart Feasibility**: Pavement width, absence of vehicle ramps, open gutters, and lack of nearby shade or potable water sources can render an otherwise high-footfall cell practically unviable.

---

## 6. Model Governance & Operational Guardrails

To operate safely within these known boundaries, the GeoDemand AI platform implements five strict operational safeguards:

1. **Non-Crossing Monotonicity**: Predictions strictly satisfy $0 \le P_{10} \le P_{50} \le P_{90}$ via deterministic clipping, eliminating inversion artifacts in extreme feature spaces.
2. **Conservative Fallback on Drift**: If feature drift exceeds $\text{PSI} \ge 0.15$ or 7-day MAE degrades by $> 20\%$, the system issues operational warnings, triggers CT pipelines, and falls back to conservative lower-bound estimates ($P_{10}$).
3. **Sub-Second Rollback Guarantee**: If a promoted candidate model exhibits post-deployment anomalies, the platform restores the verified previous champion within $< 15\,\text{ms}$ via `ModelPromotionGate.rollback_to_backup()`.
4. **Explicit Risk Classification**: The business recommender tags every prediction with uncertainty risk levels (`LOW`, `MODERATE`, `HIGH`), preventing vendors from making high-cost relocation investments during periods of high predictive variance.
5. **Human-in-the-Loop Override**: Municipal zone flags and vendor feedback can immediately blacklist specific H3 cells from recommendation candidates, overriding raw model predictions.
