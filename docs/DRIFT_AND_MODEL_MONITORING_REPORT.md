# GeoDemand AI — Drift & Model Monitoring Report (Phase 13.4)

**Document ID:** `MOVIGO-MLOPS-PHASE13.4-DRIFT-MONITORING`  
**Author:** Senior Machine Learning Engineer & MLOps Architect (Member 2)  
**Target:** Next-Hour Expected Customer Count (`expected_customer_count`)  
**Status:** Approved & Verified in Production Pipeline  
**Timestamp:** `2026-11-16T12:05:00Z`  

---

## 1. Executive Summary

Phase 13.4 implements a production-grade, mathematically rigorous **Drift Detection and Model Monitoring Framework** for GeoDemand AI. This system guarantees that spatial, temporal, categorical, and behavioral shifts in vendor environments are detected in real time, preventing silent model degradation and automating safe retraining workflows.

### Core Capabilities Delivered:
1. **Evidently AI Adapter**: Standardized integration with Evidently AI `DataDriftPreset` and a zero-dependency native fallback generating machine-readable JSON metrics and standalone HTML dashboards.
2. **Population Stability Index (PSI)**: Decile-binned continuous PSI and Laplace-smoothed categorical frequency PSI.
3. **Wasserstein Distance (Earth Mover's Distance)**: Physical distribution shift measurement via $L_1$ optimal transport.
4. **Input Feature Drift Monitoring**: Combined statistical ($p < 0.05$ with Benjamini-Hochberg FDR) and practical significance filtering ($D \ge 0.10$, $\text{PSI} \ge 0.15$).
5. **Prediction Drift Monitoring**: Real-time evaluation of $y_{pred}$ distributions, mean prediction shifts, and quantile movements ($P_{10}, P_{50}, P_{90}$).
6. **7-Day Rolling MAE Monitoring**: Delayed actuals evaluation measuring relative MAE degradation against champion baseline.
7. **Production Triggers**: Automated trigger on **Drift PSI $\ge 0.15$** and **7-Day MAE degradation $\ge 20\%$**.
8. **Delayed Actuals & Ground-Truth Guards**: Explicit protection against training on incomplete or synthetic data ($\ge 500$ verified actual customer records required).
9. **API Model Health Endpoints**: Real-time health status exposed at `/v1/system/model-health` and `/v1/model/health`.

---

## 2. Mathematical Formulations & Metrics

### 2.1 Population Stability Index (PSI)
For continuous features partitioned into $k$ reference decile bins $B_1, \dots, B_k$:

$$\text{PSI} = \sum_{i=1}^k (A_i - E_i) \cdot \ln\left(\frac{A_i}{E_i}\right)$$

Where:
- $E_i = \frac{N_{ref, i} + \epsilon}{N_{ref} + k\epsilon}$ is the reference proportion in bin $i$.
- $A_i = \frac{N_{cur, i} + \epsilon}{N_{cur} + k\epsilon}$ is the current proportion in bin $i$.
- $\epsilon = 10^{-4}$ prevents division-by-zero or $\ln(0)$ instabilities.

**Operational Thresholds:**
- $\text{PSI} < 0.10$: **Nominal** (No significant change).
- $0.10 \le \text{PSI} < 0.15$: **Moderate Drift** (Warning, increased monitoring).
- $\text{PSI} \ge 0.15$: **Critical Drift** (Activates automated retraining trigger).

### 2.2 Wasserstein Distance (First Earth Mover's Distance)
Computes the minimal work required to transform the current empirical cumulative distribution $F_{cur}(x)$ into the reference distribution $F_{ref}(x)$:

$$W_1(F_{ref}, F_{cur}) = \int_{-\infty}^{\infty} |F_{ref}(x) - F_{cur}(x)| \, dx$$

Unlike bin-dependent metrics, $W_1$ operates on continuous physical units (e.g., customers/hour, degrees Celsius, mm rainfall) and provides scale-aware physical shift magnitudes.

### 2.3 Two-Sample Kolmogorov-Smirnov Test with Multiple-Testing Correction
For continuous feature $j$:

$$D_j = \sup_x |F_{ref, j}(x) - F_{cur, j}(x)|$$

To prevent false alarms in large sample sizes ($N > 1,000$ where even negligible noise yields $p < 0.05$):
1. **Benjamini-Hochberg False Discovery Rate (FDR)**: Adjusts raw $p$-values across all $M$ features:
   $$p_{(i)}^{adj} = \min_{k \ge i} \left(\min\left(1, \frac{M}{k} p_{(k)}\right)\right)$$
2. **Practical Significance Filter**: A feature is flagged as drifted if and only if:
   $$(p_j^{adj} < 0.05 \land D_j \ge 0.10) \lor \text{PSI}_j \ge 0.15$$

---

## 3. Monitoring Architecture & Components

```
                   ┌─────────────────────────────────────────┐
                   │        Live Production Traffic          │
                   │  - Input Features (X)                   │
                   │  - Predicted Customers (y_pred)         │
                   │  - Delayed Actual Customers (y_true)    │
                   └───────────────────┬─────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
   │   FeatureDrift   │       │  PredictionDrift │       │7-Day Rolling Perf│
   │     Detector     │       │     Detector     │       │     Monitor      │
   │  - Continuous KS │       │  - Pred PSI      │       │  - MAE Tracking  │
   │  - Feature PSI   │       │  - Wasserstein   │       │  - Degradation % │
   │  - Wasserstein   │       │  - Quantile Shift│       │  - Delayed Actual│
   │  - Chi2 Categoric│       │  - KS Test       │       │    Safe Handling │
   └────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                   ┌─────────────────────────────────────────┐
                   │           ModelHealthProvider           │
                   │  - Synthesizes Health Status            │
                   │  - Evaluates Retraining Triggers        │
                   │  - Checks Ground Truth Sufficiency      │
                   └───────────────────┬─────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   ┌──────────────────┐                                  ┌──────────────────┐
   │ API Health Server│                                  │CT Retraining Gate│
   │ GET /v1/system/  │                                  │ - PSI >= 0.15    │
   │   model-health   │                                  │ - MAE deg >= 20% │
   │ GET /v1/model/   │                                  │ - N >= 500 GT    │
   │   health         │                                  └──────────────────┘
   └──────────────────┘
```

---

## 4. Retraining Trigger Policy Matrix

| Trigger Type | Parameter | Threshold | Action Taken |
| :--- | :--- | :---: | :--- |
| **Feature Drift** | Max Feature PSI | $\ge 0.15$ | Activates `DRIFT_TRIGGERED` retraining cycle |
| **Prediction Drift** | Prediction Output PSI | $\ge 0.15$ | Activates `DRIFT_TRIGGERED` retraining cycle |
| **Performance Degradation** | 7-Day Rolling MAE Change | $\ge +20.0\%$ | Activates `PERFORMANCE_DEGRADATION` retraining cycle |
| **Data Sufficiency Gate** | Labeled Ground Truth Samples | $< 500$ | **ABORTS** retraining (`SKIPPED_INSUFFICIENT_DATA`) |
| **Missing Rate Gate** | Null Feature Proportion | $> 5.0\%$ | **ABORTS** retraining |
| **Target Validity Gate** | Negative or Infinite Labels | $> 0$ | **ABORTS** retraining |

---

## 5. Model Health API Specification

### Endpoint: `GET /v1/system/model-health` (and `GET /v1/model/health`)
Provides complete visibility into active model version, drift state, 7-day performance, and retraining readiness.

#### Example Response:
```json
{
  "timestamp": "2026-11-16T12:05:00.000000+00:00",
  "status": "HEALTHY",
  "model_version": "v2.1.0",
  "model_type": "lightgbm",
  "champion_path": "models/demand_model.pkl",
  "loaded_at": "2026-11-16T08:00:00+00:00",
  "baseline_metrics": {
    "val_mae": 0.7693,
    "val_rmse": 1.0045,
    "test_mae": 0.7712,
    "quantile_coverage_80": 0.8020
  },
  "feature_drift": {
    "status": "NOMINAL",
    "max_psi": 0.0412,
    "mean_psi": 0.0185,
    "drifted_features_count": 0,
    "drifted_features": [],
    "psi_trigger_threshold": 0.15,
    "trigger_activated": false
  },
  "prediction_drift": {
    "status": "NOMINAL",
    "psi": 0.0234,
    "wasserstein_distance": 0.4210,
    "ks_statistic": 0.0350,
    "trigger_activated": false,
    "psi_trigger_threshold": 0.15
  },
  "rolling_performance_7d": {
    "status": "NOMINAL",
    "current_7d_mae": 0.7850,
    "baseline_mae": 0.7693,
    "degradation_pct": 2.04,
    "degradation_threshold_pct": 20.0,
    "degradation_alert": false,
    "sample_count": 1250,
    "message": "Stable: 7-day MAE change is +2.04% (Current: 0.7850 vs Baseline: 0.7693)."
  },
  "retraining_trigger_active": false,
  "retraining_trigger_type": null,
  "retraining_trigger_reason": null,
  "ground_truth_status": {
    "sample_count": 1250,
    "min_required_for_retraining": 500,
    "sufficient_for_retraining": true,
    "status": "SUFFICIENT",
    "message": "Ground truth volume verified"
  },
  "zero_downtime_hot_reload_enabled": true,
  "last_retraining_cycle": {
    "cycle_id": "ct_20261116_020000_a1b2c3",
    "status": "COMPLETED_PROMOTED",
    "timestamp": "2026-11-16T02:05:30+00:00"
  }
}
```

---

## 6. Test Suite & Verification Matrix

All 13 Phase 13.4 test cases in `tests/unit/test_phase13_4_monitoring.py` passed with 100% success:

| # | Test Function | Target Verified | Status |
| :-: | :--- | :--- | :---: |
| 1 | `test_evidently_adapter_execution_and_schema` | Evidently adapter execution, JSON schema, and HTML export | **PASS** |
| 2 | `test_psi_calculation_identical_and_shifted` | Decile PSI zero for identical, high for shifted distributions | **PASS** |
| 3 | `test_wasserstein_distance_properties` | Earth Mover's Distance scale and physical shift magnitude | **PASS** |
| 4 | `test_input_feature_drift_detection` | Continuous and categorical feature drift detection | **PASS** |
| 5 | `test_prediction_drift_detector` | Prediction distribution shift, PSI, and quantiles tracking | **PASS** |
| 6 | `test_seven_day_mae_degradation_trigger` | 7-day rolling MAE 20% degradation trigger | **PASS** |
| 7 | `test_roadmap_drift_trigger_threshold` | PSI $\ge 0.15$ retraining trigger activation | **PASS** |
| 8 | `test_false_positive_drift_mitigation` | Practical significance filtering prevents false alarms on $N=10,000$ | **PASS** |
| 9 | `test_missing_actual_label_scenarios` | Delayed actuals and NaN label handling without crashing | **PASS** |
| 10 | `test_avoid_retraining_without_sufficient_ground_truth`| Strictly rejects retraining when labeled samples $< 500$ | **PASS** |
| 11 | `test_trigger_retraining_workflow_satisfaction` | Validates retraining trigger activation on alert conditions | **PASS** |
| 12 | `test_candidate_promotion_and_rollback` | Evaluates promotion gating and verifies snapshot rollback | **PASS** |
| 13 | `test_api_model_health_endpoint` | Verifies `/v1/system/model-health` and `/v1/model/health` schema | **PASS** |

**Total Unit Test Suite:** 103 / 103 Passing  
**Total ML Suite (Unit + Integration):** 159 / 159 Passing
