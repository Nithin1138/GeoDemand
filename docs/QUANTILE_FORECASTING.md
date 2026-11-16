# GeoDemand AI — Calibrated Quantile Forecasting & Uncertainty Estimation

> **Document Status:** Authoritative Uncertainty Modeling Specification (Phase 20.2.4)  
> **Target Version:** GeoDemand AI v2.1.0-prod  
> **Primary Heads:** P10 ($\alpha=0.10$), P50 ($\alpha=0.50$), P90 ($\alpha=0.90$)  
> **Empirical Coverage Target:** 80% Prediction Interval ($P_{90} - P_{10}$)  
> **Production Status:** Verified in Phase 17.2 & 18.2 (Zero Inversions, 82.4% Empirical Coverage)

---

## 1. Statistical Principles: Prediction Intervals vs. Confidence Intervals

> [!IMPORTANT]
> **CRITICAL CONCEPTUAL DISTINCTION**:
> * A **Confidence Interval** estimates the sampling uncertainty of an estimated *parameter* (e.g., population mean $\mu$ or regression slope $\beta$). It narrows to zero as sample size $N \to \infty$.
> * A **Prediction Interval** estimates the bounds within which a *future individual observation* $Y_{t+1}$ will fall with a given probability (e.g., $80\%$). It accounts for both parametric uncertainty and irreducible irreducible data noise (heteroscedastic spatial-temporal variance).
> 
> GeoDemand AI computes **Prediction Intervals** via Quantile Regression, reflecting the true heteroscedastic noise in mobile vendor footfall. These outputs are **never** to be described as confidence intervals.

---

## 2. Quantile Loss Formulation (Pinball Loss)

The quantile heads optimize asymmetric pinball loss for target quantile $\alpha \in \{0.10, 0.50, 0.90\}$:
$$\mathcal{L}_{\alpha}(y, \hat{q}_\alpha) = \frac{1}{N} \sum_{i=1}^N \max\left( \alpha(y_i - \hat{q}_{\alpha, i}), (1 - \alpha)(\hat{q}_{\alpha, i} - y_i) \right)$$

```
Loss
 │                   / (Slope = 1 - alpha)
 │                  /
 │                 /
 │  (Slope = alpha)/
 │                /
 └───────────────┼──────────────── Prediction Error (y - q_hat)
                 0
```

### Head Roles in Production:
* **P10 ($\alpha=0.10$) — Downside Risk Floor**:
  - Heavily penalizes overpredictions by factor $(1 - 0.10) = 0.90$.
  - Generates a conservative demand floor where demand will exceed this number 90% of the time.
  - Used by mobile micro-retailers for non-perishable inventory purchase and minimum revenue guarantees.
* **P50 ($\alpha=0.50$) — Median Customer Demand**:
  - Symmetric absolute error ($L_1$ loss).
  - Represents the most probable expected footfall count, directly informing baseline route planning.
* **P90 ($\alpha=0.90$) — Upside Surge Ceiling**:
  - Heavily penalizes underpredictions by factor $\alpha = 0.90$.
  - Establishes peak capacity requirement where demand will only exceed this ceiling 10% of the time.
  - Used for preparation prep limits, staff allocation, and surge cart placement.

---

## 3. Dedicated LightGBM Quantile Architecture

Rather than assuming parametric Gaussian error distributions ($\hat{y} \pm 1.28\sigma$), which fail in skewed street-retail demand, GeoDemand AI trains separate, non-parametric LightGBM quantile regressors:

```python
# P10 Lower Bound Model (alpha = 0.10)
p10_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.10,
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    random_state=42,
)

# P50 Median Point Model (alpha = 0.50 / regression_l1)
p50_model = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=450,
    learning_rate=0.05,
    num_leaves=95,
    min_child_samples=25,
    random_state=42,
)

# P90 Upper Bound Model (alpha = 0.90)
p90_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.90,
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    random_state=42,
)
```

---

## 4. Post-Hoc Monotonicity & Non-Crossing Enforcement

Because the three quantile models are trained independently, empirical quantile crossing ($P_{10} > P_{50}$ or $P_{50} > P_{90}$) can mathematically occur in under-sampled regions of feature space.

To guarantee zero crossing violations and enforce physical domain constraints, `DemandModelPredictor.predict_batch_quantiles()` in [`src/geodemand/models/inference.py`](file:///c:/Users/Lenovo/projects/GeoDemand/src/geodemand/models/inference.py) executes sequential monotonic clipping:

$$P_{10} = \max\left(0, \text{round}(\hat{q}_{10})\right)$$
$$P_{50} = \max\left(P_{10}, \text{round}(\hat{q}_{50})\right)$$
$$P_{90} = \max\left(P_{50}, \text{round}(\hat{q}_{90})\right)$$

### Formal Invariants Guaranteed:
1. **Physical Non-Negativity**: $\forall q \in \{10, 50, 90\}, P_q \ge 0$.
2. **Strict Ordering**: $0 \le P_{10} \le P_{50} \le P_{90}$.
3. **Integral Unit**: Output is integer customer counts.
4. **Zero Inversions**: Invariant verified across 100% of integration test suites (`test_quantile_ordering_monotonicity`).

---

## 5. Empirical Validation & Operational Monitoring

Evaluated in Phase 17.2 & 18.2 across 26,700 out-of-sample holdout test vectors:

| Evaluation Metric | Nominal Target | Measured (Production `v2.1.0`) | Assessment |
| :--- | :---: | :---: | :---: |
| **80% Quantile Coverage** | $80.0\%$ | **$82.4\%$** | Calibrated within $[70\%, 95\%]$ |
| **Quantile Inversion Rate** | $0.0\%$ | **$0.00\%$** | Strictly zero crossings |
| **Mean Interval Width ($P_{90} - P_{10}$)** | $< 4.0$ customers | **$3.60$ customers** | Sharp, informative interval |
| **Non-negative Output Rate** | $100.0\%$ | **$100.0\%$** | Zero negative predictions |
| **Quantile Scoring Latency** | $< 1.0\text{ ms}$ | **$0.015\text{ ms}$** | Over 60× faster than SLA |

### Monitoring & Drift Gates:
In `src/geodemand/monitoring/operational_monitor.py::QuantileCoverageMonitor`:
- **Under-coverage alert**: If empirical coverage drops below $70.0\%$, an alert is triggered (model underestimates variance; vendor risks stockouts).
- **Over-coverage alert**: If empirical coverage exceeds $95.0\%$, an alert is triggered (model intervals are excessively wide/uninformative).
- **Crossing violation alert**: Any detected crossing ($P_{10} > P_{90}$) triggers an immediate fatal telemetry event.

---

## 6. Downstream Consumption in Decision Intelligence

The business economics and recommender layers (`src/api/recommender.py` and `src/api/providers/economics.py`) consume the quantile forecasts directly:

```
Quantile Triplet [P10, P50, P90]
        │
        ├── P10 ──► Minimum Guaranteed Revenue = P10 * AOV
        │           Perishable Inventory Sizing = min(Capacity, P10 * Batch_Ratio)
        │
        ├── P50 ──► Expected Revenue = P50 * AOV
        │           Net Relocation Profit = Expected Revenue - Relocation Cost
        │           Primary Ranking Objective for STAY vs MOVE
        │
        └── P90 ──► Maximum Surge Capacity = min(Capacity, P90)
                    Peak Hour Supply Prep Allocation
                    Uncertainty Spread Score = P90 - P10
```

### Risk Classification:
- **Low Risk ($\text{Spread} \le 5$)**: High spatial confidence (e.g. established commercial IT park during noon lunch).
- **Moderate Risk ($6 \le \text{Spread} \le 12$)**: Standard volatility.
- **High Risk ($\text{Spread} > 12$)**: Volatile demand (e.g. onset of monsoon rains, major transit disruption).
