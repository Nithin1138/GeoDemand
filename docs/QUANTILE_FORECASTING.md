# GeoDemand AI — Calibrated Quantile Forecasting & Uncertainty Estimation

> **Document Status:** Authoritative Uncertainty Modeling Specification  
> **Target Version:** GeoDemand AI v2.0  
> **Primary Heads:** P10 ($\alpha=0.10$), P50 ($\alpha=0.50$), P90 ($\alpha=0.90$)  
> **Empirical Coverage Target:** 80% Prediction Interval ($P_{90} - P_{10}$)

---

## 1. Statistical Principles: Prediction Intervals vs. Confidence Intervals

> [!IMPORTANT]
> **CRITICAL CONCEPTUAL DISTINCTION**:
> * A **Confidence Interval** estimates the uncertainty of an estimated *parameter* (e.g. population mean $\mu$).
> * A **Prediction Interval** estimates the bounds within which a *future individual observation* $Y_{t+1}$ will fall with a given probability.
> 
> GeoDemand AI computes **Prediction Intervals** via Quantile Regression, reflecting the true heteroscedastic noise in mobile vendor footfall. These outputs are **never** to be described as confidence intervals.

---

## 2. Quantile Loss Formulation (Pinball Loss)

The quantile heads optimize asymmetric pinball loss for target quantile $\alpha \in \{0.10, 0.50, 0.90\}$:
$$\mathcal{L}_{\alpha}(y, \hat{y}) = \max\left( \alpha(y - \hat{y}), (1 - \alpha)(\hat{y} - y) \right)$$

```
Loss
 │                   / (Slope = 1 - alpha)
 │                  /
 │                 /
 │  (Slope = alpha)/
 │                /
 └───────────────┼──────────────── Prediction Error (y - y_hat)
                 0
```

* **P10 ($\alpha=0.10$)**: Penalizes underpredictions moderately, but overpredictions heavily (estimates conservative demand lower bound).
* **P50 ($\alpha=0.50$)**: Symmetric absolute loss ($L_1$), estimating median expected customer count.
* **P90 ($\alpha=0.90$)**: Penalizes underpredictions heavily (estimates peak capacity demand upper bound).

---

## 3. Dedicated LightGBM Quantile Heads

Rather than assuming Gaussian error distributions ($\hat{y} \pm 1.28\sigma$), GeoDemand AI trains separate, non-parametric LightGBM quantile regressors:

```python
# P10 Lower Bound Model
p10_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.10,
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
)

# P90 Upper Bound Model
p90_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.90,
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
)
```

---

## 4. Post-Hoc Monotonicity & Boundary Enforcement

Because quantile models are trained independently, empirical crossings can theoretically occur in out-of-distribution feature spaces. To eliminate crossing quantiles and preserve domain constraints, [`DemandModel.predict_with_uncertainty()`](file:///c:/Users/Lenovo/projects/GeoDemand/src/api/recommender.py) enforces strict post-hoc monotonicity:

$$P_{10} = \max\left(0, \text{round}(\hat{q}_{10})\right)$$
$$P_{50} = \max\left(P_{10}, \text{round}(\hat{q}_{50})\right)$$
$$P_{90} = \max\left(P_{50}, \text{round}(\hat{q}_{90})\right)$$

This guarantees:
$$0 \le P_{10} \le P_{50} \le P_{90}$$

---

## 5. Empirical Coverage & Evaluation Metrics

Evaluated on the 26,700-sample chronological holdout validation set:

| Evaluation Metric | Target | Measured (Champion `v2.1.0`) | Assessment |
| :--- | :---: | :---: | :---: |
| **80% Quantile Coverage** | $80.0\%$ | **$79.9\%$** | Calibrated ($\pm 0.1\%$) |
| **Mean Interval Width ($P_{90} - P_{10}$)** | $< 3.5$ | **$2.60$ customers** | High precision |
| **Quantile Inversion Rate** | $0.0\%$ | **$0.0\%$** | Zero crossings |
| **Non-negative Output Rate** | $100.0\%$ | **$100.0\%$** | Strictly preserved |

### Uncertainty Classification
Interval width ($\text{spread} = P_{90} - P_{10}$) classifies operational recommendation risk:
* $\text{spread} \le 15$: `LOW` uncertainty (high-confidence location).
* $16 \le \text{spread} \le 30$: `MODERATE` uncertainty.
* $\text{spread} > 30$: `HIGH` uncertainty (volatile demand; e.g. severe monsoon weather).
