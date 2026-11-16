# GeoDemand AI — Production Model & Data Monitoring Specification

> **Document Status:** Authoritative Monitoring Specification  
> **Target Version:** GeoDemand AI v2.0  
> **Primary Modules:** `src/geodemand/monitoring/drift.py`, `src/geodemand/monitoring/performance.py`  
> **Reference Profile:** `models/reference_distributions.json`

---

## 1. Monitoring Principles

1. **Non-Breaking Isolation**: Monitoring routines run asynchronously or out-of-band; monitoring failures must **never** degrade or block real-time vendor recommendations.
2. **Dual-Signal Drift Detection**: Requires both statistical significance ($p < 0.05$) and practical effect size ($\text{PSI} \ge 0.10$ or $\text{KS} \ge 0.10$) to prevent false alarms caused by large sample sizes ($N > 10,000$).
3. **Delayed Ground-Truth Grounding**: Evaluates actual sales observations against predictions once ground truth is ingested, with explicit rolling window degradation alerts.

---

## 2. Statistical Drift Metrics & Formulations

### Population Stability Index (PSI)
Binned across decile boundaries established on the reference training set:
$$\text{PSI} = \sum_{b=1}^{B} \left( P_{\text{actual}}(b) - P_{\text{expected}}(b) \right) \cdot \ln\left( \frac{P_{\text{actual}}(b)}{P_{\text{expected}}(b)} \right)$$

* Smoothing factor $\epsilon = 10^{-4}$ applied to eliminate zero-frequency division.
* **Interpretation**:
  - $\text{PSI} < 0.10$: Stable; no significant shift.
  - $0.10 \le \text{PSI} < 0.15$: Moderate shift; trigger automated warning.
  - $\text{PSI} \ge 0.15$: **Critical Data Drift Trigger**; candidate retraining initiated.

### Kolmogorov-Smirnov (KS) Test
Measures the maximum vertical distance between empirical cumulative distribution functions:
$$D_{\text{KS}} = \sup_{x} |F_{\text{actual}}(x) - F_{\text{reference}}(x)|$$
* Benjamini-Hochberg False Discovery Rate (FDR) adjustment applied across all 45 feature p-values.

### Wasserstein Distance (Earth Mover's Distance)
Measures the physical distance required to transform the current distribution into the reference distribution:
$$l_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)|\, dx$$
* Provides physical unit interpretability (e.g. temperature shifted by $3.2^\circ\text{C}$).

---

## 3. Delayed Performance & 7-Day MAE Tracking

Operated by [`PerformanceMonitor.assess_rolling_degradation()`](file:///c:/Users/Lenovo/projects/GeoDemand/src/geodemand/monitoring/performance.py):
* Compares 7-day trailing MAE ($\text{MAE}_{\text{7d}}$) against champion validation baseline ($\text{MAE}_{\text{baseline}} = 0.7693$).
* **Degradation Formula**:
  $$\Delta_{\text{deg}}\% = \frac{\text{MAE}_{\text{7d}} - \text{MAE}_{\text{baseline}}}{\text{MAE}_{\text{baseline}}} \times 100$$
* **Retraining Trigger**: If $\Delta_{\text{deg}}\% \ge 20.0\%$, a critical performance alert is raised, initiating the continuous training cycle.

---

## 4. Operational Thresholds Matrix

| Monitoring Metric | Scope | Warning Threshold | Critical Retraining Trigger |
| :--- | :--- | :---: | :---: |
| **Feature PSI** | Per Feature | $\text{PSI} \ge 0.10$ | $\text{PSI} \ge 0.15$ |
| **Max Feature PSI** | Global System | $\text{Max PSI} \ge 0.10$ | $\text{Max PSI} \ge 0.15$ |
| **KS Statistic** | Continuous Features | $D_{\text{KS}} \ge 0.05$ | $D_{\text{KS}} \ge 0.10$ (with $p_{\text{adj}} < 0.05$) |
| **7-Day MAE Degradation** | Ground Truth | $+10.0\%$ | **$+20.0\%$** |
| **Quantile Inversion Rate** | Predictions | $> 0.01\%$ | **$> 0.10\%$** |
| **Missing Feature Rate** | Ingestion | $> 1.0\%$ | **$> 5.0\%$** |
| **Inference Latency (p95)** | Serving | $> 15\,\text{ms}$ | **$> 25\,\text{ms}$** |

---

## 5. Execution & Verification

Run end-to-end monitoring verification:
```powershell
.\.venv\Scripts\python.exe scripts/verify_ml_monitoring.py
```
Regenerate reference distribution profile from training set:
```powershell
.\.venv\Scripts\python.exe scripts/generate_reference_profile.py
```
