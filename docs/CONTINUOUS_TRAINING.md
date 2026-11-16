# GeoDemand AI — Continuous Training (CT) Pipeline Architecture

> **Document Status:** Authoritative CT Specification  
> **Target Version:** GeoDemand AI v2.0  
> **Core Module:** `src/geodemand/monitoring/ct_pipeline.py`  
> **Audit Log:** `models/retraining_audit_log.json`

---

## 1. Continuous Training Architecture

The GeoDemand Continuous Training (CT) pipeline governs automated model retraining, verification, and atomic deployment:

```
┌────────────────────────────────────────────────────────┐
│                   RETRAINING TRIGGERS                  │
│   • Data Drift (PSI >= 0.15)                           │
│   • 7-Day MAE Degradation (>= 20%)                     │
│   • Scheduled (Weekly Cron / GitHub Actions)           │
│   • Manual Operator Invocation                         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               DATA SUFFICIENCY & ELIGIBILITY           │
│   • Minimum Sample Count Threshold (>= 500 samples)    │
│   • Target Health: Non-negative, non-null, finite      │
│   • Strict Chronological Boundary (Zero Leakage)       │
└───────────────────────────┬────────────────────────────┘
                            │
              Eligible? ────┴──── No ──► [Abort: SKIPPED_INSUFFICIENT_DATA]
              │
              ▼ Yes
┌────────────────────────────────────────────────────────┐
│               TRAIN CHALLENGER CANDIDATE               │
│   • Chronological 70/15/15 Data Split                  │
│   • Multi-Model Benchmark (LightGBM, XGBoost, CatBoost)│
│   • Calibrate P10, P50, P90 Quantile Heads             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            MODEL PROMOTION GATE (POLICY CHECK)         │
│   • Val MAE_candidate <= Val MAE_champion              │
│   • Single-Sample Latency (p95) <= 25ms                │
│   • Monotonic Quantiles (P10 <= P50 <= P90)            │
│   • TreeSHAP Compatibility & Roundtrip Serialization   │
└───────────────────────────┬────────────────────────────┘
                            │
                Pass? ──────┴──── No ──► [Retain Current Champion: REJECTED]
                │
                ▼ Yes
┌────────────────────────────────────────────────────────┐
│             ATOMIC PROMOTION & ROLLBACK BACKUP         │
│   1. Snapshot current champion to champion_backup/     │
│   2. Atomically swap candidate to demand_model.pkl     │
│   3. Update demand_model_meta.json                     │
│   4. Log event to retraining_audit_log.json & MLflow   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Training Data Eligibility Criteria

Before triggering expensive training runs, [`ContinuousTrainingPipeline.check_data_eligibility()`](file:///c:/Users/Lenovo/projects/GeoDemand/src/geodemand/monitoring/ct_pipeline.py) verifies:
1. **Sample Volume**: Dataset must contain at least `min_new_samples` (default $500$).
2. **Target Integrity**: `expected_customer_count` must be present, numeric, non-null, non-negative, and finite.
3. **Missing Value Bounds**: Feature missingness across all columns must remain $< 5.0\%$.
4. **Temporal Consistency**: `decision_timestamp` must be strictly monotonically increasing.

If any check fails, the pipeline exits safely with status `"SKIPPED_INSUFFICIENT_DATA"`, preventing model degradation.

---

## 3. Retraining Triggers

1. **Drift-Triggered (`DRIFT_TRIGGERED`)**:
   - Activated when `max_psi >= 0.15` across any canonical feature in `DriftReport`.
2. **Performance-Degraded (`PERFORMANCE_DEGRADATION`)**:
   - Activated when trailing 7-day actuals MAE degrades by $\ge 20\%$ relative to the champion baseline.
3. **Scheduled (`SCHEDULED`)**:
   - Run periodically via CI/CD (e.g. GitHub Actions `.github/workflows/ml-ci.yml` or Airflow DAG).
4. **Manual (`MANUAL`)**:
   - Triggered via CLI:
     ```powershell
     .\.venv\Scripts\python.exe scripts/run_ml_workflow.py --full --promote
     ```

---

## 4. Promotion Safeguards & Human Approval

* **Automated Mode** (`require_manual_approval=False`):
  If the candidate strictly outperforms the champion on validation MAE with zero contract violations, it is atomically promoted.
* **Staged Governance Mode** (`require_manual_approval=True`):
  Candidate is saved to `models/experiments/staged_<cycle_id>.joblib` and an `approval_token` is generated. Deployment requires explicit operator sign-off.
* **Hard Invariance**: An inferior candidate (higher MAE, failing quantiles, or excessive latency) is **never** promoted.

---

## 5. Audit Trail & Rollback Runbook

Every retraining cycle records an entry in `models/retraining_audit_log.json`:
```json
{
  "cycle_id": "ct_20261116_082000_a1b2c3",
  "trigger": "DRIFT_TRIGGERED",
  "status": "COMPLETED_PROMOTED",
  "timestamp": "2026-11-16T08:20:00Z",
  "champion_id": "v2.1.0",
  "candidate_id": "v2.2.0",
  "champion_val_mae": 0.7693,
  "candidate_val_mae": 0.7512,
  "rollback_available": true
}
```

To rollback:
```python
from src.geodemand.monitoring.ct_pipeline import ContinuousTrainingPipeline

ct = ContinuousTrainingPipeline()
ct.rollback_to_previous_champion()
```
The previous champion snapshot is atomically restored within $< 100\,\text{ms}$.
