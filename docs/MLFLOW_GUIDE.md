# GeoDemand AI — MLflow Experiment Tracking & Model Registry Runbook

> **Document Status:** Authoritative MLOps Operational Guide  
> **Target Version:** GeoDemand AI v2.0  
> **Backend Storage:** Local SQLite URI (`sqlite:///mlruns/geodemand_mlflow.db`)  
> **Artifact Root:** `mlruns/`

---

## 1. MLOps Lifecycle Overview

GeoDemand AI leverages MLflow for reproducible experiment tracking, parameter versioning, model artifact storage, and lifecycle stage management:

```
[Feature Store Parquet] ──► [Training Pipeline] ──► [MLflow Run Logger]
                                     │                        │
                                     ▼                        ▼
                          [Model Promotion Gate] ──► [Artifacts & Checkpoints]
                                     │                        │
                                     ▼                        ▼
                         [models/demand_model.pkl] ◄── [Champion Snapshot]
```

---

## 2. Tracking Configuration & Storage

### Connection & Tracking URI
The tracking engine initializes via [`src/geodemand/monitoring/mlflow_logger.py`](file:///c:/Users/Lenovo/projects/GeoDemand/src/geodemand/monitoring/mlflow_logger.py):
```python
import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlruns/geodemand_mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```

### Canonical Experiments
| Experiment Name | Purpose |
| :--- | :--- |
| `GeoDemand_Demand_Forecasting` | Model training, multi-model benchmarking, and Optuna tuning |
| `GeoDemand_Continuous_Training` | Automated drift-triggered and scheduled retraining runs |
| `GeoDemand_Model_Promotion` | Gating verification events, candidate approvals, and rollbacks |
| `GeoDemand_Monitoring` | Periodic feature drift, KS tests, and PSI reports |

---

## 3. What Is Tracked in Every Run

For every training or evaluation run, the pipeline logs:
1. **Parameters**:
   - `algorithm`: e.g. `LightGBM`, `XGBoost`, `CatBoost`
   - `objective`: e.g. `regression_l1`, `quantile`
   - `learning_rate`, `n_estimators`, `num_leaves`, `max_depth`
   - `feature_count`: 45
   - `training_dataset_hash`: SHA256 of `feature_store.parquet`
   - `random_seed`: 42
2. **Metrics**:
   - Validation & Test `mae`, `rmse`, `r2`
   - `latency_p50_ms`, `latency_p95_ms`
   - Quantile coverage: `quantile_coverage_80`, `quantile_mean_width`
3. **Artifacts**:
   - Model bundle (`demand_model.pkl`)
   - Quantile head models (`p10_model.joblib`, `p90_model.joblib`)
   - Schema JSON (`feature_schema.json`)
   - Encoders (`label_encoders.joblib`)
   - Metadata sidecar (`metadata.json`)
   - Reference distributions (`reference_distributions.json`)

---

## 4. Promotion Policy & Automated Gating

Before any candidate replaces `models/demand_model.pkl`, it must pass all criteria in [`ModelPromotionPolicy`](file:///c:/Users/Lenovo/projects/GeoDemand/src/geodemand/models/promotion.py#L53-L70):
* $\text{Val MAE} \le 1.00$ (Hard ceiling)
* $\text{Val RMSE} \le 1.30$
* $\text{Val MAE}_{\text{candidate}} \le \text{Val MAE}_{\text{champion}}$ (Zero regression allowed)
* Latency $P_{95} \le 25\,\text{ms}$
* Quantile inversion rate $< 0.1\%$ ($P_{10} \le P_{50} \le P_{90}$)
* Roundtrip serialization test passed
* TreeSHAP compatibility verified

---

## 5. Rollback Procedures

If an unexpected production degradation occurs after promotion, execute instant rollback:

```powershell
# Rollback using CLI script
.\.venv\Scripts\python.exe scripts/promote_model.py --rollback
```

Or programmatically:
```python
from src.geodemand.models.promotion import ModelPromotionGate

gate = ModelPromotionGate()
gate.rollback()
```

The system atomically restores the prior champion snapshot from `models/champion_backup/` without API downtime.
