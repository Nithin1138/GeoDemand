# GeoDemand AI — Production Champion Selection Decision (Phase 13.2)

> [!IMPORTANT]
> **Decision**: `Approved_LightGBM_v2.1.0` is **RETAINED** as the primary production champion for the live API recommender.
> **Rollback Status**: Prior production artifact verified and backed up at `models/demand_model_champion_v2.1.0.pkl`.

## 1. Documented Promotion Criteria Evaluation

| Evaluation Criterion | Specification Gate | Approved LightGBM v2.1.0 | TFT / N-HiTS / PatchTST | Result |
| :--- | :--- | :--- | :--- | :---: |
| **Inference Latency** | $\le 3.0\text{ms}$ p95 per location | **1.75 ms** (Passed) | 14.2 – 19.4 ms (Exceeded by 5-6x) | **LightGBM Approved** |
| **Next-Hour Generalization** | Lowest test MAE on forward split | **0.7565** | 0.7712 – 0.7850 | **LightGBM Approved** |
| **Quantile Coverage Calibration** | 80% coverage in $[75\%, 85\%]$ | **79.9%** (Target: 80%) | 78.5% – 81.2% | **All Models Passed** |
| **Quantile Non-Crossing** | $P_{10} \le P_{50} \le P_{90}$ everywhere | **0.0% Inversion Rate** | 0.0% Inversion Rate | **All Models Passed** |
| **Memory & Operational Footprint** | $\le 50\text{{MB}}$, zero GPU requirement | **18.2 MB** | 120 – 165 MB (Requires GPU for scale) | **LightGBM Approved** |
| **Explainability Support** | Native TreeSHAP sub-millisecond | **Supported** | Requires KernelSHAP (High latency) | **LightGBM Approved** |

## 2. Shift Optimizer Role Allocation

For full-shift multi-stop route optimization (horizons $H=2..8\text{h}$):
- **`Recursive_Autoregressive_LightGBM`** is approved as the backend multi-horizon engine for Member 3's Shift Optimizer.
- Evaluates recursive autoregressive trajectories with 3.2ms latency, preserving responsive user interaction.

## 3. Rollback Safety & Safeguards

1. **Immutable Champion Artifact**: `models/demand_model_champion_v2.1.0.pkl` stored and locked.
2. **Automatic Rollback Trigger**: Any candidate exhibiting $>20\%$ MAE degradation, PSI $>0.15$, or latency $>3.0\text{ms}$ is rejected by `ModelPromotionGate`.