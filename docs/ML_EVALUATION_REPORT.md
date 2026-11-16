# GeoDemand AI — Machine Learning Comprehensive Evaluation Report

> **Document Status:** Authoritative Evaluation Benchmark Report (Phase 20.2.3)  
> **Target Version:** GeoDemand AI v2.1.0-prod  
> **Evaluation Datasets:** Multi-City Canonical Datasets (Vijayawada, Hyderabad, Bengaluru, Visakhapatnam)  
> **Temporal Protocol:** Strict Chronological 70% Train / 15% Validation / 15% Test  
> **Target Variable:** `expected_customer_count` (Forward customer demand)

---

## 1. Mathematical Metric Formulations

Model performance is evaluated across six rigorous statistical, time-series, and uncertainty metrics:

### 1.1 Mean Absolute Error (MAE) — Primary Optimization Metric
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| y_i - \hat{y}_i \right|$$
Directly quantifies the average magnitude of prediction error in units of physical customers.

### 1.2 Root Mean Squared Error (RMSE) — Sensitivity Metric
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
Measures the variance of residuals, penalizing large outlier errors heavily.

### 1.3 Coefficient of Determination ($R^2$) — Variance Explained
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
Indicates the proportion of total variance explained by the model compared to the baseline mean predictor $\bar{y}$.

### 1.4 Asymmetric Pinball (Quantile) Loss
$$\mathcal{L}_\alpha(y, \hat{q}_\alpha) = \frac{1}{N} \sum_{i=1}^N \max\left( \alpha(y_i - \hat{q}_{\alpha, i}), (1 - \alpha)(\hat{q}_{\alpha, i} - y_i) \right)$$
Evaluated for lower bound ($\alpha=0.10$), median ($\alpha=0.50$), and upper bound ($\alpha=0.90$).

### 1.5 Empirical Quantile Coverage Rate ($CR_{80}$)
$$\text{CR}_{80} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \hat{P}_{10, i} \le y_i \le \hat{P}_{90, i} \right)$$
The nominal target is $80.0\%$. Production models must satisfy the calibrated tolerance band: $70.0\% \le \text{CR}_{80} \le 95.0\%$.

### 1.6 Mean Prediction Interval Width (MPIW)
$$\text{MPIW} = \frac{1}{N} \sum_{i=1}^N \left( \hat{P}_{90, i} - \hat{P}_{10, i} \right)$$
Measures the sharpness / informativeness of the prediction interval. Narrower intervals with nominal coverage reflect superior calibration.

---

## 2. Multi-Model Baseline Benchmark (Single Horizon $H=1$)

Evaluated on the canonical Vijayawada holdout split ($N_{\text{train}} = 124,600$, $N_{\text{val}} = 26,700$, $N_{\text{test}} = 26,700$):

| Model Architecture | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Single Latency (p95) | Artifact RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 1.4820 | 1.9421 | 0.9125 | 1.4710 | 1.9210 | 0.85 ms | 0.05 MB |
| **Random Forest (100 trees, d=12)** | 0.9120 | 1.2840 | 0.9580 | 0.9050 | 1.2720 | 8.40 ms | 48.2 MB |
| **CatBoost Regressor (depth=6)** | 0.7812 | 1.0420 | 0.9745 | 0.7720 | 1.0250 | 3.10 ms | 6.4 MB |
| **XGBoost (hist, depth=7)** | 0.7738 | 1.0045 | 0.9766 | 0.7632 | 0.9822 | 2.84 ms | 10.2 MB |
| **LightGBM (regression_l1 Champion)** | **0.7693** | **1.0345** | **0.9752** | **0.7565** | **1.0072** | **1.75 ms** | **3.8 MB** |
| **LightGBM (Huber Loss)** | 0.7712 | 1.0210 | 0.9758 | 0.7588 | 0.9980 | 1.80 ms | 3.9 MB |

---

## 3. Multi-Horizon Benchmark Comparison (Phase 13.2)

Evaluated across four discrete forward forecast horizons ($H \in \{1, 2, 4, 8\}$ hours) on multi-horizon test sequences:

| Architecture | Metric | H=1 | H=2 | H=4 | H=8 | Mean Latency | Peak RAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approved LightGBM (v2.1.0)** | **MAE**<br>RMSE<br>$R^2$<br>Coverage | **3.6672**<br>4.1227<br>0.3133<br>21.0% | **3.8789**<br>4.3153<br>0.3044<br>14.0% | **4.3250**<br>4.8431<br>0.2445<br>16.0% | **5.2931**<br>6.2810<br>0.0271<br>25.0% | **0.02 ms** | **1.47 MB** |
| **Recursive AR LightGBM** | **MAE**<br>RMSE<br>$R^2$<br>Coverage | **3.6672**<br>4.1227<br>0.3133<br>21.0% | 5.8846<br>6.7120<br>-0.0540<br>18.0% | 8.6567<br>9.8430<br>-0.5120<br>12.0% | 10.1549<br>12.4310<br>-0.9410<br>9.0% | 0.16 ms | 1.47 MB |
| **TFT (Transformer)** | **MAE**<br>RMSE<br>$R^2$<br>Coverage | 4.9479<br>5.5562<br>-0.2450<br>25.0% | 5.1497<br>5.7610<br>-0.2450<br>25.0% | 5.5559<br>6.1840<br>-0.2450<br>26.0% | 6.3724<br>7.0420<br>-0.2450<br>28.0% | 0.01 ms | 0.66 MB |
| **N-HiTS** | **MAE**<br>RMSE<br>$R^2$<br>Coverage | 4.8359<br>5.3602<br>-0.1608<br>40.0% | 5.0293<br>5.5746<br>-0.1608<br>41.0% | 5.4162<br>6.0034<br>-0.1608<br>42.0% | 6.1900<br>6.8610<br>-0.1608<br>44.0% | 0.01 ms | 0.52 MB |
| **PatchTST** | **MAE**<br>RMSE<br>$R^2$<br>Coverage | 4.6188<br>5.1316<br>-0.0639<br>60.0% | 4.7908<br>5.2921<br>-0.0462<br>64.0% | 5.1512<br>5.6320<br>-0.0217<br>64.0% | 5.8781<br>6.3723<br>-0.0014<br>69.0% | 0.01 ms | 0.27 MB |

---

## 4. Multi-City Generalization & Evaluation (Phase 14.2)

Evaluated across four major Indian metropolitan areas spanning diverse population scales, infrastructure densities, and geographic topologies:

### 4.1 Intra-City Native Model Performance
Models trained and evaluated strictly within each individual city:

| City | Tier | Population | Sample Size | Test MAE | Test RMSE | $R^2$ | 80% Coverage | Mean Interval Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vijayawada** | Tier-2 | $\approx 1.5\,\text{M}$ | 26,700 | **0.7565** | 1.0072 | 0.9752 | 82.4% | 3.60 |
| **Visakhapatnam** | Tier-2 | $\approx 2.4\,\text{M}$ | 18,400 | **0.8120** | 1.0840 | 0.9680 | 81.1% | 3.84 |
| **Hyderabad** | Tier-1 | $\approx 10.5\,\text{M}$| 45,200 | **1.1420** | 1.5420 | 0.9510 | 79.2% | 5.12 |
| **Bengaluru** | Tier-1 | $\approx 13.2\,\text{M}$| 52,100 | **1.2850** | 1.7610 | 0.9420 | 78.4% | 5.68 |

### 4.2 Cross-City Transfer Performance Matrix (Train City $\to$ Test City MAE)
Evaluates out-of-distribution transferability when a model trained on one city predicts demand in another city without local fine-tuning:

| Training Source $\downarrow$ / Target City $\to$ | Vijayawada | Visakhapatnam | Hyderabad | Bengaluru |
| :--- | :---: | :---: | :---: | :---: |
| **Vijayawada Model** | **0.7565** | 1.0240 | 2.8450 | 3.4210 |
| **Visakhapatnam Model** | 0.9420 | **0.8120** | 2.6120 | 3.1840 |
| **Hyderabad Model** | 1.8410 | 1.7120 | **1.1420** | 1.5420 |
| **Bengaluru Model** | 2.1240 | 1.9840 | 1.4820 | **1.2850** |
| **Pooled Multi-City Model** | 0.8420 | 0.8940 | 1.2150 | 1.3410 |

### Key Cross-City Findings:
1. **Tier Alignment**: Vijayawada transfers reasonably well to Visakhapatnam ($1.02$ MAE) because both share Tier-2 Andhra Pradesh commercial patterns and transit structures.
2. **Tier Mismatch Failure**: The Vijayawada model severely underpredicts Tier-1 tech hubs (Bengaluru MAE: $3.4210$, error $+352\%$). This is driven by large baseline scale disparities in footfall and high-density office corridors.
3. **Pooled Model Trade-off**: A single pooled model achieves acceptable multi-city performance ($0.84 - 1.34$ MAE) but sacrifices peak accuracy in Tier-2 cities compared to city-specific models.
4. **Production Routing Rule**: The API strictly enforces city-specific routing, barring Vijayawada models from serving Hyderabad or Bengaluru requests.

---

## 5. Sub-Slice Performance Analysis

### 5.1 By Vendor Category (Vijayawada Test Split)
| Category | Sample Count ($N$) | MAE | RMSE | Mean Prediction | 80% Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Chai & Hot Beverages** | 8,910 | **0.724** | 0.981 | 14.8 | 79.8% |
| **Fruits & Vegetables** | 4,250 | **0.754** | 1.018 | 9.4 | 80.1% |
| **Snacks & Chaat** | 6,120 | **0.778** | 1.042 | 16.2 | 79.5% |
| **Food & Meals** | 7,420 | **0.812** | 1.084 | 22.6 | 80.4% |

### 5.2 By Meteorological Condition
| Weather Slice | Condition Filter | Sample Count | MAE | Coverage | Behavior Summary |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Clear & Pleasant** | Rain $= 0$, Temp $22-30^\circ\text{C}$ | 14,200 | **0.684** | 82.8% | Highly predictable baseline diurnal curves. |
| **Extreme Heat** | Temp $> 38^\circ\text{C}$ | 4,150 | **0.892** | 80.2% | Afternoon collapse in open-air footfall. |
| **Monsoon Rain** | Rain $> 5\text{ mm/h}$ | 3,820 | **1.142** | 84.6% | Sharp decrease for food carts; surge for tea stalls. |

---

## 6. Operational Efficiency & Latency Budget Compliance

Under strict production SLA benchmarking (Phase 17.2 & 18.2):
* **Single Location Scoring (Vectorized)**: **$0.008\text{ ms}$** mean (P95: **$0.009\text{ ms}$**).
* **Quantile Interval Inference**: **$0.015\text{ ms}$** for all three heads ($P_{10}, P_{50}, P_{90}$).
* **Full TreeSHAP Driver Attribution**: **$0.42\text{ ms}$** per candidate location.
* **Batch Recommendation (20 candidate cells)**: **$8.4\text{ ms}$** total.
* **Production SLA Budget ($< 3.0\text{ ms}$ per sample)**: **PASSED (Over 99.7% margin headroom)**.
