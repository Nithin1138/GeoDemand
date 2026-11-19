# 👥 GEODEMAND AI: TEAM ROLES & INDIVIDUAL VIVA DEFENSE GUIDE
## The Official Capstone Examination Playbook for All 4 Members
**Project Title:** GeoDemand AI: An Event-Driven Spatio-Temporal Decision Engine and Friction-Aware VRPTW Scheduling for Mobile Street Retail  
**Institution:** VIT-AP University | School of Computer Science & Engineering (SCOPE)  
**Project Guide:** Dr. Madugula Kiran Kumar  
**Paper Submission:** IICTDS 2026 (Paper ID: #380, Special Track 2)

---

# SECTION A: UNIVERSAL KNOWLEDGE (MUST BE MEMORIZED BY EVERY MEMBER)

*No matter who the panel points at first, ANY member must be able to recite these fundamentals without hesitation.*

### 1. The 30-Second Elevator Pitch
> *"GeoDemand AI is the world's first integrated spatio-temporal decision engine and friction-aware shift scheduling platform designed specifically for informal mobile street vendors (food trucks, tea carts, fruit sellers). Instead of forcing vendors to guess where to go, we discretize urban space into Uber H3 hexagons, forecast quantile customer footfall using a high-speed LightGBM model, calculate true net profit after fuel and ingredient costs, and generate an optimal multi-stop shift route with thermodynamic equipment setup delays using Google OR-Tools. The entire pipeline is verified by 723 passing automated tests and deployed as a real-time Progressive Web App."*

### 2. The Core Mathematical Formula (Net Profit Equation)
$$\mathbf{Net\ Profit} = \mathcal{R} - \mathcal{C}_{\text{ingr}} - \mathcal{C}_{\text{travel}} - \mathcal{C}_{\text{opp}} - \text{Fixed Overhead}$$
- **Revenue ($\mathcal{R}$):** $\min(\hat{y}_{P50}, \text{Capacity}) \times \text{Average Order Value}$
- **Ingredient Cost ($\mathcal{C}_{\text{ingr}}$):** $\mathcal{R} \times (1 - \text{Gross Margin})$
- **Travel Cost ($\mathcal{C}_{\text{travel}}$):** $\text{Distance (km)} \times \text{Fuel Rate per km}$
- **Opportunity Cost ($\mathcal{C}_{\text{opp}}$):** $\text{Travel Time (hours)} \times \text{Hourly Profit}$

### 3. The Key Metrics & Hard Numbers to Quote
- **Spatial Grid:** Uber H3 **Resolution 8** ($0.737\text{ km}^2$ per hexagon, ~461m edge).
- **Target Cities:** Primary = **Vijayawada** (817 cells), Scaled = **Hyderabad** (1,245 cells), **Bengaluru** (1,890 cells), **Visakhapatnam** (932 cells).
- **ML Champion:** **LightGBM v2.1.0** with **0.7641 Test MAE**, **0.9832 RMSE**, **$R^2 = 0.9771$**.
- **Inference Speed:** **0.34 ms per candidate cell** (300x faster than XGBoost).
- **Quantile Coverage:** 80% empirical coverage interval $[P10, P90]$ achieved **80.0% coverage** (2.57 customer interval width).
- **Automated Verification:** **723 / 723 automated tests passing** across CI/CD (`ci.yml` and `ml-ci.yml`).
- **Conference Submission:** Submitted to **IICTDS 2026** on Microsoft CMT under **Paper ID: 380**.





### Section C: Panel Trap Scenarios & Attack Deflections
- Trap 1: "Is this your own code or did you just clone a repository?" 

Proof via Git history, 723 custom unit tests, and domain-specific formulations.
- Trap 2: "711K rows of simulated data is fake; how can you trust it?" 

 100% real spatial/weather substrate, calibrated vendor profiles, and live PostGIS continuous retraining.
- Trap 3: "Isn't this just Google Maps directions for a food cart?" 

Google Maps tells you how to drive; GeoDemand AI solves where, when, how long to stay, and how to maximize profit.

---

# SECTION B: INDIVIDUAL MEMBER DEEP-DIVE & QUESTION BANKS

---

## 🟢 MEMBER 1:  R SAI NAGA ALEKHYA (23BCE20084)
**Primary Domain:** Data Engineering, Spatial Discretization, External APIs, Ingestion Gateway & Market Simulation

### What Pujitha Owns & Built:
1. **Uber H3 Spatial Indexing:** Extracted and mapped the 817 hexagonal cells of Vijayawada at Resolution 8, creating the water boundary exclusion mask for the Krishna River.
2. **External Data Ingestion:**
   - Overpass API queries extracting **1,348 real OpenStreetMap POIs** (offices, schools, colleges, bus stops, malls).
   - Historical weather ingestion from **Open-Meteo archive** (210,240 hourly records spanning 24 years).
   - Official **2011 Census 59-Ward GeoJSON** demographic boundary integration.
3. **The 711K Market Simulation Engine:** Built `src/simulation/market_simulation_v2.py` modeling 150 heterogeneous vendor profiles across 365 days with temporal, holiday, weather, and competition multipliers.
4. **Unified Feature Store:** Joined all static, dynamic, and temporal layers into `feature_store.parquet` (711,750 rows, 45 features, 0 nulls, frozen under SHA-256 manifest `v5c24f1f918f5`).
5. **Real-Time Telemetry Gateway:** Engineered `POST /v1/telemetry/transactions` with strict Pydantic v2 validation and PostGIS relational storage.

### Viva Voce Questions & Answers:

#### Q1: "Why did you choose Uber H3 Resolution 8 instead of standard lat/long coordinates or square grid cells?"
> **Answer:** *"Square grids distort distances because diagonal neighbors are 1.414 times further than orthogonal neighbors. Uber H3 hexagons have identical centroid-to-edge distances in all 6 directions, which eliminates directional bias. We chose Resolution 8 because an H3 Res-8 cell has an area of 0.737 km² and an edge of ~461 meters, which corresponds precisely to the 5-minute walking radius of a street customer looking for food or tea."*

#### Q2: "How did you ensure that there is zero data leakage (future lookahead) in your feature store?"
> **Answer:** *"We instituted a strict temporal causality contract: all dynamic features (such as rolling customer averages, weather metrics, and calendar flags) are measured strictly at or before decision hour $t-1$. When predicting demand for hour $t+1$, the feature store only uses data available prior to the decision point, completely preventing lookahead bias."*

#### Q3: "Explain how you modeled the competitor cannibalization data."
> **Answer:** *"For each cell and category, we used H3's `k_ring(1)` spatial adjacency operator to count neighboring active vendors within immediate adjacent hexagons. We then applied a non-linear saturation curve that reduces expected demand as the local competitor density increases, preventing unrealistic demand spikes in crowded zones."*

---

## 🔵 MEMBER 2: CHITTIMALA SHINNI BABU (23BCE20080)
**Primary Domain:** Machine Learning, Forecasting, Quantile Heads, Model Explainability (SHAP), MLflow & MLOps

### What Alekhya Owns & Built:
1. **Model Exploration & Benchmarking:** Implemented and benchmarked 5 distinct architectures: Linear Regression, Random Forest, CatBoost, XGBoost, and LightGBM on 711,750 samples.
2. **Champion Selection & Training:** Tuned and certified `Approved_LightGBM_v2.1.0` achieving **0.7641 Test MAE** and **0.34 ms inference latency**.
3. **Quantile Uncertainty Heads:** Trained three parallel models minimizing Pinball Loss ($\tau \in \{0.10, 0.50, 0.90\}$) to provide robust lower, median, and upper footfall bounds.
4. **SHAP Explainability Engine:** Integrated TreeSHAP via `shap.TreeExplainer` computing real-time feature attributions ($\phi_i$) for every recommendation.
5. **Multi-Horizon Forecasting:** Engineered forward forecasting sequences for $H \in \{1, 2, 4, 8\}\text{ hours}$ using lag and rolling temporal features.
6. **MLOps & Continuous Retraining:** Configured MLflow experiment tracking, Kolmogorov-Smirnov (KS) data drift detection, and automated retraining pipelines (`continuous-training.yml`).

### Viva Voce Questions & Answers:

#### Q1: "Why did you select LightGBM over deep learning models like LSTMs or Transformers?"
> **Answer:** *"For tabular spatio-temporal datasets with tabular categorical features (like vendor category, menu type, weekday), gradient-boosted decision trees consistently outperform deep neural networks in accuracy while requiring a fraction of the compute. Most importantly, LightGBM executes in 0.34 ms per cell, whereas an LSTM takes 20–40 ms. This allows us to score all 817 candidate city cells in under 50 milliseconds on standard CPU hardware."*

#### Q2: "What is Pinball Loss and why did you use Quantile Regression?"
> **Answer:** *"Standard MSE or MAE only predicts the expected mean footfall, which doesn't reflect real-world risk. Street vendors are vulnerable to downside losses. We used Pinball Loss $\mathcal{L}_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (1 - \tau)(\hat{y} - y))$ to predict the 10th percentile (P10 - pessimistic lower bound), 50th percentile (P50 - median), and 90th percentile (P90 - optimistic upper bound). P10 tells the vendor how much minimum inventory to prep so they don't lose money, while P90 prevents under-stocking during demand surges."*

#### Q3: "How does TreeSHAP work in your recommendation output?"
> **Answer:** *"TreeSHAP calculates the exact Shapley value for all 45 input features based on cooperative game theory. It decomposes the model's prediction into a base citywide expected value of 11.85 customers plus the positive or negative contribution of each feature (e.g., +1.29 from lunch hour, -4.16 from high average order value). This provides human-readable explanations directly to the vendor."*

---

## 🟣 MEMBER 3:KODURU PUJITHA (23BCE20090)

**Primary Domain:** Backend Engineering, Decision Intelligence, OR-Tools VRPTW Shift Routing & Notification Engine

### What Shinni Babu Owns & Built:
1. **FastAPI Application Core:** Architected 47 RESTful endpoints in `src/api/` with asynchronous request handling, CORS, and dependency injection.
2. **Business Economics Engine:** Converted raw customer predictions into financial unit economics: Expected Revenue, Ingredient Costs, Fuel Depreciation, and Relocation Opportunity Loss.
3. **Multi-Criteria Utility Scoring:** Formulated the ranking function balancing profit, uncertainty spread, and competitor cannibalization.
4. **Shift Optimizer (C-DFE-VRPTW):** Implemented the continuous-dwell routing engine using Google OR-Tools Guided Local Search and OSRM distance matrices.
5. **Decentralized Spatial Anti-Herding:** Engineered the Cournot-Nash dynamic penalty mechanism that degrades cell utility as other vendors reserve slots.
6. **Telegram Bot & Webhooks Adapter:** Built the interactive Telegram notification runner (`scripts/run_telegram_bot.py`) providing 1-click location queries and alerts.

### Shinni Babu's Viva Voce Questions & Answers:

#### Q1: "Explain how your Shift Optimizer works and what makes it different from standard delivery routing."
> **Answer:** *"Standard Vehicle Routing Problems (VRP) assume that stops are instantaneous drops (dwell time = 0). For street retail, dwell time is a continuous decision variable. Our C-DFE-VRPTW formulation uses Google OR-Tools Guided Local Search to optimize a vendor's entire 8-hour shift. It balances the profit earned while staying at a location against the travel time, fuel cost, and equipment setup/cooling delays incurred when moving between spots."*

#### Q2: "How do you calculate the financial cost of moving a cart from Cell A to Cell B?"
> **Answer:** *"We calculate two distinct costs: first, the physical travel cost, which is the road-network distance from OSRM multiplied by the vehicle's fuel rate per kilometer (₹8.50/km for fuel carts, ₹3.00/km for electric carts). Second, the opportunity cost, which is the transit time multiplied by the vendor's average historical hourly profit, accounting for the revenue lost while moving rather than selling."*

#### Q3: "How do you prevent the API from crashing if external weather or routing APIs go down?"
> **Answer:** *"We implemented the Resilience Circuit Breaker pattern. If an external service like Open-Meteo or OSRM fails 5 times consecutively, the circuit trips to OPEN state. The system immediately switches to our pre-computed deterministic diurnal weather tables and Haversine road-distance fallbacks without throwing 500 errors or degrading user experience."*

---

## 🔴 MEMBER 4: ISUNURI VEERA NITHIN (23BCE20064)
**Primary Domain:** Technical Lead, System Integration, Leaflet PWA Dashboard, Cloud Infrastructure, CI/CD & Academic Release

### What Nithin Owns & Built:
1. **System Leadership & Integration Contracts:** Designed and enforced the end-to-end data contracts connecting Member 1 $\to$ Member 2 $\to$ Member 3 $\to$ Member 4 with zero schema drift.
2. **Progressive Web App (PWA):** Engineered the Leaflet.js dashboard (`dashboard/app.js`), Service Worker offline caching, responsive bottom navigation bar, and live POS telemetry stream.
3. **Cloud Infrastructure as Code (Terraform):** Created the production AWS Terraform suite (`infra/terraform/`) provisioning VPC, ECS Fargate, RDS PostgreSQL, Redis, and ALB.
4. **Containerization & Reverse Proxy:** Built the production multi-stage `Dockerfile`, `docker-compose.yml`, and NGINX reverse proxy with SSL termination.
5. **CI/CD Quality Gate:** Configured GitHub Actions pipelines (`ci.yml` and `ml-ci.yml`) enforcing **723 automated tests** on every commit with 100% pass rates.
6. **Academic Publication & CMT Management:** Co-authored the research paper and submitted it to **IICTDS 2026** on Microsoft CMT under **Paper ID: 380**.

### Nithin's Viva Voce Questions & Answers:

#### Q1: "What was your architectural strategy to integrate the contributions of all 4 members seamlessly?"
> **Answer:** *"We established strict, decoupled interface contracts using Pydantic v2 schemas and versioned Parquet data stores. Member 1 outputs frozen Parquet datasets verified by SHA-256 manifests. Member 2 produces independent model artifacts that Member 3 loads via a decoupled inference wrapper without importing training pipelines. Member 4 connects the backend to the frontend via OpenAPI 3.1 contracts. Every boundary is protected by automated integration tests in CI/CD."*

#### Q2: "How does your PWA support vendors operating in areas with poor or intermittent internet?"
> **Answer:** *"The dashboard uses a custom Service Worker (`dashboard/sw.js`) that caches all map tiles, CSS, and application scripts locally. In addition, when a vendor queries their shift plan for the day, the complete schedule and polygon coordinates are stored in the browser's IndexedDB / LocalStorage, allowing them to view their route and hotspots even when cellular data drops."*

#### Q3: "Tell us about the paper you submitted to the Scopus-indexed conference."
> **Answer:** *"Our paper is titled 'GeoDemand AI: An Event-Driven Spatio-Temporal Decision Engine and Friction-Aware VRPTW Scheduling for Mobile Street Retail'. It was submitted to the Second International Conference on Computational Techniques in Data Science (IICTDS 2026) under Special Track 2: 'AI and Data-Driven Innovations for Smart and Sustainable Societal Applications' and assigned Microsoft CMT Paper ID #380."*

---

# SECTION C: CROSS-EXAMINATION ATTACK SCENARIOS & PANEL TRAPS

### Trap 1: "Is this really your original code, or did you just copy a GitHub repo?"
**How the Team Responds:**
> *"Every single module in GeoDemand AI was custom-architected for this project. You can inspect our Git commit history—spanning over 50 structured commits across data pipelines, ML benchmarks, OR-Tools formulations, and Terraform infrastructure. Furthermore, our test suite contains 723 custom unit and integration test cases that verify every single domain calculation, from H3 water boundaries to Pinball loss and thermodynamic cooling delays."*

### Trap 2: "711K rows of simulated data is just fake data. How can we trust your results?"
**How Member 1 & Member 2 Respond:**
> *"The spatial and environmental substrate is 100% real: 1,348 real OpenStreetMap POIs and 24 years of real Open-Meteo weather records for Vijayawada. The transaction outcomes were simulated using real-world economic parameters from 150 categorized street vendors. Furthermore, Member 1 built a live telemetry ingestion gateway (`POST /v1/telemetry/transactions`), and Member 2 implemented a Kolmogorov-Smirnov drift detection and continuous retraining pipeline so that as live UPI and cash transactions flow in, the model continuously adapts to ground truth."*

### Trap 3: "Isn't a food cart routing problem just standard Google Maps directions?"
**How Member 3 & Member 4 Respond:**
> *"Google Maps only tells you how to drive between point A and point B after you already know where you want to go. It has zero awareness of customer demand, food preparation times, inventory capacities, or competitor crowding. GeoDemand AI solves the inverse, much harder problem: determining WHERE the vendor should go, WHEN they should arrive, HOW LONG they should stay, and WHAT route maximizes their net profit."*






