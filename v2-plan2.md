# **Yes. If every functionality and task in the complete project roadmap is implemented, tested, integrated, and accepted, GeoDemand AI is complete against that defined project scope.**

There is one important distinction: completing the code is not the same as completing the project. Every feature must also pass its acceptance tests, work with the other modules, and be documented. Real-world commercial performance must still be validated using actual vendor data and real deployments.

I have used the complete GeoDemand AI V1 specification, the V2.0 roadmap, and the existing four-member ownership structure to prepare this plan.&#x20;

Pasted markdown(20260808-191950).md

Pasted text.txt



The plan is organized member by member, as requested. Each member gets their own phases, tasks, functions, deliverables and handover responsibilities. The phase references preserve the previous numbering:

- Phases 1–4: original member ownership.
- Phases 8–20: extended V1 stabilization and V2 roadmap.
- Each member works through their assigned phases, then hands over to the next dependent owner.

## **Project-wide ownership**

| Member           | Complete ownership                                                                                             |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| Member 1         | Data engineering, external data, H3, feature store, simulation, telemetry data and geospatial enrichment       |
| Member 2         | Machine learning, forecasting, SHAP, quantile regression, MLflow, MLOps and continuous training                |
| Member 3         | FastAPI, business economics, recommendation engine, routing, optimization, fleet management and integrations   |
| Member 4: Nithin | Technical leadership, frontend, integration, security, CI/CD, infrastructure, QA, deployment and final release |

**Important:** Each member must finish their own module tests before handing over. Nithin coordinates integration and verifies the interfaces. He should not be expected to rewrite the other three members' modules.

# **MEMBER 1: DATA ENGINEERING, EXTERNAL DATA AND SIMULATION**

Primary owner: Member 1

**Main objective:** Build and maintain the complete data foundation that supplies reliable, spatially indexed, time-aware data to the ML model, backend, optimization engine and dashboard.

**Main repository ownership:**

```
data/
src/data/
src/features/
src/simulation/
src/validation/
tests/test_data_pipeline.py
tests/test_feature_store.py
tests/test_h3.py
tests/test_simulation.py
```

## **Phase 1: Data Engineering Foundation and Repository Audit**

### **Phase 1.1: Existing Dataset Audit**

Tasks:

1. Inspect every file under `data/raw/`.
2. Inspect every file under `data/processed/`.
3. Inspect every file under `data/simulated/`.
4. Identify the canonical feature store.
5. Identify legacy datasets that must not be used for training.
6. Inspect all Parquet files and their schemas.
7. Verify row counts and column counts.
8. Verify data types of all columns.
9. Verify null and missing-value counts.
10. Identify duplicate records.
11. Identify invalid timestamps.
12. Identify invalid coordinates.
13. Verify H3 cell IDs.
14. Verify vendor IDs and category values.
15. Verify data versioning and pipeline manifests.
16. Check that training and inference datasets use compatible schemas.
17. Create an inventory of every dataset and its purpose.
18. Update `docs/DATASETS_CATALOG.md`.

Required output:

- Dataset inventory.
- Dataset schema documentation.
- Data-quality report.
- Legacy dataset classification.
- Updated dataset catalog.

### **Phase 1.2: External Data Collection**

Tasks:

1. Implement or verify OpenStreetMap data collection.
2. Implement or verify Overpass API integration.
3. Collect office locations.
4. Collect schools and colleges.
5. Collect hospitals.
6. Collect restaurants and food outlets.
7. Collect malls and shopping areas.
8. Collect parks and public spaces.
9. Collect bus stops and railway stations.
10. Collect roads and building footprints where available.
11. Implement or verify Open-Meteo current weather collection.
12. Implement historical weather collection.
13. Implement hourly weather forecasts.
14. Collect temperature.
15. Collect humidity.
16. Collect rainfall.
17. Collect wind speed.
18. Collect pressure.
19. Collect cloud cover.
20. Collect visibility and UV index where available.
21. Integrate public holiday calendars.
22. Integrate regional holidays.
23. Add public event information where available.
24. Collect competition information where supported.
25. Record the source of every collected feature.
26. Record collection timestamps.
27. Implement request timeouts.
28. Implement retry handling.
29. Implement API rate-limit handling.
30. Implement local caching.
31. Implement stale-cache handling.
32. Implement offline fallback data.
33. Implement provider failure logging.
34. Verify the target geographic coverage around Vijayawada.

Required output:

- Working weather client.
- Working OSM/Overpass client.
- Holiday and calendar pipeline.
- Event and competition data pipeline where available.
- Provider cache and fallback logic.
- External data test report.

### **Phase 1.3: H3 Spatial Processing**

Tasks:

1. Implement coordinate-to-H3 conversion.
2. Implement H3-to-coordinate conversion.
3. Enforce H3 resolution 8 for the existing V1 geographic grid.
4. Generate H3 cell boundaries.
5. Generate neighboring H3 cells.
6. Calculate cell centroids.
7. Validate cell IDs.
8. Validate coordinates.
9. Detect cells outside the target region.
10. Aggregate POI counts by H3 cell.
11. Aggregate road and building features.
12. Generate static spatial features.
13. Identify water and unsuitable cells.
14. Validate candidate location eligibility inputs.
15. Remove duplicate POIs.
16. Handle invalid geographic records.
17. Generate spatial coverage reports.
18. Test the spatial calculations across different parts of Vijayawada.

Required output:

- H3 cell dataset.
- H3 spatial utilities.
- POI aggregation pipeline.
- Static feature dataset.
- Spatial validation report.

### **Phase 1.4: Market Simulation Engine**

Tasks:

1. Audit the existing market simulator.
2. Verify simulated vendor profiles.
3. Verify the four existing vendor categories.
4. Verify category-specific demand parameters.
5. Verify product and inventory profiles.
6. Verify base customer demand.
7. Verify POI affinity effects.
8. Verify hour-of-day demand curves.
9. Verify weekday and weekend effects.
10. Verify holiday effects.
11. Verify weather effects.
12. Verify rainfall and office-density interactions.
13. Verify weekend and mall-density interactions.
14. Verify nonlinear saturation.
15. Verify heteroscedastic noise.
16. Verify hidden vendor skill variables.
17. Verify hidden cell reputation variables.
18. Confirm hidden variables are excluded from trainable features.
19. Confirm that the model cannot simply reverse-engineer a public deterministic target formula.
20. Confirm next-hour target generation.
21. Verify the decision timestamp and target timestamp contract.
22. Verify deterministic random seeds.
23. Generate versioned simulated datasets.
24. Generate metadata sidecars.
25. Record simulation parameters and version hashes.
26. Document the limitations of simulated business outcomes.

Required output:

- Final market simulation engine.
- Reproducible simulated dataset.
- Metadata sidecars.
- Simulation methodology.
- Leakage validation report.

### **Phase 1.5: Feature Store Development**

Tasks:

1. Implement static feature generation.
2. Implement dynamic feature generation.
3. Join H3 spatial features.
4. Join weather features.
5. Join calendar features.
6. Join holiday features.
7. Join event features.
8. Join competition features.
9. Join vendor business characteristics.
10. Generate `decision_timestamp`.
11. Generate `target_timestamp`.
12. Generate `as_of_timestamp` for time-sensitive features.
13. Generate `feature_version`.
14. Generate data provenance fields.
15. Validate feature types.
16. Validate feature ranges.
17. Handle missing values.
18. Handle duplicate feature rows.
19. Validate temporal alignment.
20. Ensure future information cannot enter training features.
21. Exclude IDs and grouping fields from model inputs where required.
22. Ensure the target is `expected_customer_count`.
23. Align feature names and order with Member 2's model schema.
24. Generate the canonical feature store.
25. Generate a pipeline manifest.

Required output:

- Canonical `feature_store.parquet`.
- Feature schema JSON.
- Versioned feature store.
- Pipeline manifest.
- Feature-quality report.

### **Phase 1.6: Data Testing and Handover**

Tasks:

1. Test coordinate-to-H3 conversion.
2. Test H3 neighbor generation.
3. Test POI aggregation.
4. Test weather collection.
5. Test weather offline fallback.
6. Test weather caching.
7. Test hourly forecast response contracts.
8. Test holiday features.
9. Test simulation reproducibility.
10. Test hidden-variable exclusion.
11. Test target timestamp correctness.
12. Test temporal leakage rejection.
13. Test feature schema.
14. Test missing values.
15. Test duplicate detection.
16. Test data provenance.
17. Test pipeline execution from a clean environment.
18. Document the commands required to regenerate datasets.

**Handover to Member 2:**

- Canonical feature store.
- Feature schema.
- Dataset version and metadata.
- Data-quality report.
- Simulation methodology.
- Data generation commands.
- Passing automated data tests.

**Handover condition:** Member 2 must be able to load the dataset and reproduce the expected feature schema without manually modifying it.

## **Phase 8.1: V1 Data Stabilization**

Tasks:

1. Verify all V1 datasets against the latest manifests.
2. Reconcile actual dataset row counts.
3. Identify and resolve outdated feature-store versions.
4. Confirm the canonical training dataset.
5. Verify weather fallback caching.
6. Implement the missing hourly fallback contract.
7. Verify the `forecast_for_timestamp` field.
8. Verify `forecast_generation_timestamp`.
9. Verify `hourly_forecast_offset`.
10. Remove unused placeholder data folders after checking dependencies.
11. Validate all pipeline commands.
12. Ensure a clean installation can regenerate the data.

Deliverables: Stable V1 data pipeline and passing data-related tests.

## **Phase 11.1: Real Transaction and Ground-Truth Data Pipeline**

Tasks:

1. Create `src/ingestion/` if it does not exist.
2. Define a standard transaction schema.
3. Define vendor ID.
4. Define transaction ID.
5. Define event timestamp.
6. Define amount in INR.
7. Define payment method.
8. Define product or item details.
9. Define location information.
10. Validate incoming transaction records.
11. Detect duplicate transaction IDs.
12. Handle delayed transaction events.
13. Handle corrected and cancelled transactions.
14. Handle missing location information.
15. Map vendor location to H3 resolution 8.
16. Map transactions into hourly time buckets.
17. Separate transaction count from unique customer count.
18. Separate item count from customer count.
19. Implement `src/data/actuals_aggregator.py`.
20. Generate `actual_customer_count` only when it can be measured.
21. Generate hourly sales aggregation.
22. Track source provenance.
23. Track ingestion timestamps.
24. Build a real-data quality report.
25. Keep actual data separate from simulated targets.
26. Define data consent, access and retention rules.

Deliverables:

- Transaction ingestion schema.
- Actuals aggregation module.
- Real-data feature dataset.
- Data provenance report.
- Real-data quality tests.

## **Phase 12.1: Census and Population Enrichment**

Tasks:

1. Identify Vijayawada Census 2011 ward boundaries.
2. Acquire official ward polygon data.
3. Validate coordinate reference systems.
4. Clean invalid polygons.
5. Intersect ward polygons with H3 cells.
6. Assign ward population information to H3 cells.
7. Integrate Meta HRSL where available.
8. Integrate Kontur population data where available.
9. Validate source licenses and usage terms.
10. Compare population estimates from different sources.
11. Document source years and spatial resolution.
12. Handle missing demographic data.
13. Add population features to the feature store.
14. Update feature schema and manifest.
15. Test demographic enrichment.

Deliverables:

- Census ward dataset.
- H3 population dataset.
- HRSL/Kontur integration where available.
- Population comparison report.
- Updated feature store.

## **Phase 12.2: Competitor and Operating-Hours Data**

Tasks:

1. Build a daily competitor refresh job.
2. Improve OSM competitor collection.
3. Integrate Google Places where access permits.
4. Collect competitor categories.
5. Aggregate competitors by H3 cell.
6. Collect operating hours where available.
7. Distinguish daytime and nighttime competitors.
8. Identify stale competitor records.
9. Implement daily cache refresh.
10. Handle rate limits and provider outages.
11. Generate category-specific competitor features.
12. Validate competitor counts against available source records.

Deliverables:

- Competitor refresh pipeline.
- Competitor operating-hours dataset.
- Time-aware competition features.

## **Phase 14.1: Multi-City Data Engineering**

Tasks:

1. Remove Vijayawada-only assumptions from reusable pipeline code.
2. Add configurable city boundaries.
3. Support arbitrary bounding boxes.
4. Generate H3 grids for different cities.
5. Implement city-specific POI collection.
6. Implement city-specific weather collection.
7. Implement city-specific calendar and holiday features.
8. Implement city-specific competitor features.
9. Add Hyderabad.
10. Add Cyberabad and HITEC City coverage.
11. Add Bengaluru.
12. Add Koramangala, Indiranagar and Electronic City coverage.
13. Add Visakhapatnam.
14. Add Beach Road and MVP Colony coverage.
15. Generate separate feature stores by city.
16. Validate city IDs and geographic boundaries.
17. Document the procedure for adding a new city.

Deliverables:

- Reusable multi-city data pipeline.
- Three additional city datasets.
- City onboarding guide.

## **Phase 15.1: Data Persistence and Migration**

Tasks:

1. Classify operational and analytical datasets.
2. Keep suitable analytical datasets in Parquet.
3. Design schemas for vendors and transactions.
4. Design schemas for reservations.
5. Design schemas for location events.
6. Prepare PostgreSQL migration scripts.
7. Validate PostGIS spatial fields.
8. Migrate applicable transactional data.
9. Create spatial and timestamp indexes with appropriate query plans.
10. Verify migrated record counts.
11. Verify spatial query correctness.
12. Test backup and restore.

Deliverables:

- Database schemas.
- Migration scripts.
- Migration verification report.
- Backup and restore instructions.

## **Phase 17.1, 18.1, 19.1 and 20.1: Final Data Responsibilities**

**Phase 17.1: Data monitoring**

- Monitor ingestion failures.
- Monitor data freshness.
- Monitor null and duplicate rates.
- Monitor provider availability.
- Monitor pipeline execution time.
- Configure data-quality alerts.
- Maintain data lineage and provenance.

**Phase 18.1: Data integration testing**

- Test all data sources together.
- Test actuals ingestion.
- Test population enrichment.
- Test multi-city feature generation.
- Test database migration.
- Test pipeline restart and recovery.
- Verify data passed to ML and backend.

**Phase 19.1: Data release**

- Freeze production dataset versions.
- Verify production manifests.
- Validate data sources.
- Verify scheduled ingestion.
- Test production backup and restoration.

**Phase 20.1: Data documentation**

- Document every source.
- Document every feature.
- Document data transformations.
- Document real and simulated data.
- Document temporal validation.
- Document city onboarding.
- Document data recovery.

**Final Member 1 completion:** All V1 and V2 data pipelines, simulation, telemetry, spatial enrichment, multi-city support, tests, migration and documentation are delivered.

# **MEMBER 2: MACHINE LEARNING AND MLOPS**

Primary owner: Member 2

**Main objective:** Deliver a reproducible forecasting system that predicts expected customer counts, quantifies uncertainty, explains predictions and supports controlled model updates.

**Main repository ownership:**

```
src/geodemand/models/
src/geodemand/monitoring/
src/models/
models/
scripts/train_model.py
scripts/run_ml_workflow.py
scripts/promote_model.py
tests/unit/
tests/test_ml_model.py
tests/test_lightgbm_integration.py
tests/test_ml_monitoring.py
```

## **Phase 2: Machine Learning Foundation**

### **Phase 2.1: Existing Model Audit**

Tasks:

1. Inspect all model artifacts.
2. Inspect `models/demand_model.pkl`.
3. Inspect LightGBM artifacts.
4. Inspect XGBoost artifacts.
5. Inspect candidate model bundles.
6. Review model comparison reports.
7. Check metadata and feature schemas.
8. Check training dataset versions.
9. Check the model currently loaded by the API.
10. Reproduce the XGBoost versus LightGBM comparison.
11. Reconcile conflicting MAE, RMSE and R² metrics.
12. Verify whether different metrics use different targets, splits or dataset versions.
13. Identify obsolete artifacts.
14. Approve one production model.
15. Document the model selection decision.

Deliverables:

- Model inventory.
- Reproduced comparison report.
- Approved production model decision.
- Feature compatibility report.

### **Phase 2.2: Training and Evaluation**

Tasks:

1. Load Member 1's canonical feature store.
2. Verify the customer-count target.
3. Separate identity fields from trainable features.
4. Build chronological train, validation and test splits.
5. Implement forward-chaining validation.
6. Implement rolling-window evaluation where appropriate.
7. Prevent temporal leakage.
8. Train Linear Regression.
9. Train Random Forest.
10. Train XGBoost.
11. Train LightGBM.
12. Train CatBoost.
13. Implement baseline comparisons.
14. Tune hyperparameters with Optuna.
15. Use Bayesian hyperparameter search where appropriate.
16. Evaluate MAE.
17. Evaluate RMSE.
18. Evaluate R².
19. Evaluate performance by category.
20. Evaluate performance by H3 cell or geographic segment.
21. Record model parameters.
22. Record dataset version.
23. Record feature schema version.
24. Save reproducible model artifacts.
25. Document limitations of simulated-data evaluation.

Deliverables:

- Reproducible training pipeline.
- Model comparison table.
- Optuna experiment records.
- Evaluation report.
- Approved model artifact.

### **Phase 2.3: Quantile Prediction and Uncertainty**

Tasks:

1. Implement P10 predictions.
2. Implement P50 predictions.
3. Implement P90 predictions.
4. Ensure outputs are ordered.
5. Prevent negative customer-count predictions.
6. Handle quantile crossing.
7. Measure empirical coverage.
8. Measure prediction interval width.
9. Evaluate coverage by category.
10. Evaluate coverage by forecast horizon.
11. Validate uncertainty outputs during inference.
12. Document the difference between prediction intervals and confidence intervals.

Deliverables:

- Point prediction interface.
- P10/P50/P90 interface.
- Quantile evaluation report.
- Prediction schema.

### **Phase 2.4: Explainable AI**

Tasks:

1. Integrate SHAP.
2. Implement global feature importance.
3. Implement local explanations.
4. Generate per-prediction SHAP values.
5. Identify positive demand contributors.
6. Identify negative demand contributors.
7. Support category-specific explanations.
8. Generate explanations for candidate locations.
9. Verify explanations against model outputs.
10. Create explanation test cases.
11. Document SHAP limitations.

Deliverables:

- Global SHAP report.
- Local explanation interface.
- Feature importance output.
- Explanation test suite.

### **Phase 2.5: MLflow and Model Registry**

Tasks:

1. Configure MLflow tracking.
2. Log training parameters.
3. Log model metrics.
4. Log dataset versions.
5. Log feature schema versions.
6. Store SHAP artifacts.
7. Register candidate models.
8. Implement Staging and Production stages or an equivalent versioned promotion process.
9. Define promotion gates.
10. Implement model approval.
11. Implement model rejection.
12. Implement model rollback.
13. Verify model loading after application restart.
14. Track model versions used for predictions.

Deliverables:

- MLflow experiment tracking.
- Model registry.
- Promotion policy.
- Rollback mechanism.

### **Phase 2.6: Production Model Handover**

Tasks:

1. Test independent model loading.
2. Test prediction for every vendor category.
3. Test feature schema compatibility.
4. Test reproducibility.
5. Test nonnegative outputs.
6. Test quantile ordering.
7. Test SHAP outputs.
8. Test model artifact compatibility.
9. Test restart and reload.
10. Test approved-model selection.
11. Document inference commands.
12. Provide saved reference predictions.

Handover to Member 3:

- Approved production model.
- Point prediction interface.
- Quantile prediction interface.
- Feature schema.
- Model metadata.
- SHAP explanation interface.
- Model evaluation report.
- MLflow configuration.

**Handover condition:** Member 3 can load the model through a documented interface without knowing the internal training implementation.

## **Phase 13.1: Multi-Horizon and Advanced Forecasting Data**

Tasks:

1. Coordinate with Member 1 on multi-hour feature availability.
2. Create chronological multi-horizon datasets.
3. Create lag features.
4. Create rolling-window features.
5. Create historical demand features where valid.
6. Prepare sliding-window sequence inputs.
7. Verify time alignment.
8. Verify no future features enter training.
9. Create reproducible multi-horizon splits.

Deliverables: Multi-horizon training dataset and feature specification.

## **Phase 13.2: Advanced ML Model Benchmark**

Tasks:

1. Establish the approved LightGBM baseline.
2. Implement recursive autoregressive LightGBM.
3. Implement TFT experiment.
4. Implement N-HiTS experiment.
5. Implement PatchTST experiment.
6. Compare all models on identical splits.
7. Evaluate one-hour predictions.
8. Evaluate multiple future-hour predictions.
9. Evaluate MAE.
10. Evaluate RMSE.
11. Evaluate R².
12. Evaluate quantile coverage.
13. Measure inference latency.
14. Measure memory consumption.
15. Compare accuracy and operational cost.
16. Select the champion using documented criteria.
17. Preserve the previous production model for rollback.

Deliverables:

- Advanced model implementations.
- Multi-horizon comparison report.
- Champion selection report.

## **Phase 13.3: Continuous Training Pipeline**

Tasks:

1. Implement `src/geodemand/monitoring/ct_pipeline.py`.
2. Define training-data eligibility rules.
3. Verify sufficient real data before retraining.
4. Build scheduled retraining.
5. Build drift-triggered retraining.
6. Integrate GitHub Actions or Airflow.
7. Train candidate model.
8. Evaluate candidate model.
9. Compare candidate against champion.
10. Validate feature compatibility.
11. Validate quantile quality.
12. Apply promotion gates.
13. Require approval before production promotion where configured.
14. Deploy candidate safely.
15. Support zero-downtime model replacement where architecture allows.
16. Implement rollback.
17. Log every retraining event.

Deliverables:

- Automated retraining workflow.
- Candidate evaluation.
- Promotion gates.
- Model rollback.

## **Phase 13.4: Drift and Model Monitoring**

Tasks:

1. Integrate Evidently AI where appropriate.
2. Calculate PSI.
3. Calculate Wasserstein Distance.
4. Monitor input feature drift.
5. Monitor prediction drift.
6. Monitor seven-day MAE.
7. Implement the roadmap's drift trigger of greater than 0.15.
8. Implement the seven-day MAE degradation trigger of more than 20%.
9. Test false-positive drift scenarios.
10. Test missing actual-label scenarios.
11. Avoid retraining without sufficient ground truth.
12. Trigger retraining workflow when conditions are satisfied.
13. Test candidate promotion.
14. Test rollback.
15. Expose model health information to the API.

Deliverables:

- Drift metrics.
- Monitoring configuration.
- Retraining trigger tests.
- Model health report.

## **Phase 14.2: Multi-City Model Generalization**

Tasks:

1. Evaluate the Vijayawada model on Hyderabad data.
2. Evaluate the model on Bengaluru data.
3. Evaluate the model on Visakhapatnam data.
4. Compare city-specific models.
5. Compare pooled multi-city models.
6. Perform train-one-city/test-another-city experiments.
7. Evaluate category-wise performance.
8. Identify non-generalizable features.
9. Evaluate city-specific calibration.
10. Define model selection by city.
11. Prevent incorrect model-to-city mapping.
12. Document generalization limitations.

Deliverables: Cross-city model evaluation and city model strategy.

## **Phases 17.2, 18.2, 19.2 and 20.2: Final ML Responsibilities**

**Phase 17.2: ML monitoring**

- Monitor feature drift.
- Monitor prediction drift.
- Monitor real-label MAE.
- Monitor quantile coverage.
- Monitor model loading failures.
- Monitor inference latency.
- Track model version and rollback history.

**Phase 18.2: ML integration testing**

- Test production model loading.
- Test model schema across cities.
- Test multi-hour predictions.
- Test quantile ordering.
- Test SHAP.
- Test drift.
- Test retraining.
- Test promotion and rollback.
- Test actual-data evaluation.

**Phase 19.2: Model release**

- Freeze model version.
- Freeze feature schema.
- Package dependencies.
- Run production smoke predictions.
- Verify registry metadata.
- Verify rollback.

**Phase 20.2: ML documentation**

- Document all model architectures.
- Document training and evaluation.
- Document splits and metrics.
- Document quantile methodology.
- Document SHAP methodology.
- Document drift and retraining.
- Document real versus simulated evaluation.
- Document limitations.

**Final Member 2 completion:** The approved model, advanced benchmarks, quantile outputs, SHAP, MLflow, continuous training, drift monitoring, multi-city evaluation and model tests are delivered.

# **MEMBER 3: BACKEND, BUSINESS ECONOMICS AND DECISION INTELLIGENCE**

Primary owner: Member 3

**Main objective:** Convert vendor location and contextual data into reliable demand predictions, profit calculations, location recommendations, optimized routes and fleet decisions.

**Main repository ownership:**

```
src/api/
src/api/providers/
src/optimization/
src/ingestion/
src/geodemand/
tests/test_api.py
tests/test_business_economics.py
tests/test_candidate_location.py
tests/test_external_clients.py
```

## **Phase 3: V1 Backend and Decision Intelligence**

### **Phase 3.1: API and Architecture Audit**

Tasks:

1. Inspect `src/api/app.py`.
2. Inspect all API route modules.
3. Document request schemas.
4. Document response schemas.
5. Validate Pydantic models.
6. Verify API versioning.
7. Verify API-key authentication.
8. Verify authorization requirements.
9. Verify environment variable usage.
10. Identify duplicated business logic.
11. Identify unstable response contracts.
12. Integrate Member 2's approved model interface.
13. Define the final API contract.
14. Update OpenAPI documentation.

Deliverables:

- API inventory.
- API architecture.
- Request/response schemas.
- API audit report.

### **Phase 3.2: Live Feature Assembly and Prediction**

Tasks:

1. Integrate weather provider.
2. Integrate OSM/POI provider.
3. Integrate holiday provider.
4. Integrate event features.
5. Integrate competition features.
6. Accept vendor location.
7. Accept vendor category.
8. Accept vendor inventory.
9. Assemble model features.
10. Match feature names and ordering.
11. Validate required feature fields.
12. Handle missing contextual features.
13. Handle unavailable external providers.
14. Generate next-hour customer-count predictions.
15. Generate P10/P50/P90 estimates.
16. Include prediction timestamp.
17. Include model version.
18. Include feature freshness metadata.
19. Validate prediction response.

Deliverables:

- Live feature assembler.
- Inference integration.
- Prediction API.
- Feature compatibility tests.

### **Phase 3.3: Candidate Location Generation**

Tasks:

1. Generate nearby H3 cells.
2. Apply configurable search radius.
3. Exclude current location where relocation is being evaluated.
4. Exclude water and unsuitable cells.
5. Remove invalid coordinates.
6. Remove duplicate H3 cells.
7. Resolve candidate POIs.
8. Validate candidate accessibility.
9. Support different starting locations.
10. Return eligible candidate details.
11. Validate candidate generation performance.

Deliverables:

- Candidate generation module.
- Spatial filter.
- Candidate validation tests.

### **Phase 3.4: Business Economics Engine**

Tasks:

1. Define product categories.
2. Define average selling prices.
3. Define average order values.
4. Define product mix.
5. Define inventory capacity.
6. Define cost of goods sold.
7. Calculate expected revenue.
8. Calculate expected stock cost.
9. Calculate operating expenses.
10. Calculate fuel cost.
11. Calculate travel cost.
12. Calculate time cost.
13. Calculate expected profit.
14. Handle inventory limits.
15. Handle category-specific costs.
16. Document assumptions.
17. Test using manual examples.

Deliverables:

- Business economics module.
- Product pricing configuration.
- Cost and profit calculations.
- Economics test report.

### **Phase 3.5: Ranking and Decision Engine**

Tasks:

1. Implement configurable candidate ranking.
2. Incorporate demand predictions.
3. Incorporate expected profit.
4. Incorporate travel cost.
5. Incorporate competition.
6. Incorporate vendor preferences.
7. Incorporate prediction uncertainty.
8. Calculate current-location profit.
9. Calculate candidate-location profit.
10. Calculate travel-adjusted profit uplift.
11. Apply distance limits.
12. Apply travel-time limits.
13. Apply fuel cost.
14. Implement STAY.
15. Implement CONSIDER MOVE.
16. Implement RECOMMENDED MOVE.
17. Validate the thresholds.
18. Generate `decision_reason`.
19. Generate ranked Top-5 results.
20. Ensure high ranking alone cannot force a move recommendation.

Deliverables:

- Ranking engine.
- Decision engine.
- Economics integration.
- Recommendation explanations.
- Decision test report.

### **Phase 3.6: Routing and Provider Reliability**

Tasks:

1. Integrate OSRM routing.
2. Calculate road distance.
3. Calculate travel time.
4. Validate route responses.
5. Handle route timeouts.
6. Handle routing provider failures.
7. Implement routing fallback.
8. Validate traffic proxy integration.
9. Validate fuel-price proxy integration.
10. Ensure missing routing data cannot generate false relocation decisions.
11. Cache repeated routing queries.

Deliverables:

- Routing interface.
- Routing cache.
- Provider fallback logic.
- Routing test report.

### **Phase 3.7: V1 API Completion**

Verify and test:

- `POST /v1/location/context`
- `POST /v1/recommendations/live`
- `GET /v1/system/data-health`
- `GET /v1/system/ping`
- Scenario simulation endpoint.
- Any additional endpoints already implemented in the repository.

Also implement or verify the original project API capabilities:

1. Demand prediction endpoint.
2. Top-5 recommendation endpoint.
3. Scenario simulation endpoint.
4. API-key authentication.
5. Response caching.
6. Batch scoring support.
7. Versioned API contracts.

Scenario simulation must support:

- Rainfall changes.
- Competitor changes.
- Fuel price changes.
- Product or vendor category changes.
- Different starting hours.
- Recalculation without retraining.

Deliverables:

- Stable FastAPI application.
- Documented endpoint schemas.
- Scenario simulation.
- API test report.
- Error-handling tests.

## **Phase 9.3: Multi-Stop Shift Optimization**

Tasks:

1. Create `src/optimization/`.
2. Implement `shift_optimizer.py`.
3. Integrate Google OR-Tools.
4. Formulate the Vehicle Routing Problem with Time Windows.
5. Accept shift start time.
6. Accept shift duration.
7. Support 8–12-hour shifts.
8. Accept maximum relocation stops.
9. Accept minimum dwell time.
10. Include setup time.
11. Include teardown time.
12. Include parking and relocation overhead.
13. Include travel time.
14. Include travel distance.
15. Include fuel cost.
16. Include expected revenue at each candidate.
17. Include COGS and fixed hourly costs.
18. Maximize expected shift net profit.
19. Enforce time windows.
20. Enforce shift-end constraints.
21. Enforce relocation limits.
22. Enforce dwell-time limits.
23. Handle infeasible solutions.
24. Handle solver time limits.
25. Return ordered H3 locations.
26. Return arrival and departure times.
27. Return dwell durations.
28. Return expected profit per stop.
29. Return total shift profit.
30. Compare the itinerary against staying at the current location.
31. Implement `time_windows.py`.
32. Implement `POST /v1/routes/shift-plan`.

Deliverables:

- OR-Tools shift optimizer.
- Dynamic time-window engine.
- Shift planning API.
- Optimization test suite.

## **Phase 10.3: Fleet Coordination and Collision Avoidance**

Tasks:

1. Implement `fleet_manager.py`.
2. Define vendor identity.
3. Define vendor category.
4. Define current vendor location.
5. Define vendor availability.
6. Define cell capacity.
7. Define category-specific competition.
8. Calculate demand-sharing penalties.
9. Implement the roadmap's effective-demand calculation:

\\[ D\_{\text{effective}}(H)= \frac{D\_{\text{base}}(H)} {1+\lambda N\_{\text{same category}}(H)} \\]

10. Track active reservations.
11. Track pending reservations.
12. Implement soft H3 reservations.
13. Implement reservation expiry.
14. Implement reservation cancellation.
15. Prevent conflicting reservations.
16. Use atomic database operations or suitable locking.
17. Prevent duplicate dispatch.
18. Calculate fleet-wide expected profit.
19. Implement multi-vendor dispatch.
20. Allocate vendors to candidate zones.
21. Minimize unnecessary vendor clustering.
22. Support independent vendors and organized fleets.
23. Implement `POST /v1/fleet/reserve-cell`.
24. Implement `POST /v1/fleet/dispatch`.
25. Implement `GET /v1/fleet/cell-occupancy`.
26. Return reservation status.
27. Return vendor assignments.
28. Return fleet dispatch explanations.

Deliverables:

- Fleet manager.
- Competition-sharing logic.
- Reservation engine.
- Fleet dispatch engine.
- Fleet APIs.
- Concurrency tests.

## **Phase 11.3: Transaction and Messaging Integrations**

Tasks:

1. Create payment webhook endpoints.
2. Integrate Razorpay events where supported.
3. Integrate Paytm events where supported.
4. Integrate PhonePe events where supported.
5. Validate webhook signatures.
6. Validate event timestamps.
7. Validate vendor identity.
8. Validate transaction IDs.
9. Implement idempotency.
10. Handle duplicate webhook delivery.
11. Handle retry behavior.
12. Handle failed-event recovery.
13. Implement a dead-letter process for unrecoverable events.
14. Integrate WhatsApp Business API where permitted.
15. Integrate Telegram bot.
16. Accept vendor sales reports.
17. Accept vendor location pins.
18. Validate user and vendor identity.
19. Forward validated data to Member 1's ingestion pipeline.
20. Return ingestion status.
21. Record errors without exposing payment credentials or sensitive data.

Deliverables:

- Payment webhook adapters.
- WhatsApp reporting.
- Telegram reporting.
- Secure event processing.
- Integration tests.

**Architecture note:** The V2 architecture includes a stream/event ingestion gateway and Kafka or RabbitMQ. Select a broker after measuring expected event volume and delivery requirements. If used, implement event publication, consumers, retries, dead-letter handling and idempotent processing. Do not add a broker merely to claim that it is present.

## **Phase 12.3: Live Competitor Provider**

Tasks:

1. Implement a common competitor provider interface.
2. Integrate Google Places where permitted.
3. Integrate OSM Overpass.
4. Retrieve competitor category.
5. Retrieve competitor location.
6. Retrieve operating hours where available.
7. Resolve competitor coordinates to H3.
8. Aggregate competitors by H3 and category.
9. Implement scheduled refresh.
10. Implement caching.
11. Handle provider rate limits.
12. Handle missing operating hours.
13. Return freshness metadata.
14. Integrate competitor features into live recommendation assembly.

Deliverables:

- Competitor provider interface.
- Scheduled competitor refresh.
- API integration.
- Provider tests.

## **Phase 16.3: Notification and Vendor Communication Backend**

Tasks:

1. Implement a notification trigger service.
2. Implement impending-rain alerts.
3. Implement event-based demand alerts.
4. Implement demand-spike alerts.
5. Calculate expected profit uplift before recommending relocation.
6. Include the destination H3 cell.
7. Include the recommended departure time.
8. Include expected arrival time.
9. Include forecast freshness.
10. Integrate Firebase Cloud Messaging.
11. Integrate WhatsApp Business API where permitted.
12. Implement notification preferences.
13. Implement notification opt-out.
14. Implement duplicate-notification prevention.
15. Implement notification expiry.
16. Track delivery status.
17. Handle failed delivery.
18. Ensure notifications do not recommend a move using stale or unreliable forecasts.

Deliverables:

- Notification engine.
- FCM integration.
- WhatsApp integration.
- Notification API and event contracts.
- Delivery tests.

## **Phase 17.3: Backend Reliability and Security**

Tasks:

1. Implement structured JSON logs.
2. Implement request correlation IDs.
3. Implement API latency metrics.
4. Add provider timeouts.
5. Add controlled retries.
6. Add circuit breakers where appropriate.
7. Validate request payloads.
8. Implement API rate limiting.
9. Implement role-based access control for vendor and fleet operations.
10. Protect payment webhooks.
11. Secure API keys.
12. Prevent unauthorized reservation changes.
13. Handle database failures.
14. Handle Redis failures.
15. Handle routing provider failures.
16. Validate concurrent webhook processing.
17. Validate concurrent reservations.
18. Validate database transaction consistency.
19. Document error codes.
20. Implement safe fallback responses.

Deliverables:

- Secure backend.
- Reliability mechanisms.
- Structured logging.
- Backend security and failure tests.

## **Phase 18.3: Backend and Optimization Integration Testing**

Tasks:

1. Test all V1 API endpoints.
2. Test demand prediction.
3. Test candidate generation.
4. Test water filtering.
5. Test economics.
6. Test ranking.
7. Test all three decisions.
8. Test travel adjustment.
9. Test scenario simulation.
10. Test multi-stop planning.
11. Test time windows.
12. Test setup and teardown.
13. Test infeasible shift schedules.
14. Test reservations.
15. Test reservation expiry.
16. Test simultaneous dispatch.
17. Test webhook signatures.
18. Test duplicate payment events.
19. Test Telegram and WhatsApp reporting.
20. Test notification triggers.
21. Test database and Redis outages.
22. Test authorization and rate limits.

Deliverables: Backend integration test report and passing API/optimization tests.

## **Phase 19.3 and 20.3: Backend Release and Documentation**

**Phase 19.3: Backend release**

- Freeze API contracts.
- Verify authentication.
- Verify database migrations.
- Verify provider credentials.
- Verify payment integrations.
- Verify reservation consistency.
- Run production smoke tests.
- Verify API health and error handling.

**Phase 20.3: Backend documentation**

- Document all API endpoints.
- Document request and response schemas.
- Document economics formulas.
- Document ranking and decision rules.
- Document OR-Tools formulation.
- Document fleet dispatch.
- Document reservations.
- Document payment and messaging integrations.
- Document error handling and fallbacks.

**Final Member 3 completion:** All V1 APIs, decision intelligence, scenario simulation, multi-stop optimization, fleet management, payment integrations, messaging and notifications are delivered, tested and documented.

# **MEMBER 4: NITHIN, TECHNICAL LEAD, INTEGRATION, FRONTEND, INFRASTRUCTURE AND RELEASE**

Primary owner: Member 4

**Main objective:** Integrate all members' work into one working application, deliver the vendor-facing interface, manage infrastructure, automate testing, secure the repository and release the final system.

**Main repository ownership:**

```
dashboard/
.github/workflows/
Dockerfile
docker-compose.yml
pyproject.toml
uv.lock
.python-version
.env.example
.gitignore
README.md
docs/
tests/integration/
tests/e2e/
infra/
```

## **Phase 4: Technical Leadership and V1 Integration**

### **Phase 4.1: Repository Stabilization and Security**

Tasks:

1. Review the complete repository structure.
2. Add `.gitignore`.
3. Remove `.env` from Git tracking.
4. Identify exposed credentials.
5. Rotate exposed credentials.
6. Verify `.env.example` contains placeholders only.
7. Remove unused temporary files.
8. Review generated artifacts.
9. Review dependency versions.
10. Verify `pyproject.toml`.
11. Verify `uv.lock`.
12. Verify `.python-version`.
13. Standardize installation instructions.
14. Remove outdated documentation.
15. Establish branch ownership.
16. Protect `main`.
17. Require pull requests.
18. Require passing tests before merge.
19. Establish code review rules.
20. Maintain a task and defect tracker.

Deliverables:

- Clean repository.
- Secure environment configuration.
- Updated dependencies.
- Branch and PR workflow.
- Repository audit report.

### **Phase 4.2: Dashboard Completion**

Tasks:

1. Audit the existing Leaflet dashboard.
2. Preserve the existing HTML, CSS, JavaScript and Leaflet architecture.
3. Connect the frontend to FastAPI.
4. Implement or verify GPS location detection.
5. Implement manual location selection.
6. Display current vendor location.
7. Display H3 cell boundaries.
8. Display demand heatmap.
9. Display candidate locations.
10. Display Top-5 recommendations.
11. Display predicted customers.
12. Display P10/P50/P90.
13. Display expected revenue.
14. Display estimated profit.
15. Display travel cost.
16. Display travel distance.
17. Display travel time.
18. Display STAY, CONSIDER MOVE and RECOMMENDED MOVE.
19. Display recommendation explanations.
20. Display SHAP-based explanation data supplied by backend.
21. Display data freshness.
22. Display model version.
23. Display provider fallback status.
24. Implement loading states.
25. Implement no-result states.
26. Implement error states.
27. Implement responsive desktop and mobile layouts.
28. Verify displayed values against API responses.

Deliverables:

- Integrated Leaflet dashboard.
- Map and recommendation interface.
- Responsive UI.
- Dashboard QA report.

**Important:** The original master document mentions Streamlit, but the current repository has a Leaflet-based frontend. Continue with the actual implemented frontend unless the team explicitly approves a frontend migration. Do not maintain two competing dashboards unnecessarily.

### **Phase 4.3: Cross-Module Integration**

Tasks:

1. Receive Member 1's final feature store.
2. Receive Member 2's approved model.
3. Receive Member 3's stable API.
4. Validate the data-to-model contract.
5. Validate the model-to-API contract.
6. Validate the API-to-dashboard contract.
7. Validate feature names and ordering.
8. Validate model artifact loading.
9. Validate response schemas.
10. Validate economics calculations.
11. Validate decision outputs.
12. Validate SHAP explanation responses.
13. Remove obsolete mocked frontend responses.
14. Remove duplicate calculations from frontend.
15. Ensure all displayed values originate from the backend.
16. Implement cross-module integration tests.
17. Verify clean application startup.

Deliverables:

- Integrated application.
- Stable contracts.
- Cross-module tests.
- Integration report.

### **Phase 4.4: CI/CD and Automated Testing**

Tasks:

1. Create `.github/workflows/ci.yml`.
2. Configure dependency installation.
3. Configure formatting checks.
4. Configure linting.
5. Configure mypy.
6. Run unit tests.
7. Run integration tests.
8. Validate model artifacts.
9. Validate dataset schemas.
10. Build Docker image.
11. Publish test results.
12. Block PR merges when critical tests fail.
13. Add branch protection.
14. Add deployment workflow after CI is stable.
15. Configure release versioning.
16. Configure rollback workflow.

Deliverables:

- GitHub Actions workflow.
- Passing CI.
- Branch protection.
- Docker build validation.
- Release workflow.

### **Phase 4.5: Docker and V1 Deployment**

Tasks:

1. Audit Dockerfile.
2. Audit Docker Compose.
3. Verify environment variables.
4. Verify model artifact availability.
5. Verify dataset availability.
6. Verify API startup.
7. Verify dashboard startup.
8. Verify frontend-to-API communication.
9. Configure health checks.
10. Configure logs.
11. Configure persistent data where needed.
12. Test clean startup.
13. Test shutdown.
14. Test restart.
15. Document local deployment.

Deliverables:

- Working Docker image.
- Working Docker Compose.
- Deployment instructions.
- Container test report.

### **Phase 4.6: V1 Integration and Release QA**

Tasks:

1. Run complete V1 user journey.
2. Test location input.
3. Test vendor category selection.
4. Test context collection.
5. Test demand prediction.
6. Test quantile predictions.
7. Test candidate generation.
8. Test unsuitable-cell filtering.
9. Test revenue calculation.
10. Test profit calculation.
11. Test travel adjustment.
12. Test recommendation ranking.
13. Test decision reasons.
14. Test dashboard/API consistency.
15. Test provider failures.
16. Test application restart.
17. Test clean Docker deployment.
18. Fix integration defects.
19. Publish V1 release report.

Deliverables:

- V1 E2E report.
- Defect closure report.
- Stable V1 release.

## **Phase 8.4: V1 Security and Integration Baseline**

Tasks:

1. Recheck repository secrets.
2. Verify model artifact consistency.
3. Verify API contract stability.
4. Verify weather fallback fix.
5. Verify dependency reproducibility.
6. Verify branch protection.
7. Verify existing CI test execution.
8. Freeze the approved V1 contracts.
9. Confirm V1 issues are closed before V2 feature integration.

Deliverable: Approved V1 baseline for V2 development.

## **Phase 9.4: Shift Optimization Dashboard**

Tasks:

1. Integrate shift planning endpoint.
2. Add shift start-time selection.
3. Add shift duration selection.
4. Display ordered stops.
5. Display arrival times.
6. Display departure times.
7. Display dwell durations.
8. Display expected customers per stop.
9. Display revenue and profit per stop.
10. Display setup and teardown time.
11. Display travel distance.
12. Display travel time.
13. Display total shift profit.
14. Display optimized route on Leaflet.
15. Display solver failure or infeasible-plan states.
16. Compare the planned shift against the stay-at-current-location baseline.

Deliverable: Full-shift planning interface.

## **Phase 10.4: Fleet Dashboard**

Tasks:

1. Integrate fleet occupancy API.
2. Display active vendor locations.
3. Display occupied H3 cells.
4. Display reserved H3 cells.
5. Display available H3 cells.
6. Display competitor density.
7. Display category-specific occupancy.
8. Display reservation status.
9. Display fleet dispatch assignments.
10. Display dispatch conflicts.
11. Test dashboard updates after reservations expire.

Deliverable: Fleet operations dashboard.

## **Phase 11.4: Telemetry Dashboard Integration**

Tasks:

1. Integrate sales reporting interface.
2. Display transaction ingestion status.
3. Display data validation errors.
4. Display actual sales totals.
5. Display actual customer counts only where measurable.
6. Display actual-versus-predicted comparisons.
7. Display real versus simulated data labels.
8. Protect vendor transaction details.
9. Verify that unauthorized users cannot view another vendor's records.

Deliverable: Vendor telemetry and actual-versus-predicted interface.

## **Phase 12.4: Geospatial Dashboard Integration**

Tasks:

1. Integrate population layers.
2. Display population density.
3. Display competitor density.
4. Display competitor categories.
5. Display competitor operating hours where available.
6. Display data freshness.
7. Display source information.
8. Validate H3 layer alignment.
9. Validate all map layers across supported cities.

Deliverable: Enriched geospatial dashboard.

## **Phase 13.4: MLOps Platform Integration**

Tasks:

1. Integrate MLflow experiment tracking.
2. Integrate model registry information.
3. Integrate drift monitoring.
4. Display PSI.
5. Display Wasserstein Distance.
6. Display prediction performance.
7. Display model version.
8. Display last retraining time.
9. Display retraining status.
10. Display promotion and rollback history.
11. Configure retraining workflow triggers.
12. Ensure a drift event alone cannot automatically approve a model.

Deliverable: Model operations dashboard and controlled retraining workflow.

## **Phase 14.4: Multi-City Application Integration**

Tasks:

1. Implement city selector.
2. Load the correct city boundary.
3. Load the correct H3 cells.
4. Load the correct feature configuration.
5. Select the correct model or city configuration.
6. Display city-specific recommendations.
7. Handle unsupported regions.
8. Validate Hyderabad.
9. Validate Bengaluru.
10. Validate Visakhapatnam.
11. Document city onboarding.

Deliverable: Multi-city GeoDemand AI application.

## **Phase 15.4: Cloud-Native Infrastructure and AWS**

Tasks:

1. Design the production cloud architecture.
2. Create the `infra/` directory.
3. Define Terraform project structure.
4. Configure Terraform providers and state management.
5. Define VPC.
6. Define public and private subnets.
7. Define security groups.
8. Provision PostgreSQL 16 with PostGIS using RDS.
9. Provision Redis 7 using an appropriate managed service.
10. Configure ECS Fargate or EKS.
11. Select one orchestration platform and document why.
12. Configure ECS task definitions or Kubernetes manifests.
13. Configure container registry.
14. Configure application load balancer.
15. Configure SSL termination.
16. Configure DNS.
17. Configure HTTPS.
18. Configure horizontal autoscaling.
19. Configure AWS S3 for versioned datasets and artifacts.
20. Configure model artifact storage.
21. Configure MLflow tracking storage.
22. Configure AWS Secrets Manager.
23. Configure GitHub Actions deployment credentials securely.
24. Configure database backups.
25. Configure disaster recovery.
26. Configure deployment health checks.
27. Configure application logs.
28. Configure Prometheus metrics.
29. Verify infrastructure can be recreated using Terraform.
30. Test rollback and recovery.

Deliverables:

- AWS architecture.
- Terraform infrastructure.
- RDS/PostGIS.
- Redis.
- ECS/EKS deployment.
- S3 artifact storage.
- Secure secrets.
- Infrastructure deployment guide.

## **Phase 16.4: PWA and Mobile Vendor Experience**

Tasks:

1. Convert the existing dashboard into a Progressive Web App.
2. Create a web app manifest.
3. Configure service worker.
4. Support installation on compatible devices.
5. Improve mobile-first navigation.
6. Implement vendor login and profile access.
7. Implement GPS permission handling.
8. Implement location tracking controls.
9. Implement background location tracking where supported.
10. Respect operating-system and browser restrictions.
11. Display current vendor location.
12. Display demand recommendations.
13. Display shift itineraries.
14. Display fleet status where authorized.
15. Display push notifications.
16. Implement notification preferences.
17. Implement notification opt-out.
18. Add one-click navigation to recommended H3 cell centroids.
19. Support Google Maps external navigation.
20. Support Apple Maps external navigation.
21. Evaluate whether a Flutter application is necessary.
22. If selected, build Flutter vendor login, location, recommendation, notification and navigation features.
23. Test supported mobile browsers.
24. Test installation.
25. Test GPS permissions.
26. Test background behavior.
27. Test battery impact.
28. Test notification delivery.
29. Test mobile responsiveness.
30. Document mobile limitations.

Deliverables:

- Installable PWA.
- GPS experience.
- Push notification interface.
- External navigation.
- Flutter application only if approved as a separate client.

**Decision:** Do not build both a full PWA and a full Flutter application just to increase the feature count. Select the required production client based on vendor usability and supported-device requirements.

## **Phase 17.4: Platform Observability and Performance**

Tasks:

1. Integrate Prometheus.
2. Define application metrics.
3. Define API metrics.
4. Define ML metrics.
5. Define optimization metrics.
6. Define database metrics.
7. Define Redis metrics.
8. Build monitoring dashboards.
9. Configure critical failure alerts.
10. Configure service availability alerts.
11. Configure latency alerts.
12. Configure data freshness alerts.
13. Configure model drift alerts.
14. Measure single-location recommendation latency.
15. Measure shift optimization latency.
16. Run load tests.
17. Run concurrency tests.
18. Profile memory usage.
19. Profile CPU usage.
20. Test database query performance.
21. Test Redis cache performance.
22. Test routing cache performance.
23. Validate branch coverage of at least 90%.
24. Validate mypy.
25. Audit unused dependencies.
26. Run dependency vulnerability scans.
27. Test secrets handling.
28. Test backup restoration.
29. Test application rollback.
30. Publish performance evidence.

Deliverables:

- Prometheus monitoring.
- Operational dashboards.
- Alerting configuration.
- Performance report.
- Security report.
- Coverage report.
- mypy validation report.

## **Phase 18.4: Full-System Integration and Acceptance Testing**

Tasks:

1. Create the master integration test plan.
2. Create the master E2E test plan.
3. Define test environments.
4. Define test users and vendor scenarios.
5. Test the complete data pipeline.
6. Test the production ML model.
7. Test API contracts.
8. Test economics calculations.
9. Test ranking and decision logic.
10. Test scenario simulation.
11. Test multi-stop shift planning.
12. Test fleet dispatch.
13. Test reservations.
14. Test payment ingestion.
15. Test messaging integrations.
16. Test actuals aggregation.
17. Test population enrichment.
18. Test competitor updates.
19. Test multi-city support.
20. Test PostgreSQL/PostGIS.
21. Test Redis.
22. Test cloud deployment.
23. Test PWA.
24. Test GPS permissions.
25. Test notifications.
26. Test external navigation.
27. Test authentication and authorization.
28. Test provider outages.
29. Test database outages.
30. Test cache outages.
31. Test concurrent vendor activity.
32. Test application restart.
33. Test recovery after failure.
34. Compare dashboard values against API responses.
35. Fix critical integration defects.
36. Obtain sign-off from all three module owners.

Deliverables:

- Full integration report.
- E2E test report.
- Performance report.
- Security test report.
- Defect closure report.
- Acceptance sign-off.

## **Phase 19.4: Production Deployment and Release**

Tasks:

1. Freeze the release scope.
2. Freeze data versions.
3. Freeze model version.
4. Freeze feature schema.
5. Freeze API contracts.
6. Build versioned Docker images.
7. Deploy database and cache.
8. Deploy backend.
9. Deploy frontend/PWA.
10. Configure HTTPS.
11. Configure production environment variables.
12. Configure secrets.
13. Configure health checks.
14. Configure logs.
15. Configure metrics.
16. Configure scheduled jobs.
17. Configure ingestion jobs.
18. Configure retraining workflows.
19. Run production smoke tests.
20. Run database migration checks.
21. Run model inference checks.
22. Run recommendation checks.
23. Run fleet reservation checks.
24. Run notification checks.
25. Test backup.
26. Test restore.
27. Test rollback.
28. Create release tag.
29. Publish deployment instructions.
30. Record release version and known limitations.

Deliverables:

- Production deployment.
- Release version.
- CI/CD release pipeline.
- Smoke-test report.
- Rollback procedure.
- Deployment runbook.

## **Phase 20.4: Final Documentation, Research and Handover**

Tasks:

1. Update README.
2. Update `PROJECT_MASTER_DOC.md`.
3. Document final architecture.
4. Document all system modules.
5. Create architecture diagrams.
6. Create data-flow diagrams.
7. Create sequence diagrams.
8. Create deployment diagrams.
9. Document API endpoints.
10. Update OpenAPI/Swagger.
11. Document data sources.
12. Document feature definitions.
13. Document simulation methodology.
14. Document ML methodology.
15. Document quantile forecasting.
16. Document SHAP.
17. Document model comparisons.
18. Document optimization mathematics.
19. Document fleet coordination.
20. Document telemetry ingestion.
21. Document payment integrations.
22. Document geospatial enrichment.
23. Document multi-city support.
24. Document infrastructure.
25. Document monitoring.
26. Document security.
27. Document troubleshooting.
28. Document disaster recovery.
29. Document local setup.
30. Document cloud deployment.
31. Prepare final project presentation.
32. Prepare a working product demonstration.
33. Prepare screenshots and test evidence.
34. Prepare research methodology.
35. Prepare experimental results.
36. Document simulated-data limitations.
37. Document real-data limitations.
38. Prepare research paper draft.
39. Prepare technical interview documentation.
40. Conduct final team handover.

Deliverables:

- Complete technical documentation.
- Architecture diagrams.
- API documentation.
- Deployment runbooks.
- Research paper draft.
- Final presentation.
- Demo evidence.
- Team handover package.

**Final Member 4 completion:** All modules are integrated, secure, tested, deployed, monitored and documented. The final application is released with reproducible setup and recovery procedures.

# **Shared execution rules for all four members**

These rules apply to every phase. They are part of the completion criteria, not optional project-management suggestions.

## **1. Branch and pull-request workflow**

Use the agreed branch structure:

```
main
├── dev/member-1
├── dev/member-2
├── dev/member-3
└── dev/nithin
```

1. Every member works on their own branch.
2. No direct pushes to protected `main`.
3. Every completed task goes through a pull request.
4. Each pull request must include tests.
5. Each pull request must include relevant documentation updates.
6. Dependencies and shared interfaces must be discussed before changing them.
7. Nithin reviews integration and cross-module compatibility.
8. Failed critical CI checks block merging.

## **2. Module handover order**

1. **Member 1 → Member 2**

   Provides validated data, simulation, feature store, schema and metadata.
2. **Member 2 → Member 3**

   Provides approved model, inference interface, quantile outputs and SHAP.
3. **Member 3 → Member 4**

   Provides stable APIs, economics, decisions, optimization and integrations.
4. **Member 4 → All members**

   Integrates the system, performs cross-module testing and coordinates release.

For V2, members will work in parallel when dependencies allow. For example, Member 1 can prepare transaction schemas while Member 3 builds webhook endpoints, but neither should declare the integration complete until actual events pass through the full pipeline.

## **3. Mandatory handover package**

Every module owner must submit:

- Source code.
- Configuration files.
- Dependency changes.
- Unit tests.
- Integration tests where applicable.
- Test execution results.
- API or module interface documentation.
- Known issues and limitations.
- Setup and execution instructions.
- Pull request and review approval.

## **4. Final project acceptance criteria**

Project completion standard

| Area                  | Final acceptance requirement                                                    |
| --------------------- | ------------------------------------------------------------------------------- |
| Data                  | Reproducible pipelines, validated features, provenance and temporal correctness |
| Simulation            | Deterministic and documented, with hidden variables excluded from model inputs  |
| ML                    | One approved model, reproducible evaluation, quantiles and SHAP                 |
| MLOps                 | MLflow, drift monitoring, controlled retraining, promotion and rollback         |
| Decision intelligence | Valid demand, revenue, profit, travel economics and decisions                   |
| Scenario simulation   | Different conditions can be evaluated without retraining                        |
| Routing               | Valid multi-stop plans with time and operational constraints                    |
| Fleet                 | Reservations, competition adjustment and multi-vendor dispatch                  |
| Telemetry             | Secure payment/reporting ingestion and actuals aggregation                      |
| Geospatial            | Population, competitors, operating hours and H3 enrichment                      |
| Multi-city            | Correct data, model and recommendations for supported cities                    |
| Infrastructure        | PostgreSQL/PostGIS, Redis, Docker, Terraform and cloud deployment               |
| Mobile                | Working PWA, GPS permissions, notifications and navigation                      |
| Security              | Secure secrets, authentication, authorization and protected webhooks            |
| Testing               | Unit, integration, E2E, concurrency and recovery tests                          |
| Performance           | Measured latency, resource usage and load-test results                          |
| Quality               | At least 90% branch coverage target and clean mypy validation                   |
| Monitoring            | Structured logs, Prometheus, alerts and drift reporting                         |
| Deployment            | Reproducible deployment, health checks, backups and rollback                    |
| Documentation         | Setup, architecture, APIs, operations and troubleshooting                       |
| Research              | Reproducible experiments, honest metrics and disclosed limitations              |

### **Important targets from the V2 roadmap**

| Metric                                 | Target            |
| -------------------------------------- | ----------------- |
| Single-location recommendation latency | Under 150 ms      |
| Full-shift optimization latency        | Under 1.5 seconds |
| Branch coverage                        | At least 90%      |
| Temporal leakage                       | Zero              |
| Type checking                          | mypy clean        |

These must be measured under documented test conditions. Do not mark them complete based on assumptions.

# **Final feature coverage: what the team must actually deliver**

The final product must support this complete journey:

## **GeoDemand AI complete product workflow**

1. Vendor registration, profile and business configuration.
2. GPS or manual location selection.
3. Real weather, calendar, POI and competitor context.
4. H3 spatial feature generation and demographic enrichment.
5. Next-hour and multi-hour demand forecasting.
6. P10/P50/P90 uncertainty and SHAP explanations.
7. Revenue, COGS, expenses, fuel and profit calculations.
8. Candidate location generation and ranking.
9. STAY, CONSIDER MOVE and RECOMMENDED MOVE decisions.
10. What-if scenario simulation.
11. Full-shift OR-Tools route optimization.
12. Fleet demand sharing, reservations and dispatch.
13. Real transaction and vendor sales ingestion.
14. Actual-versus-predicted monitoring.
15. Multi-city recommendations.
16. Mobile PWA, GPS, proactive alerts and navigation.
17. Continuous model evaluation and controlled retraining.
18. Production infrastructure, monitoring and recovery.

## **Three decisions that must be settled before implementation**

These are alternatives or architectural choices in the roadmap. Do not implement every alternative simultaneously without a reason.

1. **ECS Fargate or EKS:** Select one deployment platform. Both are not required.
2. **PWA or Flutter:** Build the PWA first unless native mobile functionality justifies Flutter.
3. **Kafka or RabbitMQ:** Use a broker if event throughput, asynchronous processing and delivery guarantees require it. Otherwise, start with a simpler event-processing design and preserve the ability to add a broker.

Also retain the existing Leaflet dashboard rather than rebuilding it in Streamlit without an explicit architecture decision.

## **My final recommendation on execution**

Follow this order:

1. Finish Member 1's V1 data foundation.
2. Finish Member 2's V1 model and MLOps foundation.
3. Finish Member 3's V1 backend and decision engine.
4. Finish Member 4's V1 integration and release.
5. Freeze and validate V1.
6. Execute V2 routing and fleet coordination.
7. Execute real telemetry and geospatial enrichment.
8. Execute advanced ML and multi-city support.
9. Execute cloud infrastructure and mobile experience.
10. Complete reliability, integration testing, deployment and documentation.

**Do not declare the entire project complete just because all four members have finished their assigned coding tasks.** The project is complete only when the final integrated application passes the acceptance criteria, the release is reproducible, and the remaining limitations are documented.

This is the member-wise execution structure I would use as the team's master task plan. It covers the original V1 architecture and the V2 functional roadmap while keeping each member's ownership clear.