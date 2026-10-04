## 📌 Description
<!-- Provide a clear, concise summary of the changes introduced in this pull request and the motivation behind them. -->

## 🎯 Type of Change
<!-- Select all that apply by replacing [ ] with [x] -->
- [ ] 🚀 **New Feature** (non-breaking change which adds functionality)
- [ ] 🐛 **Bug Fix** (non-breaking change which fixes an issue)
- [ ] ⚡ **Performance / Latency Improvement** (optimizations to inference, geospatial indexing, or routing)
- [ ] 🧠 **Machine Learning & Features** (model training, LightGBM quantile regression, TreeSHAP, feature engineering)
- [ ] 🗺️ **Geospatial & Ingestion** (H3 grid resolution, OSM POI pipeline, Open-Meteo weather, polygon filtering)
- [ ] 🛠️ **DevOps & Infrastructure** (Docker, docker-compose, CI/CD GitHub Actions, dependencies)
- [ ] 📝 **Documentation** (README, architecture specs, data catalog, API contracts)
- [ ] 🧹 **Refactor / Code Quality** (code clean-up without changing behavior)

---

## 🔍 Domain & Engineering Verification
<!-- Verify key GeoDemand AI production contracts -->
- [ ] **Anti-Circularity Contract**: Verified no future data leakage ($t \to t+1$ temporal boundary preserved).
- [ ] **Spatial & H3 Resolution**: Uber H3 resolution-8 cell integrity, coordinate transforms, and water-body exclusions verified.
- [ ] **Quantile Monotonicity**: Predictions adhere to non-crossing quantile bounds ($P_{10} \le P_{50} \le P_{90}$).
- [ ] **Explainability**: TreeSHAP feature attributions run within expected latency budget (<15ms) and return valid contributions.
- [ ] **Economic & Route Friction**: Travel duration penalty ($\frac{60 - T_{\text{travel}}}{60}$) and net profit calculations validated.
- [ ] **API & Schema Contracts**: Pydantic input/output schemas remain backward-compatible or versioned properly.

---

## 🧪 Testing & Validation
<!-- Detail how the changes were verified -->
- [ ] Unit & contract test suite passed:
  ```bash
  python -m pytest tests/ -v --tb=short
  ```
- [ ] Model inference verification executed successfully:
  ```bash
  PYTHONPATH="src:src/data:src/features:src/api:." python -c "from api.recommender import DemandModel; print(DemandModel.get().meta)"
  ```
- [ ] Manual endpoint testing conducted (e.g. `/v1/recommend`, `/v1/system/ping`)
- [ ] (If applicable) Dashboard visualization verified in browser

---

## 📸 Screenshots / Visual Evidence (If applicable)
<!-- Attach screenshots, dashboard diffs, or terminal outputs demonstrating the change -->

---

## ⚠️ Breaking Changes & Environment Updates
- [ ] Requires updates to `.env` or `.env.example`
- [ ] Requires database / feature store parquet rebuild
- [ ] Requires model retrain / artifact re-generation (`models/demand_model.pkl`)
- [ ] None of the above

---

## 📋 Git Standards Checklist
- [ ] **Branch Naming**: Branch follows convention (`feature/<name>`, `fix/<name>`, `docs/<name>`, `chore/<name>`).
- [ ] **Commit Messages**: Commits adhere to conventional formatting (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`, `ci:`). No vague commit messages.

---

## 🔗 Related Issues
<!-- Link related issues or discussions: Closes #123, Fixes #456 -->
- Closes #
