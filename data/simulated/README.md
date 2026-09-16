# Legacy v1 Simulated Dataset

`dataset_v2bc7185bc94d.parquet` (+ its `.meta.json`) is the output of the
**v1 monolithic simulator** (`src/simulation/market_simulator.py`) —
one flat CSV-shaped table generated in a single pass, before the pipeline
was restructured into the 10 independent, single-responsibility datasets
under `src/data/` (see `docs/PROJECT_MASTER_DOC.md`, Section 6.6).

It's kept here deliberately, not deleted, as a visible record of the
design iteration: monolithic table → separated pipeline with real data
sources, explicit join contracts, and a leakage guard. If asked "why does
this file still exist," that's the honest answer — it's superseded, not
forgotten.

The current, actively-used dataset is `data/processed/feature_store.parquet`,
built by `src/features/feature_store.py` from the datasets in
`data/processed/`.
