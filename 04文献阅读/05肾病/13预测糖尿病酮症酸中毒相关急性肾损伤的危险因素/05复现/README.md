# MIMIC-IV Reproduction Template: DKA → AKI (7-day) Prediction

This scaffold implements the optimized plan we discussed: DuckDB + CSV + Python pipeline with
grouped splits, leakage-safe preprocessing, LASSO preselection, multi-model comparison, calibration, DCA, and SHAP.

> **Scope & Disclaimer**
> - This is a **template**. You must fill in the exact ICD and itemid lists in `configs/mimic_ids.yaml`.
> - KDIGO label generation is implemented with clear, auditable steps, but you should **confirm** the baseline SCr rule and urine-output windowing in your context (see `00_config.yaml`).
> - The code expects access to *deidentified* MIMIC-IV CSVs mounted locally (read-only).

## 0) Environment

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 1) Configure

Edit `00_config.yaml` and `configs/mimic_ids.yaml`:
- point `mimic_csv_dir` to your local MIMIC-IV CSV root (e.g., `/path/to/mimic-iv-2.2`).
- confirm **time windows** (24h features; 7-day outcome; KDIGO ops).
- update ICD-9/10 lists for **DKA**; update `itemids` for SCr, BUN, PLT, urine output, weight, glucose, fluids, vitals.

## 2) Run (end-to-end)

```bash
bash scripts/run_all.sh
```

This performs:
1. `01_extract_duckdb.py`: cohort & 24h window extraction to Parquet
2. `02_make_labels.py`: KDIGO-based AKI within 7d
3. `03_features_24h.py`: feature table build & missingness filtering
4. `04_train_eval.py`: 85/15 grouped split, 10-fold CV, models, metrics
5. `05_calibration_dca.py`: calibration curves & DCA arrays
6. `06_shap_pdp.py`: SHAP importance & PDP utilities

Outputs appear under `intermediate/` and `work/`.

## 3) Repro Controls

- Random seed: `seed` in `00_config.yaml` (default 42).
- Grouped split: `group_by = subject_id`; test_size = 0.15.
- Missingness: drop columns with fraction > `variable_missing_threshold` (default 0.2); KNN impute (train-only fit) with `n_neighbors` (default 5).
- LASSO preselect to target `lasso_target_k` (default 7 features), then main models.
- Thresholding: Youden J on ROC (default) + report at 0.5 for comparability.

## 4) Notes on KDIGO (Implementations Here)

- SCr criteria: increase ≥0.3 mg/dL in 48h **or** ≥1.5× baseline within 7 days.
- Urine output criteria: <0.5 mL/kg/h for ≥6h (we implement rolling windows; requires weight).

> You can switch baseline rule via `config.outcomes.kdigo.baseline_method`:
> - `first_24h`: first SCr within 24h after ICU intime (default)
> - `lowest_24h`: lowest SCr within 24h after ICU intime
> - `pre48h_to_post6h_min`: lowest SCr from 48h before to 6h after ICU intime (if available)

## 5) Compare with Paper

Use the generated `work/metrics_summary.json` and `work/feature_importance.json`.
Add your **paper-vs-reproduction** table in your manuscript or appendix.

---

**Tip**: If performance drifts >±0.03 AUC from the paper, check: cohort ICDs, itemids, baseline SCr rule, urine windowing, imputation config, and 85/15 grouping.
