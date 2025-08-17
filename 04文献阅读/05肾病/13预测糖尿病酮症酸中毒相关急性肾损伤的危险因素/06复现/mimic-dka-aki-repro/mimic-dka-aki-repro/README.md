# MIMIC-IV DKA→AKI (7-day) Reproduction

This repository reproduces a study predicting **AKI within 7 days** among **DKA** ICU patients using MIMIC-IV.
It aligns to the paper's core choices: **24h feature window**, **KDIGO**-based AKI labeling, **LASSO→XGBoost** modeling,
plus **calibration curves** and **Decision Curve Analysis (DCA)**.

## Quick Start

1. Install dependencies (conda recommended):
   ```bash
   conda create -n mimic-repro python=3.10 -y
   conda activate mimic-repro
   pip install -r requirements.txt
   ```

2. Edit **conf/config.yaml** to point to your local MIMIC-IV data. Two modes:
   - **DuckDB over CSV/Parquet** (recommended for local files): set `data_source.type: duckdb`
     and provide the `paths.mimic_root` pointing to folders such as `mimiciv/hosp/*.csv.gz`,
     `mimiciv/icu/*.csv.gz` or Parquet files. The code maps common filenames automatically.
   - **Postgres** (optional): set `data_source.type: postgres` and fill host/port/db/user/password.

3. Run the end-to-end pipeline:
   ```bash
   python -m src.run_all --config conf/config.yaml
   ```

Artifacts:
- Intermediate tables: `data/interim/*.parquet`
- Final features & labels: `data/processed/features.parquet`, `data/processed/labels.parquet`
- Model & metrics: `data/models/*`, `outputs/reports/metrics.json`
- Plots: `outputs/plots/roc.png`, `outputs/plots/pr.png`, `outputs/plots/calibration.png`, `outputs/plots/dca.png`

> Note: You must have local access to MIMIC-IV (DUA-compliant). For `chartevents` scale, DuckDB is efficient.

## Project Layout

```
mimic-dka-aki-repro/
├─ conf/
│  └─ config.yaml
├─ codesets/
│  └─ icd_dka_codes.csv
├─ sql/
│  ├─ cohort.sql
│  └─ itemids_lookup.sql
├─ src/
│  ├─ data_source.py
│  ├─ extract.py
│  ├─ labels_kdigo.py
│  ├─ features.py
│  ├─ train.py
│  ├─ eval.py
│  ├─ plotting_calibration.py
│  ├─ plotting_dca.py
│  └─ run_all.py
├─ data/
│  ├─ raw/        # optional local snapshots
│  ├─ interim/
│  ├─ processed/
│  └─ models/
├─ outputs/
│  ├─ plots/
│  └─ reports/
└─ requirements.txt
```

## Repro Settings (align to paper)

- Population: adults (≥18y), **first ICU stay** per hospitalization.
- Case selection: **DKA** by ICD-9/10 (see `codesets/icd_dka_codes.csv`).
- Features: first/summary values **within 0–24h** of ICU admission (BUN, urine output, weight, age, PLT, glucose, input volume, etc.).
- Outcome: **AKI within 7 days** by KDIGO (Scr/urine criteria).
- Split: **random 85/15**, 10-fold CV inside training; LASSO for feature pre-selection → XGBoost as main model.
- Evaluation: AUC, AUPRC, Brier, ECE, sensitivity, specificity, accuracy; **calibration curve with 95% CI**; **DCA**.

For any departures (e.g., missing fields in your local MIMIC), adapt `conf/config.yaml` accordingly.
