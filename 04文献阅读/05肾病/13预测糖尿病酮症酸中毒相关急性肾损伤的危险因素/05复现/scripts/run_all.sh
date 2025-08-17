#!/usr/bin/env bash
set -euo pipefail

python src/01_extract_duckdb.py --config 00_config.yaml
python src/02_make_labels.py --config 00_config.yaml
python src/03_features_24h.py --config 00_config.yaml
python src/04_train_eval.py --config 00_config.yaml
python src/05_calibration_dca.py --config 00_config.yaml
python src/06_shap_pdp.py --config 00_config.yaml
