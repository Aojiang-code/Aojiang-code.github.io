import argparse, pandas as pd, numpy as np
from pathlib import Path
from utils import load_config, ensure_dirs

def main(cfg_path):
    cfg = load_config(cfg_path); ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["out_dir"])

    df = pd.read_parquet(out_dir / "cohort_with_labels.parquet")

    thr = cfg["preprocessing"]["variable_missing_threshold"]
    miss = df.isna().mean()
    drop_cols = miss[miss > thr].index.tolist()
    keep = [c for c in df.columns if c not in drop_cols]
    print(f"[Info] Dropping {len(drop_cols)} columns for missingness > {thr}: {drop_cols}")
    df = df[keep]

    id_cols = ["subject_id","hadm_id","stay_id","intime"]
    feat_num = cfg["features"]["candidate_numeric"]
    feat_cat = cfg["features"]["candidate_categorical"]
    cols = [c for c in id_cols + feat_num + feat_cat + ["aki_7d"] if c in df.columns]
    X = df[cols].copy()

    X.to_parquet(out_dir / "features_24h.parquet", index=False)
    print("[OK] Features saved:", out_dir / "features_24h.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
