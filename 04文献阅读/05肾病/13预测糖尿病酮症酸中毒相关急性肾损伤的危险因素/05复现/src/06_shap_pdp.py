import argparse, pandas as pd, numpy as np
from pathlib import Path
from utils import load_config, ensure_dirs, save_json

def main(cfg_path):
    cfg = load_config(cfg_path); ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["out_dir"]); work_dir = Path(cfg["paths"]["work_dir"])
    df = pd.read_parquet(out_dir / "features_24h.parquet")
    y = df["aki_7d"].astype(int).values
    num_cols = [c for c in cfg["features"]["candidate_numeric"] if c in df.columns]
    cat_cols = [c for c in cfg["features"]["candidate_categorical"] if c in df.columns]

    try:
        import xgboost as xgb, shap
    except Exception as e:
        print("[WARN] xgboost/shap not available:", e); return

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", KNNImputer(n_neighbors=cfg["preprocessing"]["imputation"]["knn"]["n_neighbors"])),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ],
        remainder="drop"
    )
    model = Pipeline([("pre", pre),
                      ("clf", xgb.XGBClassifier(
                          n_estimators=500, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=cfg["seed"], n_jobs=-1, eval_metric="auc"
                      ))])
    X = df[num_cols+cat_cols]
    model.fit(X, y)
    Xp = model.named_steps["pre"].transform(X)
    booster = model.named_steps["clf"]
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(Xp, check_additivity=False)
    import numpy as np
    mean_abs = np.abs(shap_values).mean(axis=0).tolist()
    feature_names = model.named_steps["pre"].get_feature_names_out().tolist()
    imp = sorted(list(zip(feature_names, mean_abs)), key=lambda x: x[1], reverse=True)
    save_json([{"feature": f, "mean_abs_shap": float(v)} for f, v in imp], work_dir / "feature_importance.json")
    print("[OK] Saved SHAP importance to", work_dir / "feature_importance.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
