import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from utils import load_config, ensure_dirs, youden_threshold, compute_clf_metrics, brier_score, save_json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

def main(cfg_path):
    cfg = load_config(cfg_path); ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["out_dir"]); work_dir = Path(cfg["paths"]["work_dir"])

    df = pd.read_parquet(out_dir / "features_24h.parquet")
    y = df["aki_7d"].astype(int).values
    groups = df[cfg["split"]["group_by"]].values
    num_cols = [c for c in cfg["features"]["candidate_numeric"] if c in df.columns]
    cat_cols = [c for c in cfg["features"]["candidate_categorical"] if c in df.columns]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", KNNImputer(n_neighbors=cfg["preprocessing"]["imputation"]["knn"]["n_neighbors"])),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ],
        remainder="drop"
    )

    lasso = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegressionCV(
            Cs=10, cv=5, penalty="l1", solver="saga", max_iter=5000, scoring="roc_auc", n_jobs=-1, refit=True
        ))
    ])

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["split"]["test_size"], random_state=cfg["seed"])
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    df_tr, df_te = df.iloc[train_idx], df.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Proxy preselection: rank numeric features by univariate AUC on train
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in num_cols:
        m = df_tr[c].values
        mask = np.isfinite(m)
        if mask.sum() < 10:
            aucs.append((c, 0.0)); continue
        try:
            aucs.append((c, roc_auc_score(y_tr[mask], m[mask])))
        except Exception:
            aucs.append((c, 0.0))
    aucs.sort(key=lambda x: x[1], reverse=True)
    k = cfg["features"]["lasso_target_k"]
    top_num = sorted(set([x[0] for x in aucs[:k]] + [c for c in num_cols if c in ["scr_first24h"]]))
    print("[Info] Selected numeric vars (proxy):", top_num)

    pre_sel = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", KNNImputer(n_neighbors=cfg["preprocessing"]["imputation"]["knn"]["n_neighbors"])),
                              ("sc", StandardScaler())]), top_num),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ],
        remainder="drop"
    )

    models = {}
    models["logreg"] = Pipeline([("pre", pre_sel),
                                 ("clf", LogisticRegressionCV(Cs=10, cv=5, penalty="l2", solver="lbfgs",
                                                              scoring="roc_auc", max_iter=2000, n_jobs=-1, refit=True))])
    models["svm"] = Pipeline([("pre", pre_sel),
                              ("clf", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced"))])
    models["adaboost"] = Pipeline([("pre", pre_sel),
                                   ("clf", AdaBoostClassifier(n_estimators=300, random_state=cfg["seed"]))])
    models["mlp"] = Pipeline([("pre", pre_sel),
                              ("clf", MLPClassifier(hidden_layer_sizes=(64,32), alpha=1e-3, max_iter=300, random_state=cfg["seed"]))])
    try:
        import xgboost as xgb
        models["xgboost"] = Pipeline([("pre", pre_sel),
                                      ("clf", xgb.XGBClassifier(
                                          n_estimators=500, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                          random_state=cfg["seed"], n_jobs=-1, eval_metric="auc"
                                      ))])
    except Exception as e:
        print("[WARN] xgboost not available:", e)

    results = {}
    for name, pipe in models.items():
        pipe.fit(df_tr[top_num+cat_cols], y_tr)
        prob = pipe.predict_proba(df_te[top_num+cat_cols])[:,1]
        thr = (youden_threshold(y_te, prob) if cfg["evaluation"]["thresholds"]["optimize"]=="youden"
               else cfg["evaluation"]["thresholds"]["fixed_value"])
        metrics = compute_clf_metrics(y_te, prob, thr=thr)
        metrics["Brier"] = float(np.mean((prob - y_te)**2))
        metrics["Threshold"] = float(thr)
        results[name] = metrics
        print(f"[{name}] AUC={metrics['AUC']:.3f} PR-AUC={metrics['PR_AUC']:.3f} Acc={metrics['Accuracy']:.3f} "
              f"Sens={metrics['Sensitivity']:.3f} Spec={metrics['Specificity']:.3f} Thr={thr:.3f}")

        # Save DCA arrays
        from utils import decision_curve
        dca = decision_curve(y_te, prob,
                             p_min=cfg["evaluation"]["dca"]["p_min"],
                             p_max=cfg["evaluation"]["dca"]["p_max"],
                             p_step=cfg["evaluation"]["dca"]["p_step"])
        from utils import save_json
        save_json(dca, work_dir / f"dca_{name}.json")

    save_json(results, work_dir / "metrics_summary.json")
    print("[OK] Saved metrics to", work_dir / "metrics_summary.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
