import os, json, numpy as np, pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def _selector_model(cfg: Dict[str, Any]):
    sel = cfg["model"]["selector"]
    if sel["type"] == "logreg_l1":
        return LogisticRegression(penalty="l1", solver="saga", C=sel.get("C", 0.1), max_iter=sel.get("max_iter", 2000), n_jobs=-1)
    else:
        raise NotImplementedError

def _main_model(cfg: Dict[str, Any]):
    main = cfg["model"]["main"]
    if main["type"] == "xgboost":
        return XGBClassifier(**main["params"])
    elif main["type"] == "lightgbm":
        return LGBMClassifier(**main["params"])
    elif main["type"] == "logreg":
        return LogisticRegression(max_iter=5000, n_jobs=-1)
    else:
        raise NotImplementedError

def _calibrator(estimator, method: str):
    if method == "isotonic":
        return CalibratedClassifierCV(estimator, method="isotonic", cv=5)
    elif method == "platt":
        return CalibratedClassifierCV(estimator, method="sigmoid", cv=5)
    else:
        return estimator

def prepare_xy(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feats = pd.read_parquet(os.path.join(cfg["paths"]["processed"], "features.parquet"))
    labels = pd.read_parquet(os.path.join(cfg["paths"]["processed"], "labels.parquet"))
    df = feats.merge(labels, on=["subject_id", "hadm_id", "stay_id"], how="inner")

    # target
    y = df["aki7d"].astype(int)
    # drop ID columns & target
    drop_cols = ["subject_id", "hadm_id", "stay_id", "aki7d"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # basic encoding for admission_type if present
    if "admission_type" in X.columns:
        X = pd.get_dummies(X, columns=["admission_type"], drop_first=True)
    # scale numeric
    X_num = X.select_dtypes(include=[np.number])
    X[X_num.columns] = X_num  # already numeric
    return X, y, df

def lasso_select(cfg: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    sel = _selector_model(cfg)
    pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("sel", sel)])
    pipe.fit(X, y)
    if hasattr(sel, "coef_"):
        coef = sel.coef_.ravel()
        keep = X.columns[np.where(coef!=0)[0]]
        if len(keep)==0:  # fallback keep top based on absolute coef
            abs_idx = np.argsort(np.abs(coef))[::-1][:min(10, X.shape[1])]
            keep = X.columns[abs_idx]
    else:
        keep = X.columns
    return X[keep]

def train_eval(cfg: Dict[str, Any]):
    X, y, df = prepare_xy(cfg)
    # Train/val split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=cfg["split"]["random_train_ratio"],
        stratify=y, random_state=cfg["split"]["random_state"]
    )

    # LASSO selection on train only
    Xtr_sel = lasso_select(cfg, Xtr, ytr)
    keep_cols = Xtr_sel.columns.tolist()
    Xte_sel = Xte[keep_cols]

    # main model
    model = _main_model(cfg)
    model.fit(Xtr_sel, ytr)

    # calibration
    cal_method = cfg["model"]["calibrator"]
    calibrated = _calibrator(model, cal_method)
    if isinstance(calibrated, CalibratedClassifierCV):
        calibrated.fit(Xtr_sel, ytr)
    else:
        calibrated = model

    # predict
    yprob_tr = calibrated.predict_proba(Xtr_sel)[:,1]
    ypred_tr = (yprob_tr >= 0.5).astype(int)

    yprob_te = calibrated.predict_proba(Xte_sel)[:,1]
    ypred_te = (yprob_te >= 0.5).astype(int)

    # CV on training
    cv = StratifiedKFold(n_splits=cfg["split"]["cv_folds"], shuffle=True, random_state=cfg["split"]["random_state"])
    cv_auc = cross_val_score(calibrated, Xtr_sel, ytr, scoring="roc_auc", cv=cv, n_jobs=-1).mean()

    # Save
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    joblib.dump({"model": model, "calibrated": calibrated, "features": keep_cols}, os.path.join(cfg["paths"]["models"], "model.joblib"))
    with open(os.path.join(cfg["paths"]["models"], "keep_features.json"), "w") as f:
        json.dump(keep_cols, f, indent=2)

    return (Xtr_sel, ytr, yprob_tr, ypred_tr, Xte_sel, yte, yprob_te, ypred_te, cv_auc)
