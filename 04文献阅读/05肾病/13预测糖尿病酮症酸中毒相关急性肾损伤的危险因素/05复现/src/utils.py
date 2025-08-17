import yaml, json, os, warnings, numpy as np, pandas as pd
from pathlib import Path

def load_config(path: str|Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dirs(cfg):
    Path(cfg["paths"]["out_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["work_dir"]).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return np.mean((y_prob - y_true)**2)

def youden_threshold(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k])

def confusion_at_threshold(y_true, y_prob, thr):
    import numpy as np
    y_pred = (y_prob >= thr).astype(int)
    TP = int(((y_true==1)&(y_pred==1)).sum())
    TN = int(((y_true==0)&(y_pred==0)).sum())
    FP = int(((y_true==0)&(y_pred==1)).sum())
    FN = int(((y_true==1)&(y_pred==0)).sum())
    return dict(TP=TP,TN=TN,FP=FP,FN=FN)

def compute_clf_metrics(y_true, y_prob, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    y_pred = (y_prob >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred))
    spec = float(((y_true==0)&(y_pred==0)).sum() / (y_true==0).sum())
    f1 = float(f1_score(y_true, y_pred))
    return dict(AUC=auc, PR_AUC=ap, Accuracy=acc, Sensitivity=sens, Specificity=spec, F1=f1)

def decision_curve(y_true, y_prob, p_min=0.1, p_max=0.6, p_step=0.02):
    import numpy as np
    y_true = np.asarray(y_true).astype(int)
    ps = np.arange(p_min, p_max + 1e-9, p_step)
    n = len(y_true)
    out = []
    for pt in ps:
        thr = pt
        y_pred = (y_prob >= thr).astype(int)
        TP = ((y_true==1)&(y_pred==1)).sum()
        FP = ((y_true==0)&(y_pred==1)).sum()
        net_benefit = (TP/n) - (FP/n)*(pt/(1-pt))
        out.append(dict(threshold=float(pt), net_benefit=float(net_benefit)))
    return out
