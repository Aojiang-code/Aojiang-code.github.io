import os, json, numpy as np, pandas as pd
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
from .plotting_calibration import plot_calibration_with_ci
from .plotting_dca import decision_curve

def evaluate_and_plot(cfg: Dict[str, Any], y_true, y_prob, y_pred, split_tag="val"):
    plots_dir = cfg["paths"]["plots"]
    os.makedirs(plots_dir, exist_ok=True)

    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    # specificity = TN/(TN+FP)
    tn = np.sum((y_pred==0) & (np.array(y_true)==0))
    fp = np.sum((y_pred==1) & (np.array(y_true)==0))
    spec = tn / (tn + fp + 1e-9)

    # ROC
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({split_tag}) AUC={auc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"roc_{split_tag}.png"), dpi=200)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({split_tag}) AP={auprc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"pr_{split_tag}.png"), dpi=200)

    # Calibration
    cal_res = plot_calibration_with_ci(y_true, y_prob, n_bins=cfg["evaluation"]["calibration_bins"],
                                       n_boot=cfg["evaluation"]["calibration_bootstrap"],
                                       title=f"Calibration ({split_tag})",
                                       out_path=os.path.join(plots_dir, f"calibration_{split_tag}.png"))

    # DCA
    decision_curve(y_true, y_prob, title=f"DCA ({split_tag})",
                   out_path=os.path.join(plots_dir, f"dca_{split_tag}.png"))

    metrics = {
        "auc": float(auc), "auprc": float(auprc), "brier": float(brier),
        "accuracy": float(acc), "sensitivity": float(sens), "specificity": float(spec),
        "ece": float(cal_res["ece"])
    }

    rep_path = os.path.join(cfg["paths"]["reports"], f"metrics_{split_tag}.json")
    with open(rep_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
