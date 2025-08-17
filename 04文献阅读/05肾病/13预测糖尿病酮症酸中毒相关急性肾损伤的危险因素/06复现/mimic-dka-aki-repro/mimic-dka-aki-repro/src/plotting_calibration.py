import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_with_ci(y_true, y_prob, n_bins=10, n_boot=1000, title="Calibration", out_path=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')

    # bootstrap CI
    rng = np.random.RandomState(42)
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, len(y_true), len(y_true))
        yt_b = y_true[idx]; yp_b = y_prob[idx]
        try:
            pt_b, pp_b = calibration_curve(yt_b, yp_b, n_bins=n_bins, strategy='quantile')
            if len(pt_b) == len(prob_true):
                boots.append(pt_b)
        except Exception:
            continue

    if len(boots) > 0:
        import numpy as np
        boots = np.vstack(boots)
        lower = np.percentile(boots, 2.5, axis=0)
        upper = np.percentile(boots, 97.5, axis=0)
    else:
        lower = upper = None

    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle='--', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    if lower is not None:
        plt.fill_between(prob_pred, lower, upper, alpha=0.2, label='95% CI')
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    plt.title(f"{title} (ECE={ece:.3f})")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend(loc='best')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    return {"prob_pred": prob_pred.tolist(), "prob_true": prob_true.tolist(), "ece": ece}
