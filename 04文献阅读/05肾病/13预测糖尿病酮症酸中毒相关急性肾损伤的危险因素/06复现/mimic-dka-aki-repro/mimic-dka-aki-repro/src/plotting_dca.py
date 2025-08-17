import numpy as np
import matplotlib.pyplot as plt

def net_benefit(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= threshold).astype(int)
    tp = np.sum((pred==1) & (y_true==1))
    fp = np.sum((pred==1) & (y_true==0))
    n  = len(y_true)
    w = threshold/(1-threshold)
    return (tp/n) - (fp/n)*w

def decision_curve(y_true, y_prob, thresholds=None, title="Decision Curve", out_path=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y_true = np.asarray(y_true).astype(int)
    event_rate = float(np.mean(y_true))

    nb_model = [net_benefit(y_true, y_prob, t) for t in thresholds]
    nb_all   = [event_rate - (1-event_rate)*t/(1-t) for t in thresholds]
    nb_none  = [0.0]*len(thresholds)

    plt.figure(figsize=(6,4.5))
    plt.plot(thresholds, nb_model, label='Model')
    plt.plot(thresholds, nb_all, linestyle='--', label='Treat all')
    plt.plot(thresholds, nb_none, linestyle='--', label='Treat none')
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    return {"thresholds": thresholds.tolist(), "nb_model": nb_model, "nb_all": nb_all, "nb_none": nb_none}
