import argparse, yaml, os, json
from .extract import build_cohort
from .labels_kdigo import make_labels
from .features import extract_features
from .train import train_eval
from .eval import evaluate_and_plot

def main(args):
    cfg = yaml.safe_load(open(args.config, "r"))
    # Ensure dirs
    for k in ["raw","interim","processed","models","plots","reports"]:
        os.makedirs(cfg["paths"][k], exist_ok=True)

    print("Stage 1: Build cohort...")
    cohort = build_cohort(cfg)
    print(f"  cohort: n={len(cohort)}")

    print("Stage 2: Make KDIGO labels (AKI@7d)...")
    labels = make_labels(cfg)
    print(f"  labels: n={len(labels)}  events={labels['aki7d'].sum()} ({labels['aki7d'].mean():.3f})")

    print("Stage 3: Extract features (0–24h)...")
    feats = extract_features(cfg)
    print(f"  features: n={len(feats)}  cols={feats.shape[1]}")

    print("Stage 4: Train + Evaluate...")
    Xtr, ytr, p_tr, z_tr, Xte, yte, p_te, z_te, cv_auc = train_eval(cfg)
    print(f"  CV AUC (train): {cv_auc:.3f}")

    print("Stage 5: Plots + Metrics...")
    m_tr = evaluate_and_plot(cfg, ytr, p_tr, z_tr, split_tag="train")
    m_te = evaluate_and_plot(cfg, yte, p_te, z_te, split_tag="val")

    # Save overall metrics
    all_metrics = {"cv_auc_train": float(cv_auc), "train": m_tr, "val": m_te}
    with open(os.path.join(cfg["paths"]["reports"], "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("Done. See outputs/plots and outputs/reports.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config.yaml")
    args = ap.parse_args()
    main(args)
