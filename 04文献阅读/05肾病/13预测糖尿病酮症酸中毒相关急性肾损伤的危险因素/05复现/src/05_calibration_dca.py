import argparse
from utils import load_config, ensure_dirs

def main(cfg_path):
    cfg = load_config(cfg_path); ensure_dirs(cfg)
    print("[Info] Calibration & DCA arrays are produced in 04_train_eval.py and saved under work/.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
