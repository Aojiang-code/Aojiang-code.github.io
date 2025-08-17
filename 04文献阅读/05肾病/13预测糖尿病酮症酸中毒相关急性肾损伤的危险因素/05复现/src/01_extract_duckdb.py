import argparse, duckdb, pandas as pd, numpy as np
from pathlib import Path
import yaml, sys, os
from utils import load_config, ensure_dirs

def main(cfg_path):
    cfg = load_config(cfg_path); ensure_dirs(cfg)
    csv_dir = cfg["paths"]["mimic_csv_dir"]
    out_dir = Path(cfg["paths"]["out_dir"])

    con = duckdb.connect(cfg["paths"]["duckdb_path"])
    con.execute("PRAGMA threads=4;")

    def reg(name, rel):
        con.register(name, rel)

    def csv(path):
        return f"read_csv_auto('{path}', IGNORE_ERRORS=true)"

    base = csv_dir
    reg("diagnoses_icd", con.sql(csv(f"{base}/hosp/diagnoses_icd.csv")))
    reg("icustays", con.sql(csv(f"{base}/icu/icustays.csv")))
    reg("labevents", con.sql(csv(f"{base}/hosp/labevents.csv")))
    reg("chartevents", con.sql(csv(f"{base}/icu/chartevents.csv")))
    reg("outputevents", con.sql(csv(f"{base}/icu/outputevents.csv")))
    reg("inputevents", con.sql(csv(f"{base}/icu/inputevents.csv")))
    reg("patients", con.sql(csv(f"{base}/core/patients.csv")))
    reg("admissions", con.sql(csv(f"{base}/core/admissions.csv")))

    ids = yaml.safe_load(open("configs/mimic_ids.yaml","r",encoding="utf-8"))

    dka_codes = (ids.get("diagnoses_icd",{}).get("dka_icd9",[]) +
                 ids.get("diagnoses_icd",{}).get("dka_icd10",[]))
    if not dka_codes or "TO_FILL" in dka_codes:
        print("[WARN] Please fill DKA ICD codes in configs/mimic_ids.yaml", file=sys.stderr)

    lab_bun = ids["items"].get("bun_itemids",[])
    lab_scr = ids["items"].get("scr_itemids",[])
    lab_glu = ids["items"].get("glucose_itemids",[])
    plt_ids = ids["items"].get("plt_itemids",[])
    hr_ids = ids["items"].get("hr_itemids",[])
    sbp_ids = ids["items"].get("sbp_itemids",[])
    dbp_ids = ids["items"].get("dbp_itemids",[])
    rr_ids  = ids["items"].get("rr_itemids",[])
    temp_ids= ids["items"].get("temp_itemids",[])
    wt_ids  = ids["items"].get("weight_itemids",[])
    urine_ids = ids["items"].get("urine_output_itemids",[])
    fluid_ids = ids["items"].get("fluid_input_itemids",[])

    def to_rel(name, arr, col="itemid"):
        if not arr: arr = [-1]
        df = pd.DataFrame({col: arr})
        con.register(name, df)

    to_rel("dka_icd_codes", dka_codes, col="code")
    to_rel("lab_bun_itemids", lab_bun)
    to_rel("lab_scr_itemids", lab_scr)
    to_rel("lab_glucose_itemids", lab_glu)
    to_rel("plt_itemids", plt_ids)
    to_rel("hr_itemids", hr_ids)
    to_rel("sbp_itemids", sbp_ids)
    to_rel("dbp_itemids", dbp_ids)
    to_rel("rr_itemids", rr_ids)
    to_rel("temp_itemids", temp_ids)
    to_rel("weight_itemids", wt_ids)
    to_rel("urine_output_itemids", urine_ids)
    to_rel("fluid_input_itemids", fluid_ids)

    sql_path = Path("src/01_extract_duckdb.sql")
    sql = sql_path.read_text(encoding="utf-8")
    df = con.sql(sql).df()
    out = out_dir / "cohort_24h_firstmeasures.parquet"
    df.to_parquet(out, index=False)
    print(f"[OK] Wrote {out} with shape={df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
