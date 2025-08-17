import argparse, pandas as pd, numpy as np, yaml
from pathlib import Path
from utils import load_config, ensure_dirs

# KDIGO AKI within 7 days:
# - SCr criteria: increase >= 0.3 mg/dL within 48h OR >= 1.5x baseline within 7 days.
# - Urine criteria: < 0.5 mL/kg/h for >= 6h (rolling windows).
# Baseline SCr rule is set via cfg['outcomes']['kdigo']['baseline_method'].

def select_baseline_scr(scr_timeseries, intime, method="first_24h"):
    win_24h = (scr_timeseries["charttime"] >= intime) & (scr_timeseries["charttime"] <= intime + np.timedelta64(24,'h'))
    if method == "first_24h":
        df = scr_timeseries.loc[win_24h].sort_values("charttime")
        return df["valuenum"].iloc[0] if len(df) else np.nan
    elif method == "lowest_24h":
        df = scr_timeseries.loc[win_24h]
        return df["valuenum"].min() if len(df) else np.nan
    elif method == "pre48h_to_post6h_min":
        df = scr_timeseries[(scr_timeseries["charttime"] >= intime - np.timedelta64(48,'h')) &
                            (scr_timeseries["charttime"] <= intime + np.timedelta64(6,'h'))]
        return df["valuenum"].min() if len(df) else np.nan
    else:
        return np.nan

def kdigo_scr_flag(scr_timeseries, baseline, intime):
    win_48h = scr_timeseries[(scr_timeseries["charttime"] >= intime) & (scr_timeseries["charttime"] <= intime + np.timedelta64(48,'h'))]
    within48 = False
    if len(win_48h):
        earliest = win_48h.sort_values("charttime").iloc[0]["valuenum"]
        max48 = win_48h["valuenum"].max()
        within48 = (max48 - float(earliest)) >= 0.3
    win_7d = scr_timeseries[(scr_timeseries["charttime"] >= intime) & (scr_timeseries["charttime"] <= intime + np.timedelta64(168,'h'))]
    ratio7d = False
    if len(win_7d) and np.isfinite(baseline):
        ratio7d = (win_7d["valuenum"].max() / float(baseline)) >= 1.5
    return bool(within48 or ratio7d)

def kdigo_urine_flag(urine_ts, weight_kg, intime, mlkg_thr=0.5, low_hours=6):
    if not np.isfinite(weight_kg) or weight_kg <= 0 or len(urine_ts)==0:
        return False
    start = intime; end = intime + np.timedelta64(48,'h')
    df = urine_ts[(urine_ts["charttime"] >= start) & (urine_ts["charttime"] <= end)].copy()
    if len(df)==0:
        return False
    df["hour"] = ((df["charttime"] - start) / np.timedelta64(1,'h')).astype(int)
    hourly = df.groupby("hour", as_index=False)["valuenum"].sum()
    hourly = hourly.set_index("hour").reindex(range(0, 48), fill_value=0.0)["valuenum"].values
    ml_per_kg_per_h = hourly / float(weight_kg)
    below = (ml_per_kg_per_h < mlkg_thr).astype(int)
    run = 0
    for v in below:
        run = run + 1 if v==1 else 0
        if run >= low_hours:
            return True
    return False

def main(cfg_path):
    cfg = load_config(cfg_path); ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["out_dir"])
    csv_dir = Path(cfg["paths"]["mimic_csv_dir"])

    cohort = pd.read_parquet(out_dir / "cohort_24h_firstmeasures.parquet")
    import duckdb
    con = duckdb.connect()
    def csv(path): return f"read_csv_auto('{path}', IGNORE_ERRORS=true)"
    labevents = con.sql(csv(f"{csv_dir}/hosp/labevents.csv")).df()
    outputevents = con.sql(csv(f"{csv_dir}/icu/outputevents.csv")).df()

    ids = yaml.safe_load(open("configs/mimic_ids.yaml","r",encoding="utf-8"))
    scr_ids = ids["items"].get("scr_itemids",[])
    urine_ids = ids["items"].get("urine_output_itemids",[])

    labevents = labevents[labevents["itemid"].isin(scr_ids)]
    outputevents = outputevents[outputevents["itemid"].isin(urine_ids)]

    for c in ["charttime","storetime","starttime","endtime"]:
        if c in labevents.columns: labevents[c] = pd.to_datetime(labevents[c], errors="coerce")
        if c in outputevents.columns: outputevents[c] = pd.to_datetime(outputevents[c], errors="coerce")

    labels = []
    kd = cfg["outcomes"]["kdigo"]
    for row in cohort.itertuples(index=False):
        stay_id = row.stay_id; hadm_id = row.hadm_id; intime = pd.to_datetime(row.intime)
        weight = getattr(row, "weight") if hasattr(row, "weight") else np.nan
        scr_ts = labevents[labevents["hadm_id"]==hadm_id][["charttime","valuenum"]].dropna()
        urine_ts = outputevents[outputevents["stay_id"]==stay_id][["charttime","valuenum"]].dropna()

        baseline = select_baseline_scr(scr_ts, intime, method=kd.get("baseline_method","first_24h"))
        scr_f = kdigo_scr_flag(scr_ts, baseline, intime)
        urine_f = kdigo_urine_flag(urine_ts, weight, intime,
                                   mlkg_thr=kd["urine_ml_per_kg_per_h_threshold"],
                                   low_hours=kd["urine_low_duration_hours"])
        aki = int(scr_f or urine_f)
        labels.append(dict(stay_id=stay_id, hadm_id=hadm_id, aki_7d=aki))

    lab = pd.DataFrame(labels)
    merged = cohort.merge(lab, on=["hadm_id","stay_id"], how="left")
    merged.to_parquet(out_dir / "cohort_with_labels.parquet", index=False)
    print("[OK] Labels written:", out_dir / "cohort_with_labels.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
