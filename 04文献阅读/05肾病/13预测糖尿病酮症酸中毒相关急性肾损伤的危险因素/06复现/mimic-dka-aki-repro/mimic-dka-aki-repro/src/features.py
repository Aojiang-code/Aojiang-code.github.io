import os, pandas as pd, numpy as np
from typing import Dict, Any
from .data_source import DataSource


def extract_features(cfg: Dict[str, Any]) -> pd.DataFrame:
    ds = DataSource(cfg).connect()
    cohort = pd.read_parquet(os.path.join(cfg["paths"]["interim"], "cohort.parquet"))
    obs_hours = cfg["prediction"]["observation_window_hours"]

    # Helper: labs first value in 24h
    def first_lab_in_24h(label_pattern: str):
        q = f"""
        WITH ids AS (
          SELECT itemid FROM hosp.d_labitems
          WHERE lower(label) ~ '{label_pattern.lower()}'
        )
        SELECT le.subject_id, le.hadm_id, le.charttime, le.valuenum, le.itemid
        FROM hosp.labevents le
        JOIN ids ON le.itemid = ids.itemid
        WHERE le.valuenum IS NOT NULL
        """
        return ds.read_sql(q)

    # collect lab frames
    bun = first_lab_in_24h("urea|bun|blood urea")
    cre = first_lab_in_24h("creatinine")
    glu = first_lab_in_24h("glucose")

    for df in (bun, cre, glu):
        df["charttime"] = pd.to_datetime(df["charttime"])

    # Platelet
    pltab = first_lab_in_24h("platelet|plt")
    pltab["charttime"] = pd.to_datetime(pltab["charttime"])

    # Inputs (fluids) total in 24h
    q_in = """
    SELECT subject_id, hadm_id, charttime, amount AS volume
    FROM icu.inputevents
    WHERE amount IS NOT NULL
    """
    ins = ds.read_sql(q_in)
    ins["charttime"] = pd.to_datetime(ins["charttime"])

    # Urine 24h sum
    q_out = """
    SELECT subject_id, hadm_id, charttime, value AS volume
    FROM icu.outputevents
    WHERE value IS NOT NULL
    """
    outs = ds.read_sql(q_out)
    outs["charttime"] = pd.to_datetime(outs["charttime"])

    # Weight (nearest around ICU intime)
    q_wt = """
    WITH ids AS (
      SELECT itemid FROM icu.d_items WHERE lower(label) LIKE '%weight%'
    )
    SELECT ce.subject_id, ce.hadm_id, ce.stay_id, ce.charttime, ce.valuenum AS weight
    FROM icu.chartevents ce
    JOIN ids ON ce.itemid = ids.itemid
    WHERE ce.valuenum IS NOT NULL
    """
    wt = ds.read_sql(q_wt)
    wt["charttime"] = pd.to_datetime(wt["charttime"])

    # Build feature rows per cohort
    rows = []
    for _, r in cohort.iterrows():
        sid, hid, stid = int(r.subject_id), int(r.hadm_id), int(r.stay_id)
        t0 = pd.to_datetime(r.icu_intime)
        t1 = t0 + pd.Timedelta(hours=obs_hours)

        # age, admission_type
        age = float(r.age)
        admtype = str(r.admission_type)

        def first_in(df, col="valuenum"):
            sub = df[(df.subject_id==sid) & (df.hadm_id==hid)]
            sub = sub[(sub.charttime >= t0) & (sub.charttime <= t1)]
            if len(sub)==0: return np.nan
            return float(sub.sort_values("charttime", ascending=True).iloc[0][col])

        BUN = first_in(bun)
        Creatinine = first_in(cre)
        Glucose = first_in(glu)
        Platelet = first_in(pltab)

        # Input/Urine sums
        def sum_in(df, vol_col="volume"):
            sub = df[(df.subject_id==sid) & (df.hadm_id==hid)]
            sub = sub[(sub.charttime >= t0) & (sub.charttime <= t1)]
            if len(sub)==0: return np.nan
            return float(sub[vol_col].sum())

        Input24h = sum_in(ins)
        Urine24h = sum_in(outs)

        # Weight nearest
        wt_h = wt[(wt.subject_id==sid) & (wt.hadm_id==hid)].copy()
        wt_h["dt"] = (wt_h.charttime - t0).abs()
        Weight = float(wt_h.sort_values("dt").iloc[0].weight) if len(wt_h)>0 else np.nan

        rows.append({
            "subject_id": sid, "hadm_id": hid, "stay_id": stid,
            "age": age, "admission_type": admtype,
            "bun": BUN, "creatinine": Creatinine, "glucose": Glucose,
            "platelet": Platelet, "input_24h": Input24h, "urine_24h": Urine24h,
            "weight": Weight
        })

    feats = pd.DataFrame(rows)

    # Missingness handling
    miss_thr = cfg["features"]["missing_threshold_drop"]
    # drop columns with >threshold missing
    col_miss = feats.isna().mean()
    drop_cols = list(col_miss[col_miss>miss_thr].index)
    keep = [c for c in feats.columns if c not in drop_cols]
    feats = feats[keep]

    # save
    out_path = os.path.join(cfg["paths"]["processed"], "features.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feats.to_parquet(out_path, index=False)
    ds.close()
    return feats
