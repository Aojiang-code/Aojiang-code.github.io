import os, pandas as pd, numpy as np
from typing import Dict, Any
from .data_source import DataSource


def _first_creatinine_within(df_labs, subject_id, hadm_id, start_ts, end_ts):
    sub = df_labs[(df_labs.subject_id==subject_id) & (df_labs.hadm_id==hadm_id)]
    sub = sub[(sub.charttime >= start_ts) & (sub.charttime <= end_ts)]
    sub = sub.sort_values("charttime", ascending=True)
    if len(sub)==0:
        return None
    return sub.iloc[0]["valuenum"]


def make_labels(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    KDIGO AKI within horizon_days after ICU admission.
    - Scr criteria: +0.3 mg/dL within 48h OR 1.5x baseline within 7d
    - Urine criteria: <0.5 mL/kg/h sustained ≥6h (we implement 6/12/24h windows and take union)
    Baseline Scr: nearest pre-ICU within admission; fallback to first ICU Scr if none.
    """
    ds = DataSource(cfg).connect()
    cohort = pd.read_parquet(os.path.join(cfg["paths"]["interim"], "cohort.parquet"))
    horizon_days = cfg["prediction"]["horizon_days"]
    obs_hours = cfg["prediction"]["observation_window_hours"]

    # load creatinine & urine
    # Find creatinine itemids from hosp.d_labitems by pattern
    q_cre = """
    WITH ids AS (
      SELECT itemid FROM hosp.d_labitems
      WHERE lower(label) LIKE '%creatinine%'
    )
    SELECT le.subject_id, le.hadm_id, le.charttime, le.valuenum
    FROM hosp.labevents le
    JOIN ids ON le.itemid = ids.itemid
    WHERE le.valuenum IS NOT NULL
    """
    cr = ds.read_sql(q_cre)
    cr["charttime"] = pd.to_datetime(cr["charttime"])

    q_out = """
    SELECT subject_id, hadm_id, charttime, value as volume
    FROM icu.outputevents
    WHERE value IS NOT NULL
    """
    ur = ds.read_sql(q_out)
    ur["charttime"] = pd.to_datetime(ur["charttime"])

    # weight for urine scaling
    # Try to get a nearest weight around ICU intime (kg). Simplify: from chartevents by "weight"
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

    rows = []
    for _, r in cohort.iterrows():
        sid, hid, stid = int(r.subject_id), int(r.hadm_id), int(r.stay_id)
        t0 = pd.to_datetime(r.icu_intime)
        obs_end = t0 + pd.Timedelta(hours=obs_hours)
        horizon_end = t0 + pd.Timedelta(days=horizon_days)

        # Baseline Scr = nearest pre-ICU within admission; fallback to first ICU Scr
        cr_h = cr[(cr.subject_id==sid) & (cr.hadm_id==hid)].copy()
        cr_h = cr_h.sort_values("charttime")
        base = None
        pre = cr_h[cr_h.charttime < t0]
        if len(pre)>0:
            base = pre.iloc[-1].valuenum
        else:
            first_icu = cr_h[(cr_h.charttime >= t0) & (cr_h.charttime <= obs_end)]
            if len(first_icu)>0:
                base = first_icu.iloc[0].valuenum
        # Scr criteria within horizon
        scr_window = cr_h[(cr_h.charttime >= t0) & (cr_h.charttime <= horizon_end)]
        scr_48h = cr_h[(cr_h.charttime >= t0) & (cr_h.charttime <= t0+pd.Timedelta(hours=48))]

        aki_scr = False
        if base is not None and len(scr_window)>0:
            # 0.3 within 48h
            if len(scr_48h)>0:
                if (scr_48h.valuenum - base).max() >= 0.3:
                    aki_scr = True
            # 1.5x within 7d
            if not aki_scr:
                if (scr_window.valuenum / base).max() >= 1.5:
                    aki_scr = True

        # Urine criteria over sustained low output
        ur_h = ur[(ur.subject_id==sid) & (ur.hadm_id==hid)].copy()
        ur_h = ur_h[(ur_h.charttime >= t0) & (ur_h.charttime <= horizon_end)]
        # Approximate weight: nearest around ICU intime within +/- 24h
        wt_h = wt[(wt.subject_id==sid) & (wt.hadm_id==hid)].copy()
        wt_h["dt"] = (wt_h.charttime - t0).abs()
        weight = np.nan
        if len(wt_h)>0:
            weight = wt_h.sort_values("dt").iloc[0].weight
        # Aggregate urine into rolling 6h buckets
        aki_urine = False
        if len(ur_h)>0 and pd.notna(weight) and weight>0:
            ur_h = ur_h.sort_values("charttime")
            ur_h["hour"] = ur_h["charttime"].dt.floor("h")
            hourly = ur_h.groupby("hour", as_index=False)["volume"].sum()
            # 6-hour windows
            hourly = hourly.set_index("hour")
            hourly = hourly.asfreq("h").fillna(0.0)
            roll6 = hourly["volume"].rolling(6, min_periods=6).sum()
            rate6 = roll6 / (6 * weight)  # mL/kg/h
            if (rate6 < 0.5).any():
                aki_urine = True
        aki = int(aki_scr or aki_urine)
        rows.append({"subject_id": sid, "hadm_id": hid, "stay_id": stid, "aki7d": aki})

    labels = pd.DataFrame(rows)
    out_path = os.path.join(cfg["paths"]["processed"], "labels.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels.to_parquet(out_path, index=False)
    ds.close()
    return labels
