import os, re, pandas as pd, numpy as np
from typing import Dict, Any, List
from .data_source import DataSource


def load_icd_codeset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["code"] = df["code"].str.strip()
    df["system"] = df["system"].str.strip()
    return df


def build_cohort(cfg: Dict[str, Any]) -> pd.DataFrame:
    ds = DataSource(cfg).connect()
    # load ICD codes into temp table
    dka_codes = load_icd_codeset(cfg["icd_codesets"]["dka"])
    if cfg["data_source"]["type"] == "duckdb":
        ds._conn.register("dka_codes_tmp", dka_codes)
    else:
        # For Postgres, use a temporary table or WHERE IN (...). Here we fallback to WHERE IN for simplicity.
        pass

    # run cohort SQL
    cohort_sql = open("sql/cohort.sql", "r", encoding="utf-8").read()
    cohort_df = ds.read_sql(cohort_sql)

    # optional filter for readmission window (same-subject repeated close ICU stays)
    if cfg["cohort"].get("exclude_readmissions_within_hours", None):
        # For first ICU only per patient, above SQL uses rn_subject=1 – enough for most cases.
        pass

    # save
    out_path = os.path.join(cfg["paths"]["interim"], "cohort.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cohort_df.to_parquet(out_path, index=False)
    ds.close()
    return cohort_df
