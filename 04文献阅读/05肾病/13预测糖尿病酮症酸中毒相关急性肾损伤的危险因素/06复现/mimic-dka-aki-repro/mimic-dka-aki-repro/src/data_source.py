import os
import pandas as pd
from typing import Optional, Dict, Any
from sqlalchemy import create_engine
import duckdb
import glob


class DataSource:
    """
    Unified access layer for MIMIC-IV via DuckDB over CSV/Parquet OR Postgres.
    For DuckDB mode, we map common table names to local files under paths.mimic_root.
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.ds_type = config["data_source"]["type"]
        if self.ds_type not in ("duckdb", "postgres"):
            raise ValueError("data_source.type must be 'duckdb' or 'postgres'")
        self._conn = None

    def connect(self):
        if self.ds_type == "duckdb":
            self._conn = duckdb.connect(database=":memory:")
            # register CSV/Parquet folders as tables
            self._map_duckdb_tables(self.cfg["paths"]["mimic_root"])
        else:
            pg = self.cfg["data_source"]["postgres"]
            uri = f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}"
            self._engine = create_engine(uri, pool_pre_ping=True)
        return self

    def _map_duckdb_tables(self, root: str):
        con = self._conn
        # Try to detect files; supports csv/csv.gz/parquet under hosp/icu schemas
        mapping = {
            "hosp.patients": ["hosp/patients.*"],
            "hosp.admissions": ["hosp/admissions.*"],
            "hosp.diagnoses_icd": ["hosp/diagnoses_icd.*"],
            "hosp.d_labitems": ["hosp/d_labitems.*"],
            "hosp.labevents": ["hosp/labevents.*"],
            "icu.icustays": ["icu/icustays.*"],
            "icu.chartevents": ["icu/chartevents.*"],
            "icu.d_items": ["icu/d_items.*"],
            "icu.inputevents": ["icu/inputevents.*", "icu/inputevents_cv.*", "icu/inputevents_mv.*"],
            "icu.outputevents": ["icu/outputevents.*"]
        }
        for table, patterns in mapping.items():
            for pat in patterns:
                files = glob.glob(os.path.join(root, pat))
                if files:
                    # Use DuckDB's file globbing & automatic format detection
                    path_glob = os.path.join(root, pat.replace("\\", "/"))
                    # Create a view for this table (DuckDB can read CSV/Parquet transparently)
                    con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path_glob}')")
                    # If parquet is not found, fallback to CSV/CSV.GZ
                    try:
                        con.execute(f"SELECT 1 FROM {table} LIMIT 1")
                    except Exception:
                        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_csv_auto('{path_glob}')")
                    break

    def read_sql(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        if self.ds_type == "duckdb":
            return self._conn.execute(sql, params or {}).df()
        else:
            return pd.read_sql(sql, self._engine, params=params)

    def execute(self, sql: str, params: Optional[dict] = None):
        if self.ds_type == "duckdb":
            return self._conn.execute(sql, params or {})
        else:
            with self._engine.begin() as conn:
                conn.execute(sql, params or {})

    def close(self):
        if self.ds_type == "duckdb" and self._conn is not None:
            self._conn.close()
