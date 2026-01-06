# backend/aki/data_loader.py

from typing import Optional
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from backend.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_engine(echo: bool = False) -> Engine:
    """
    创建 SQLAlchemy Engine，用于连接 PostgreSQL.
    """
    url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url, echo=echo)
    return engine


def load_aki_cohort(engine: Engine,
                    limit: Optional[int] = None) -> pd.DataFrame:
    """
    读取 aki_cohort 视图（步骤2中创建）.
    可选 limit 用于调试时只取一部分数据.
    """
    sql = "SELECT * FROM aki_cohort"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    df = pd.read_sql(sql, engine)
    return df


def load_aki_labs_firstday(engine: Engine) -> pd.DataFrame:
    """
    读取 aki_labs_firstday 视图（步骤2中创建）.
    """
    sql = "SELECT * FROM aki_labs_firstday"
    df = pd.read_sql(sql, engine)
    return df


def load_aki_full_dataset(engine: Engine,
                          limit: Optional[int] = None) -> pd.DataFrame:
    """
    合并 aki_cohort 和 aki_labs_firstday，
    返回一个包含人口学、ICU信息、结局和首日实验室指标的 DataFrame.
    """
    cohort = load_aki_cohort(engine, limit=limit)
    labs = load_aki_labs_firstday(engine)

    # 根据 stay_id 合并（也可以用 subject_id + hadm_id + stay_id）
    merged = cohort.merge(
        labs,
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
        suffixes=("", "_lab")
    )

    return merged


if __name__ == "__main__":
    # 简单自测：打印样本量和几个字段
    engine = get_engine()
    df = load_aki_full_dataset(engine, limit=1000)
    print(df.head())
    print("Sample size (subset):", len(df))
