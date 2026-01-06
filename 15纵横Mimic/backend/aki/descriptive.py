# backend/aki/descriptive.py

from typing import Dict, List
import numpy as np
import pandas as pd


def compute_basic_stats(df: pd.DataFrame) -> Dict:
    """
    计算一些整体性的描述指标：
    - 样本量
    - 年龄均值/中位数
    - ICU/住院死亡率
    - ICU/住院住院时间中位数
    """
    n = len(df)

    basic = {
        "n": int(n),
        "age_mean": float(df["age"].mean(skipna=True)),
        "age_median": float(df["age"].median(skipna=True)),
        "icu_mortality_rate": float(df["icu_mortality"].mean(skipna=True)),
        "hosp_mortality_rate": float(df["hosp_mortality"].mean(skipna=True)),
        "icu_los_median": float(df["icu_los_days"].median(skipna=True)),
        "hosp_los_median": float(df["hosp_los_days"].median(skipna=True)),
    }
    return basic


def age_group_distribution(df: pd.DataFrame,
                           bins: List[int] = None) -> Dict:
    """
    按年龄分组，生成直方图/条形图可用的数据结构.
    返回：
    {
      "age_groups": [...],
      "counts": [...],
      "proportions": [...],
      "n": int
    }
    """
    if bins is None:
        bins = [18, 30, 40, 50, 60, 70, 80, 120]

    labels = [
        "18–29",
        "30–39",
        "40–49",
        "50–59",
        "60–69",
        "70–79",
        "80+",
    ]

    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    counts = df["age_group"].value_counts().sort_index()
    n = int(counts.sum())
    proportions = (counts / n).fillna(0.0)

    return {
        "age_groups": labels,
        "counts": [int(counts.get(label, 0)) for label in labels],
        "proportions": [float(proportions.get(label, 0.0)) for label in labels],
        "n": n,
    }


def categorical_distribution(df: pd.DataFrame,
                             column: str,
                             categories_order: List[str] = None,
                             display_map: Dict[str, str] = None) -> Dict:
    """
    通用的分类变量分布，用于性别、保险、种族等.

    返回：
    {
      "categories": [...],
      "display_categories": [...],
      "counts": [...],
      "proportions": [...],
      "n": int
    }
    """
    series = df[column].astype("category")

    if categories_order:
        series = series.cat.set_categories(categories_order)

    counts = series.value_counts().sort_index()
    n = int(counts.sum())
    proportions = (counts / n).fillna(0.0)

    # 显示名称映射（用于前端显示中文）
    categories = list(counts.index.astype(str))
    if display_map:
        display_categories = [display_map.get(c, c) for c in categories]
    else:
        display_categories = categories

    return {
        "categories": categories,
        "display_categories": display_categories,
        "counts": [int(c) for c in counts],
        "proportions": [float(p) for p in proportions],
        "n": n,
    }


def mortality_distribution(df: pd.DataFrame) -> Dict:
    """
    生成 ICU 和住院死亡率的简单结构：
    {
      "outcomes": ["ICU mortality", "Hospital mortality"],
      "rates": [0.xx, 0.xx]
    }
    """
    icu_rate = float(df["icu_mortality"].mean(skipna=True))
    hosp_rate = float(df["hosp_mortality"].mean(skipna=True))

    return {
        "outcomes": ["icu_mortality", "hosp_mortality"],
        "display_outcomes": ["ICU死亡率", "住院死亡率"],
        "rates": [icu_rate, hosp_rate],
    }


def generate_table1(df: pd.DataFrame,
                    numeric_vars: List[str],
                    categorical_vars: List[str]) -> pd.DataFrame:
    """
    简单版 Table 1：
    - 数值变量：给出 n, mean, std, median, Q1, Q3
    - 分类变量：给出每个类别的 count 与 proportion
    返回一个“长表”（长格式）的 DataFrame，方便写 csv / 用在网页里。
    """
    rows = []

    # 数值变量
    for var in numeric_vars:
        series = df[var]
        clean = series.dropna()
        if clean.empty:
            continue

        rows.append({
            "variable": var,
            "level": "",
            "type": "numeric",
            "n": int(clean.count()),
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "median": float(clean.median()),
            "q1": float(clean.quantile(0.25)),
            "q3": float(clean.quantile(0.75)),
            "count": np.nan,
            "proportion": np.nan,
        })

    # 分类变量
    for var in categorical_vars:
        series = df[var].astype("category")
        counts = series.value_counts(dropna=False)
        n = int(counts.sum())
        proportions = counts / n

        for level, count in counts.items():
            rows.append({
                "variable": var,
                "level": str(level),
                "type": "categorical",
                "n": n,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "count": int(count),
                "proportion": float(proportions[level]),
            })

    table1 = pd.DataFrame(rows)
    return table1
