# backend/aki/export_results.py

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from backend.paths import OUTPUT_DIR_AKI


def save_json(data: Dict, filename: str) -> Path:
    """
    将 dict 保存为 JSON 文件到 outputs/aki/
    """
    path = OUTPUT_DIR_AKI / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON saved to {path}")
    return path


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """
    将 DataFrame 保存为 CSV 文件到 outputs/aki/
    """
    path = OUTPUT_DIR_AKI / filename
    df.to_csv(path, index=False)
    print(f"[OK] CSV saved to {path}")
    return path
