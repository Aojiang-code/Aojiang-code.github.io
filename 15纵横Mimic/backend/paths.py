# backend/paths.py

from pathlib import Path

# 项目根目录：假设 backend/ 与 docs/ 在同一层
BASE_DIR = Path(__file__).resolve().parent.parent

# 输出目录：outputs/aki/
OUTPUT_DIR_AKI = BASE_DIR / "outputs" / "aki"
OUTPUT_DIR_AKI.mkdir(parents=True, exist_ok=True)
