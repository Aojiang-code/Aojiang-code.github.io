"""
01_load_raw.py
功能：加载 MIMIC 或远程合成数据，检查表结构
"""

from pyhealth.datasets import MIMIC3Dataset

# 可配置为本地路径或远程
mimic_path = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/"

dataset = MIMIC3Dataset(
    root=mimic_path,
    tables=["LABEVENTS"]
)

print(f"加载成功，共有 {len(dataset.patients)} 位患者")
