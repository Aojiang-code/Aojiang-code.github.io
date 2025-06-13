"""
03_build_dataset.py
功能：加载中间结构，构造 SampleBaseDataset 数据集
"""

import pickle
from pyhealth.datasets import SampleBaseDataset

with open("./data/processed/patients.pkl", "rb") as f:
    patients = pickle.load(f)

dataset = SampleBaseDataset(
    root="./data/processed/",
    samples=patients,
    dataset_name="AKI_LAB"
)

dataset.stat()
