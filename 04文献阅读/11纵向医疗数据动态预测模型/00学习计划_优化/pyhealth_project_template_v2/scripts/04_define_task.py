"""
04_define_task.py
功能：设定 PyHealth 二分类任务（如 AKI 预测）
"""

from pyhealth.tasks import BinaryPredictionTask
from pyhealth.datasets import SampleBaseDataset
import pickle

# 加载 Dataset
with open("./data/processed/patients.pkl", "rb") as f:
    patients = pickle.load(f)

dataset = SampleBaseDataset(root="./data/processed/", samples=patients, dataset_name="AKI_LAB")

# 设置任务
task = BinaryPredictionTask(
    dataset=dataset,
    event_type="lab",
    label_key="aki",
    time_window=7,
    use_time=True,
    use_visit=True
)

processed_dataset = dataset.set_task(task)
processed_dataset.stat()
