"""
06_evaluate_model.py
功能：加载模型并进行测试评估
"""

from pyhealth.trainer import Trainer
from pyhealth.models import RETAIN
import pickle

# 加载数据与模型
with open("./data/processed/processed_dataset.pkl", "rb") as f:
    processed_dataset = pickle.load(f)

model = RETAIN(dataset=processed_dataset)
trainer = Trainer(model=model, metrics=["auc", "accuracy", "f1"])
trainer.load_model("./checkpoints/retain_best.pth")

_, _, test_ds = split_by_patient(processed_dataset, [0.7, 0.15, 0.15])
test_loader = get_dataloader(test_ds, batch_size=64)
results = trainer.evaluate(test_loader)
print(results)
