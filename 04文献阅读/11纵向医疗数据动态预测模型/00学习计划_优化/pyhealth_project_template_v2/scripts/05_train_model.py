"""
05_train_model.py
功能：训练模型并保存
"""

from pyhealth.models import RETAIN
from pyhealth.trainer import Trainer
from pyhealth.datasets import split_by_patient, get_dataloader
import pickle

# 假设已构建 processed_dataset
with open("./data/processed/processed_dataset.pkl", "rb") as f:
    processed_dataset = pickle.load(f)

# 划分数据
train_ds, val_ds, test_ds = split_by_patient(processed_dataset, [0.7, 0.15, 0.15])
train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=64)

# 模型训练
model = RETAIN(dataset=processed_dataset)
trainer = Trainer(model=model, metrics=["auc", "accuracy", "f1"])
trainer.train(train_loader, val_loader, epochs=50, monitor="auc", patience=5)
trainer.save_model("./checkpoints/retain_best.pth")
