"""
08_compare_models.py
功能：统一对比多个模型表现
"""

from pyhealth.models import GRUD, RETAIN, Transformer
from pyhealth.trainer import Trainer
from pyhealth.datasets import split_by_patient, get_dataloader
import pandas as pd, pickle

with open("./data/processed/processed_dataset.pkl", "rb") as f:
    processed_dataset = pickle.load(f)

train_ds, val_ds, test_ds = split_by_patient(processed_dataset, [0.7, 0.15, 0.15])
train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=64)
test_loader = get_dataloader(test_ds, batch_size=64)

def run_model(model_class, name):
    model = model_class(dataset=processed_dataset)
    trainer = Trainer(model=model, metrics=["auc", "f1", "accuracy"])
    trainer.train(train_loader, val_loader, epochs=50, monitor="auc", patience=5)
    result = trainer.evaluate(test_loader)
    result["model"] = name
    return result

results = [
    run_model(GRUD, "GRU-D"),
    run_model(RETAIN, "RETAIN"),
    run_model(Transformer, "Transformer")
]

df = pd.DataFrame(results)
df.to_csv("./data/output/model_comparison.csv", index=False)
print(df)
