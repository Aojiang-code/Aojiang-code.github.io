"""
07_attention_analysis.py
功能：RETAIN 注意力机制可视化
"""

import matplotlib.pyplot as plt
from pyhealth.models import RETAIN
import pickle

with open("./data/processed/processed_dataset.pkl", "rb") as f:
    processed_dataset = pickle.load(f)

model = RETAIN(dataset=processed_dataset)
model.load("./checkpoints/retain_best.pth")

sample = processed_dataset.samples[0]
pred, visit_attn, var_attn = model.forward([sample], return_attention=True)

plt.plot(visit_attn[0])
plt.title("Visit-level Attention (α)")
plt.show()
