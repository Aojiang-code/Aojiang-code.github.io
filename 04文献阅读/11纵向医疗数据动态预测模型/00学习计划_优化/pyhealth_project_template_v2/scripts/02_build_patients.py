"""
02_build_patients.py
功能：从 LABEVENTS 中提取肌酐事件，构造 Patient 与 Visit
"""

from pyhealth.data import Patient, Visit, Event
import pickle

# 示例逻辑伪代码，需配合实际 dataset 对象
patients = []

# 遍历示例（待补充真实 dataset）
# for p in dataset.patients:
#     patient = Patient(patient_id=p.patient_id, data_source="mimic")
#     ...
#     patients.append(patient)

# 保存中间结构
with open("./data/processed/patients.pkl", "wb") as f:
    pickle.dump(patients, f)
