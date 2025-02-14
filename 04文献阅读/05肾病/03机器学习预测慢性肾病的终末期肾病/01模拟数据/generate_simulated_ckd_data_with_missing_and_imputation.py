import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 模拟数据的样本量
n_samples = 748

# 创建保存路径
save_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\05肾病\03机器学习预测慢性肾病的终末期肾病\01模拟数据"
os.makedirs(save_path, exist_ok=True)

# 模拟连续变量
continuous_data = {
    "Age": norm.rvs(loc=57.8, scale=17.6, size=n_samples),
    "Systolic Blood Pressure (SBP)": norm.rvs(loc=129.5, scale=17.8, size=n_samples),
    "Diastolic Blood Pressure (DBP)": norm.rvs(loc=77.7, scale=11.1, size=n_samples),
    "BMI": norm.rvs(loc=24.8, scale=3.7, size=n_samples),
    "Creatinine (µmol/L)": norm.rvs(loc=130.0, scale=30.0, size=n_samples),  # 假设标准差为30
    "Urea (mmol/L)": norm.rvs(loc=7.9, scale=2.0, size=n_samples),  # 假设标准差为2
    "Total Protein (g/L)": norm.rvs(loc=71.6, scale=8.4, size=n_samples),
    "Albumin (g/L)": norm.rvs(loc=42.2, scale=5.5, size=n_samples),
    "ALT (U/L)": norm.rvs(loc=17.0, scale=5.0, size=n_samples),  # 假设标准差为5
    "AST (U/L)": norm.rvs(loc=18.0, scale=3.0, size=n_samples),  # 假设标准差为3
    "ALP (U/L)": norm.rvs(loc=60.0, scale=10.0, size=n_samples),  # 假设标准差为10
    "Urine Acid (µmol/L)": norm.rvs(loc=374.0, scale=70.0, size=n_samples),  # 假设标准差为70
    "Calcium (mmol/L)": norm.rvs(loc=2.2, scale=0.1, size=n_samples),
    "Phosphorous (mmol/L)": norm.rvs(loc=1.2, scale=0.2, size=n_samples),
    "Calcium-Phosphorus Product (Ca×P)": norm.rvs(loc=33.5, scale=5.6, size=n_samples),
    "Blood Leukocyte Count (10⁹/L)": norm.rvs(loc=7.1, scale=2.4, size=n_samples),
    "Hemoglobin (g/L)": norm.rvs(loc=131.0, scale=20.3, size=n_samples),
    "Platelet Count (10⁹/L)": norm.rvs(loc=209.8, scale=57.1, size=n_samples),
    "eGFR (ml/min/1.73m²)": norm.rvs(loc=46.1, scale=15.0, size=n_samples),  # 假设标准差为15
    "Total Cholesterol (mmol/L)": norm.rvs(loc=5.1, scale=0.6, size=n_samples),  # 假设标准差为0.6
    "Triglyceride (mmol/L)": norm.rvs(loc=1.8, scale=0.5, size=n_samples),  # 假设标准差为0.5
    "HDL-c (mmol/L)": norm.rvs(loc=1.3, scale=0.2, size=n_samples),  # 假设标准差为0.2
    "LDL-c (mmol/L)": norm.rvs(loc=3.0, scale=0.5, size=n_samples),  # 假设标准差为0.5
    "Potassium (mmol/L)": norm.rvs(loc=4.3, scale=0.5, size=n_samples),
    "Sodium (mmol/L)": norm.rvs(loc=140.2, scale=2.8, size=n_samples),
    "Chlorine (mmol/L)": norm.rvs(loc=106.9, scale=3.7, size=n_samples),
    "Bicarbonate (mmol/L)": norm.rvs(loc=25.9, scale=3.6, size=n_samples)
}

# 模拟分类变量
categorical_data = {
    "Gender": np.random.choice(["Male", "Female"], size=n_samples, p=[419/748, 329/748]),
    "Primary Disease": np.random.choice(
        ["Primary GN", "Diabetes", "Hypertension", "CIN", "Others", "Unknown"],
        size=n_samples,
        p=[292/748, 224/748, 97/748, 64/748, 18/748, 53/748]
    ),
    "Hypertension History": np.random.choice(["Yes", "No"], size=n_samples, p=[558/748, 1 - 558/748]),
    "Diabetes Mellitus History": np.random.choice(["Yes", "No"], size=n_samples, p=[415/748, 1 - 415/748]),
    "Cardiovascular or Cerebrovascular Disease History": np.random.choice(["Yes", "No"], size=n_samples, p=[177/748, 1 - 177/748]),
    "Smoking History": np.random.choice(["Yes", "No"], size=n_samples, p=[91/748, 1 - 91/748])
}

# 模拟肾病分期
ckd_stages = np.random.choice(
    ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"],
    size=n_samples,
    p=[58/748, 183/748, 352/748, 119/748, 36/748]
)

# 模拟目标变量
eskd_status = np.random.choice(["ESKD+", "ESKD-"], size=n_samples, p=[70/748, 1 - 70/748])

# 合并所有数据
simulated_data = pd.DataFrame({**continuous_data, **categorical_data, "CKD Stage": ckd_stages, "ESKD Status": eskd_status})

# 引入缺失数据
missing_rates = {
    "Creatinine (µmol/L)": 0.017,
    "Urea (mmol/L)": 0.017,
    "ALT (U/L)": 0.013,
    "AST (U/L)": 0.013,
    "ALP (U/L)": 0.013,
    "Total Protein (g/L)": 0.0,
    "Albumin (g/L)": 0.0,
    "Calcium (mmol/L)": 0.0,
    "Phosphorous (mmol/L)": 0.0,
    "Calcium-Phosphorus Product (Ca×P)": 0.0,
    "Blood Leukocyte Count (10⁹/L)": 0.0,
    "Hemoglobin (g/L)": 0.0,
    "Platelet Count (10⁹/L)": 0.0,
    "eGFR (ml/min/1.73m²)": 0.0,
    "Total Cholesterol (mmol/L)": 0.0,
    "Triglyceride (mmol/L)": 0.0,
    "HDL-c (mmol/L)": 0.0,
    "LDL-c (mmol/L)": 0.0,
    "Potassium (mmol/L)": 0.0,
    "Sodium (mmol/L)": 0.0,
    "Chlorine (mmol/L)": 0.0,
    "Bicarbonate (mmol/L)": 0.0
}

# 随机引入缺失数据
for column, rate in missing_rates.items():
    if rate > 0:
        missing_indices = np.random.choice(n_samples, size=int(rate * n_samples), replace=False)
        simulated_data.loc[missing_indices, column] = np.nan

# 不使用多重插补处理缺失数据
imputer = IterativeImputer(random_state=42, max_iter=10, n_nearest_features=5)
# simulated_data_imputed = pd.DataFrame(imputer.fit_transform(simulated_data), columns=simulated_data.columns)
simulated_data_imputed = pd.DataFrame(simulated_data, columns=simulated_data.columns)

# 保存模拟数据
file_path = os.path.join(save_path, "simulated_ckd_data_with_missing_and_imputation.csv")
simulated_data_imputed.to_csv(file_path, index=False)
print(f"Simulated data with missing values and imputation saved to {file_path}")