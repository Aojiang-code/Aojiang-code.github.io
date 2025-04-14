# simulate_full_cap_ctd_data_with_features.py

import numpy as np
import pandas as pd
import os

# 设置随机种子保证可复现
np.random.seed(42)

# 模拟数据数量
n = 1000

# --------------- 连续变量模拟函数 ---------------
def simulate_from_iqr(median, q1, q3, dist="normal"):
    if dist == "normal":
        std = (q3 - q1) / 1.35  # 估算正态分布标准差
        return np.random.normal(loc=median, scale=std, size=n)
    elif dist == "lognormal":
        sigma = (np.log(q3) - np.log(q1)) / 1.35
        mu = np.log(median)
        return np.random.lognormal(mean=mu, sigma=sigma, size=n)
    else:
        raise ValueError("Invalid distribution type.")

# 连续变量模拟（符合补充材料中的“Table S1”信息）
data = {
    "age": simulate_from_iqr(56, 47, 66),
    "pH": simulate_from_iqr(7.41, 7.38, 7.44),
    "BUN (mmol/L)": simulate_from_iqr(6.8, 4.5, 10.7),
    "sodium (mmol/L)": simulate_from_iqr(137.3, 133.6, 140.4),
    "glucose (mmol/L)": simulate_from_iqr(6.93, 4.68, 10.78, dist="lognormal"),
    "hematocrit (L/L)": simulate_from_iqr(0.36, 0.31, 0.40),
    "PF ratio": simulate_from_iqr(207, 166, 279),
    "hemoglobin (g/L)": simulate_from_iqr(115, 103, 128),
    "RDW (%)": simulate_from_iqr(14.7, 13.7, 16.2),
    "platelet (×10^9/L)": simulate_from_iqr(182, 133, 245),
    "neutrophil (×10^9/L)": simulate_from_iqr(7.53, 5.02, 11.07),
    "lymphocyte (×10^9/L)": simulate_from_iqr(0.98, 0.59, 1.47, dist="lognormal"),
    "monocyte (×10^9/L)": simulate_from_iqr(0.33, 0.19, 0.51),
    "bilirubin (µmol/L)": simulate_from_iqr(9.4, 5.5, 11.9),
    "ALT (U/L)": simulate_from_iqr(23, 14, 53),
    "AST (U/L)": simulate_from_iqr(25, 18, 51),
    "albumin (g/L)": simulate_from_iqr(34.0, 28.7, 39.3),
    "globulin (g/L)": simulate_from_iqr(28.1, 23.2, 34.7),
    "creatinine (µmol/L)": simulate_from_iqr(55.00, 44.00, 71.00),
    "cystatin C (mg/L)": simulate_from_iqr(1.11, 0.94, 1.33),
    "triglyceride (mmol/L)": simulate_from_iqr(1.39, 1.00, 1.91),
    "HDL-C (mmol/L)": simulate_from_iqr(1.02, 0.74, 1.36),
    "LDL-C (mmol/L)": simulate_from_iqr(2.19, 1.56, 2.78),
    "creatine kinase (U/L)": simulate_from_iqr(52, 26, 154),
    "LDH (U/L)": simulate_from_iqr(246, 189, 355),
    "potassium (mmol/L)": simulate_from_iqr(3.50, 3.14, 3.83),
    "myoglobin (ng/mL)": simulate_from_iqr(43.51, 21.17, 106.60),
    "CK-MB (ng/mL)": simulate_from_iqr(2.25, 1.09, 4.84),
    "NT-proBNP (ng/L)": simulate_from_iqr(393, 149, 929, dist="lognormal"),
    "Troponin T (ng/L)": simulate_from_iqr(23.0, 11.2, 47.9),
    "CRP (mg/L)": simulate_from_iqr(29.50, 10.40, 86.00, dist="lognormal"),
    "Procalcitonin (ng/mL)": simulate_from_iqr(0.09, 0.05, 0.40),
    "PT (s)": simulate_from_iqr(11.3, 10.4, 12.3),
    "APTT (s)": simulate_from_iqr(27.6, 24.5, 31.3),
    "fibrinogen (g/L)": simulate_from_iqr(3.34, 2.55, 4.22),
    "AT III (%)": simulate_from_iqr(84.1, 70.4, 100.1),
    "D dimer (mg/L)": simulate_from_iqr(1.78, 0.74, 4.85),
    "PaCO2 (mmHg)": simulate_from_iqr(37.4, 32.9, 41.7),
    "lactate (mmol/L)": simulate_from_iqr(1.61, 1.16, 2.33),
    "CD4 + T cell (cell/µL)": simulate_from_iqr(346, 195, 516),
    "CD8 + T cell (cell/µL)": simulate_from_iqr(251, 130, 384),
}

# --------------- 分类变量模拟函数 ---------------
def simulate_binary(prob):
    return np.random.binomial(1, prob, size=n)

# 所有分类变量模拟（符合补充材料中提供的比例）
data.update({
    "male": simulate_binary(0.322),
    "confusion": simulate_binary(0.029),
    "positive_G_test": simulate_binary(0.219),
    "positive_GM_test": simulate_binary(0.027),
    "pleural_effusion": simulate_binary(0.357),
    "ICU_admission": simulate_binary(0.270),
    "Need_for_vasopressors": simulate_binary(0.256),
    "Need_for_IMV": simulate_binary(0.243),
    "hospital_mortality": simulate_binary(0.155),
    "ILD": simulate_binary(0.651),
    "COPD": simulate_binary(0.099),
    "diabetes": simulate_binary(0.129),
    "hypertension": simulate_binary(0.210),
    "cancer": simulate_binary(0.043),
    "chronic_liver_disease": simulate_binary(0.058),
    "chronic_renal_disease": simulate_binary(0.057),
    "congestive_heart_failure": simulate_binary(0.103),
    "cerebrovascular_disease": simulate_binary(0.037),
    "coronary_heart_disease": simulate_binary(0.041),
})

# 构建DataFrame
df = pd.DataFrame(data)

# --------------- 保存数据 ---------------
output_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "simulated_full_cap_ctd_data_with_features.csv")
df.to_csv(output_file, index=False)

print(f"模拟数据已成功保存至：{output_file}")
