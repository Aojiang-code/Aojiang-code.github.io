# simulate_cap_ctd_data.py

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

# 连续变量模拟（仅选部分代表性变量作为示例）
data = {
    "age": simulate_from_iqr(56, 47, 66),
    "sodium": simulate_from_iqr(137.3, 133.6, 140.4),
    "glucose": simulate_from_iqr(6.93, 4.68, 10.78, dist="lognormal"),
    "CRP": simulate_from_iqr(29.5, 10.4, 86.0, dist="lognormal"),
    "NT_proBNP": simulate_from_iqr(393, 149, 929, dist="lognormal"),
    "CD4_T_cell": simulate_from_iqr(346, 195, 516),
    "lymphocyte": simulate_from_iqr(0.98, 0.59, 1.47, dist="lognormal"),
    "serum_sodium": simulate_from_iqr(137.3, 133.6, 140.4),
    "platelet": simulate_from_iqr(182, 133, 245),
    "PF_ratio": simulate_from_iqr(207, 166, 279),
}

# --------------- 分类变量模拟函数 ---------------
def simulate_binary(prob):
    return np.random.binomial(1, prob, size=n)

# 分类变量模拟（选代表性变量）
data.update({
    "male": simulate_binary(0.322),
    "ICU_admission": simulate_binary(0.27),
    "positive_G_test": simulate_binary(0.219),
    "COPD": simulate_binary(0.099),
    "diabetes": simulate_binary(0.129),
    "hypertension": simulate_binary(0.21),
    "pleural_effusion": simulate_binary(0.357),
    "hospital_mortality": simulate_binary(0.155),
})

# 构建DataFrame
df = pd.DataFrame(data)

# --------------- 保存数据 ---------------
output_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "simulated_cap_ctd_data.csv")
df.to_csv(output_file, index=False)

print(f"模拟数据已成功保存至：{output_file}")
