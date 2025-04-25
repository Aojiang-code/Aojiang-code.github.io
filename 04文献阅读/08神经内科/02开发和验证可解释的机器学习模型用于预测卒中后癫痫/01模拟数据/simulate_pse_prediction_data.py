import pandas as pd
import numpy as np

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 模拟数据的数量
n_samples = 1000

# 连续变量模拟
data_continuous = {
    'Age (years)': np.random.normal(64.33, 11.84, n_samples),  # 正态分布，均值64.33，标准差11.84
    'Length of stay (days)': np.random.choice([7, 8, 9, 10, 11], size=n_samples),  # 假设住院天数为离散的5个数值
    'NIHSS at admission': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], size=n_samples),  # 假设NIHSS的离散值
    'Fasting blood glucose (mmol/L)': np.random.normal(5.38, 1.14, n_samples),
    'Total cholesterol (mmol/L)': np.random.normal(4.44, 1.12, n_samples),
    'Triglycerides (mmol/L)': np.random.normal(1.2, 0.56, n_samples),
    'LDL cholesterol (mmol/L)': np.random.normal(2.66, 0.94, n_samples),
    'D-dimer (ng/mL)': np.random.normal(450, 900, n_samples)  # 假设D-二聚体是正态分布
}

# 分类变量模拟
data_categorical = {
    'Sex (Male)': np.random.choice([0, 1], size=n_samples, p=[0.349, 0.651]),  # 0为女性，1为男性
    'Hypertension': np.random.choice([0, 1], size=n_samples, p=[0.2778, 0.7222]),  # 0为无，1为有
    'Diabetes': np.random.choice([0, 1], size=n_samples, p=[0.666, 0.334]),  # 0为无，1为有
    'Hyperlipidemia': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
    'Atrial fibrillation': np.random.choice([0, 1], size=n_samples, p=[0.926, 0.074]),
    'Coronary heart disease': np.random.choice([0, 1], size=n_samples, p=[0.828, 0.171]),
    'Stroke Cause - Large-artery atherosclerosis': np.random.choice([0, 1], size=n_samples, p=[0.4378, 0.5622]),
    'Stroke Cause - Cardioembolism': np.random.choice([0, 1], size=n_samples, p=[0.939, 0.061]),
    'Stroke Cause - Small-vessel occlusion': np.random.choice([0, 1], size=n_samples, p=[0.715, 0.285]),
    'Early Seizure': np.random.choice([0, 1], size=n_samples, p=[0.992, 0.008]),  # 0为无，1为有
    'Cortical Involvement': np.random.choice([0, 1], size=n_samples, p=[0.744, 0.256]),
    'Multiple Lobes Involvement': np.random.choice([0, 1], size=n_samples, p=[0.765, 0.235])
}

# 创建DataFrame
df = pd.DataFrame({**data_continuous, **data_categorical})

# 保存为CSV文件
save_path = r"04文献阅读\08神经内科\02开发和验证可解释的机器学习模型用于预测卒中后癫痫\01模拟数据\01data\simulated_data.csv"
df.to_csv(save_path, index=False)

print(f"模拟数据已保存至 {save_path}")
