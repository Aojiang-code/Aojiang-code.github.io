import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子以确保可重复性
np.random.seed(289)

# 模拟患者ID
num_patients = 100
patient_ids = [f'patient_{i}' for i in range(1, num_patients + 1)]

# 模拟日期
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 1, 1)
date_range = pd.date_range(start=start_date, end=end_date, freq='M')

# 创建模拟数据
data = {
    'id': np.repeat(patient_ids, len(date_range)),
    'date': np.tile(date_range, num_patients),
    'egfr': np.random.uniform(5, 100, num_patients * len(date_range)),  # 模拟EGFR值
    'creatinine': np.random.uniform(0.5, 5.0, num_patients * len(date_range)),  # 模拟肌酐值
    'glucose': np.random.uniform(70, 200, num_patients * len(date_range)),  # 模拟血糖值
    'height': np.random.uniform(150, 200, num_patients * len(date_range)),  # 模拟身高
    'bmi': np.random.uniform(18, 35, num_patients * len(date_range)),  # 模拟BMI
    'weight': np.random.uniform(50, 100, num_patients * len(date_range)),  # 模拟体重
    'sit_dia': np.random.randint(60, 90, num_patients * len(date_range)),  # 模拟坐位舒张压
    'sit_sys': np.random.randint(90, 140, num_patients * len(date_range)),  # 模拟坐位收缩压
    'sta_dia': np.random.randint(60, 90, num_patients * len(date_range)),  # 模拟站位舒张压
    'sta_sys': np.random.randint(90, 140, num_patients * len(date_range)),  # 模拟站位收缩压
    'sit_hr': np.random.randint(60, 100, num_patients * len(date_range)),  # 模拟坐位心率
    'sta_hr': np.random.randint(60, 100, num_patients * len(date_range)),  # 模拟站位心率
    'hgba1c': np.random.uniform(4, 10, num_patients * len(date_range)),  # 模拟HbA1c值
    'protcreatinine': np.random.uniform(0, 5, num_patients * len(date_range)),  # 模拟蛋白尿
    'proteinuria': np.random.uniform(0, 5, num_patients * len(date_range)),  # 模拟尿蛋白
}

# 创建DataFrame
df = pd.DataFrame(data)

# 添加一些额外的列以符合R代码的需求
df['value'] = np.random.uniform(0, 1, num_patients * len(date_range))  # 模拟值
df['gender'] = np.random.choice(['Male', 'Female'], num_patients * len(date_range))  # 模拟性别
df['age.init'] = np.random.randint(20, 80, num_patients * len(date_range))  # 模拟初始年龄

# 保存为CSV文件
df.to_csv('./data/LMprediction.csv', index=False)