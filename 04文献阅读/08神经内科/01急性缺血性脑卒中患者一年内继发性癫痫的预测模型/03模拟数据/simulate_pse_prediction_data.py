import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于数学计算
import os  # 导入os库，用于操作系统相关的功能，如文件路径操作

def simulate_pse_data(n_samples=1000, seed=42):
    """
    模拟急性缺血性脑卒中患者继发性癫痫预测数据
    包含连续变量（正态分布）和分类变量（二项分布）
    """
    np.random.seed(seed)  # 设置随机种子，以确保每次生成的随机数据相同

    # 创建一个数据框，其中包含模拟的变量数据
    data = pd.DataFrame({
        # 连续变量（根据抓取的均值、标准差、范围生成）
        "NIHSS": np.clip(np.random.normal(7.2, 3.1, n_samples).round(1), 0, 24),  # NIHSS评分，取值范围[0, 24]，正态分布
        "WBC": np.clip(np.random.normal(8.5, 2.3, n_samples).round(1), 3.2, 18.1),  # 白细胞计数，正态分布
        "D-dimer": np.clip(np.random.normal(0.98, 0.45, n_samples).round(2), 0.1, 2.6),  # D-二聚体，正态分布
        "Lactate": np.clip(np.random.normal(1.8, 0.6, n_samples).round(2), 0.5, 4.0),  # 乳酸水平，正态分布
        "AST": np.clip(np.random.normal(25.4, 10.7, n_samples).round(1), 10, 75),  # 谷丙转氨酶，正态分布
        "CRP": np.clip(np.random.normal(9.1, 4.2, n_samples).round(1), 1.2, 23.5),  # C反应蛋白，正态分布
        "HbA1c": np.clip(np.random.normal(6.4, 1.1, n_samples).round(2), 4.8, 10.2),  # 糖化血红蛋白，正态分布
        "CK-MB": np.clip(np.random.normal(18.5, 5.6, n_samples).round(1), 5.2, 45.0),  # CK-MB，正态分布

        # 分类变量（二项分布模拟，基于文献中阳性构成比）
        "Sex_Female": np.random.binomial(1, 0.547, n_samples),  # 性别，二项分布，女性比例为54.7%
        "Hypertension": np.random.binomial(1, 0.676, n_samples),  # 高血压，二项分布，患病比例为67.6%
        "DVT": np.random.binomial(1, 0.091, n_samples),  # 深静脉血栓，二项分布，患病比例为9.1%
        "Atrial_fibrillation": np.random.binomial(1, 0.172, n_samples),  # 心房颤动，二项分布，患病比例为17.2%
        "Hyperuricemia": np.random.binomial(1, 0.134, n_samples),  # 高尿酸血症，二项分布，患病比例为13.4%
        "Hydrocephalus": np.random.binomial(1, 0.109, n_samples),  # 脑积水，二项分布，患病比例为10.9%
        "Fatty_liver": np.random.binomial(1, 0.093, n_samples),  # 脂肪肝，二项分布，患病比例为9.3%
    })

    return data  # 返回生成的模拟数据

# 模拟数据生成
simulated_data = simulate_pse_data(n_samples=1000)  # 调用simulate_pse_data函数生成1000个样本的模拟数据

# 设置保存路径
save_dir = "04文献阅读/08神经内科/01急性缺血性脑卒中患者一年内继发性癫痫的预测模型/03模拟数据/01data"  # 设置文件保存的目录路径
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，创建目录

# 保存为CSV
file_path = os.path.join(save_dir, "simulated_pse_data.csv")  # 生成保存数据的完整文件路径
simulated_data.to_csv(file_path, index=False)  # 将数据保存为CSV文件，不保存行索引

print(f"模拟数据已保存至：{file_path}")  # 输出保存成功的路径
