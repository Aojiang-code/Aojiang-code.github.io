# 导入所需的库
import numpy as np  # 用于数值计算和随机数生成
import pandas as pd  # 用于数据操作和DataFrame结构
import os  # 用于文件和目录操作

# 设置随机种子，确保每次运行生成的随机数一致，便于复现
np.random.seed(42)

# 根据中位数和四分位数(Q1, Q3)模拟符合正态分布的连续变量数据
def simulate_from_iqr(median, q1, q3, size):
    std_approx = (q3 - q1) / 1.35  # 使用IQR（四分位距）近似标准差（正态分布下的经验值）
    return np.random.normal(loc=median, scale=std_approx, size=size)  # 返回模拟的正态分布数据

# 模拟某个分组下的全部数据（包括连续变量和分类变量）
def simulate_group_data(group_info, size_total, size_aad):
    size_non_aad = size_total - size_aad  # 非AAD样本数量
    df_data = {}  # 用于存储所有变量的数据

    # 处理连续变量
    for var, ((med_non, q1_non, q3_non), (med_aad, q1_aad, q3_aad)) in group_info["continuous"].items():
        # 分别对非AAD和AAD组模拟数据，并拼接为一个数组
        df_data[var] = np.concatenate([
            simulate_from_iqr(med_non, q1_non, q3_non, size_non_aad),
            simulate_from_iqr(med_aad, q1_aad, q3_aad, size_aad)
        ])

    # 处理分类变量（二分类，1表示阳性，0表示阴性）
    for var, ((pos_non, total_non), (pos_aad, total_aad)) in group_info["categorical"].items():
        neg_non = total_non - pos_non  # 非AAD组中阴性数量
        neg_aad = total_aad - pos_aad  # AAD组中阴性数量
        df_data[var] = np.concatenate([
            np.random.choice([1, 0], size=total_non, p=[pos_non/total_non, neg_non/total_non]),  # 非AAD组模拟
            np.random.choice([1, 0], size=total_aad, p=[pos_aad/total_aad, neg_aad/total_aad])   # AAD组模拟
        ])

    # 添加标签列，0表示非AAD，1表示AAD
    df_data["AAD"] = np.array([0]*size_non_aad + [1]*size_aad)
    
    # 返回构造好的DataFrame
    return pd.DataFrame(df_data)

# 定义训练集的连续变量信息（每个变量给出：非AAD组 和 AAD组 的中位数、Q1和Q3）
continuous_info = {
    "age": [(73.0, 66.0, 81.0), (74.0, 67.5, 82.5)],
    "BMI": [(23.8, 21.3, 25.6), (23.8, 22.6, 25.0)],
    "Hb": [(107.0, 92.0, 122.0), (95.0, 83.5, 109.0)],
    "CRP": [(1.2, 0.2, 3.9), (3.5, 1.3, 7.9)],
    "PCT": [(0.1, 0.1, 0.6), (0.8, 0.2, 2.2)],
    "Scr": [(72.0, 54.9, 92.1), (80.2, 57.8, 106.6)]
}

# 训练集的分类变量信息（每个变量给出：阳性数量，总人数）
categorical_info_train = {
    "Male": [(260, 457), (83, 139)],
    "RRT": [(31, 457), (23, 139)],
    "Enteral_nutrition": [(137, 457), (103, 139)]
}

# 测试集的分类变量信息（同样的结构）
categorical_info_test = {
    "Male": [(108, 197), (32, 55)],
    "RRT": [(10, 197), (14, 55)],
    "Enteral_nutrition": [(59, 197), (39, 55)]
}

# 构建训练集模拟数据
group_train = {
    "continuous": continuous_info,
    "categorical": categorical_info_train
}
df_train = simulate_group_data(group_train, size_total=596, size_aad=139)

# 定义测试集的连续变量信息（与训练集数值不同）
continuous_info_test = {
    "age": [(74.0, 68.0, 82.0), (75.0, 67.5, 82.0)],
    "BMI": [(23.2, 20.8, 24.6), (23.8, 21.0, 25.6)],
    "Hb": [(103.0, 91.0, 116.0), (93.0, 83.5, 109.0)],
    "CRP": [(1.4, 0.3, 4.2), (3.9, 1.6, 8.9)],
    "PCT": [(0.1, 0.1, 0.7), (1.2, 0.2, 3.2)],
    "Scr": [(69.8, 56.0, 87.4), (88.1, 66.0, 118.6)]
}

# 构建测试集模拟数据
group_test = {
    "continuous": continuous_info_test,
    "categorical": categorical_info_test
}
df_test = simulate_group_data(group_test, size_total=252, size_aad=55)

# 指定数据保存路径
output_path = "04文献阅读/09ICU/01抗生素相关性腹泻/01模拟数据/data"
os.makedirs(output_path, exist_ok=True)  # 若目录不存在则创建

# 保存生成的训练集和测试集为CSV文件
df_train.to_csv(os.path.join(output_path, "aad_icu_train.csv"), index=False)
df_test.to_csv(os.path.join(output_path, "aad_icu_test.csv"), index=False)

# 打印提示信息
print("✅ 模拟数据已保存为：aad_icu_train.csv 与 aad_icu_test.csv")
