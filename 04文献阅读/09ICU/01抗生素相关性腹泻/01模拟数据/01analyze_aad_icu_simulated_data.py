# 导入相关库
import pandas as pd  # 用于读取和操作数据
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 基于matplotlib的高级可视化库
import os  # 用于文件路径处理和文件夹创建

# 设置Seaborn绘图风格和字体大小
sns.set(style="whitegrid", font_scale=1.2)

# 定义数据所在的基础路径
base_path = "04文献阅读/09ICU/01抗生素相关性腹泻/01模拟数据/data"
train_path = os.path.join(base_path, "aad_icu_train.csv")  # 训练集路径
test_path = os.path.join(base_path, "aad_icu_test.csv")    # 测试集路径

# 读取训练集和测试集数据
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 添加新列标记数据来源（训练/测试）
df_train["dataset"] = "train"
df_test["dataset"] = "test"

# 合并训练集和测试集为一个整体数据集
df_all = pd.concat([df_train, df_test], ignore_index=True)

# 保存合并后的数据为CSV文件
df_all.to_csv(os.path.join(base_path, "aad_icu_all.csv"), index=False)

# 创建图像保存目录，如果不存在则自动创建
img_path = os.path.join(base_path, "figures")
os.makedirs(img_path, exist_ok=True)

# 定义变量名称列表
continuous_vars = ["age", "BMI", "Hb", "CRP", "PCT", "Scr"]  # 连续变量
categorical_vars = ["Male", "RRT", "Enteral_nutrition"]      # 分类变量

# 1. 绘制箱线图：用于展示连续变量在 AAD 组和非AAD组的分布情况
for var in continuous_vars:
    plt.figure(figsize=(8, 5))  # 设置图像大小
    sns.boxplot(x="AAD", y=var, data=df_all, palette="Set2")  # 绘制箱线图
    plt.title(f"Boxplot of {var} by AAD")  # 设置标题
    plt.xlabel("AAD (0 = No, 1 = Yes)")  # 设置x轴标签
    plt.savefig(os.path.join(img_path, f"boxplot_{var}.png"), dpi=300)  # 保存图像
    plt.close()  # 关闭当前图像，防止叠加

# 2. 绘制小提琴图：在箱线图基础上增加分布密度信息
for var in continuous_vars:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="AAD", y=var, data=df_all, palette="Set3", inner="quartile")  # 使用quartile标示中位数和四分位数
    plt.title(f"Violin Plot of {var} by AAD")
    plt.savefig(os.path.join(img_path, f"violin_{var}.png"), dpi=300)
    plt.close()

# 3. 绘制连续变量的分布直方图（按AAD分组）
for var in continuous_vars:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df_all, x=var, hue="AAD", kde=True, palette="coolwarm",
        element="step", common_norm=False  # 不统一归一化，让每组独立显示频率
    )
    plt.title(f"Distribution of {var} by AAD")
    plt.savefig(os.path.join(img_path, f"hist_{var}.png"), dpi=300)
    plt.close()

# 4. 绘制连续变量之间的相关性热图
plt.figure(figsize=(10, 8))
corr_matrix = df_all[continuous_vars].corr()  # 计算连续变量之间的相关系数矩阵
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", square=True)  # 带数值标注
plt.title("Correlation Heatmap of Continuous Variables")
plt.savefig(os.path.join(img_path, "heatmap_continuous_vars.png"), dpi=300)
plt.close()

# 打印完成提示
print("✅ 可视化分析完成，图像已保存至 figures 文件夹")
