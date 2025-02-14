import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 加载数据
file_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\05肾病\文献信息.csv"
try:
    df = pd.read_csv(file_path)
    print("数据加载成功！")
except FileNotFoundError:
    print(f"未找到文件：{file_path}")
    exit()
except Exception as e:
    print(f"加载数据时发生错误：{e}")
    exit()

# 检查影响因子列是否存在
if '影响因子' not in df.columns:
    print("数据中不存在'影响因子'列，请检查数据文件。")
    exit()

# 尝试将影响因子列转换为浮点数
try:
    df['影响因子'] = df['影响因子'].astype(float)
except ValueError as e:
    print(f"数据转换错误：'影响因子'列包含无法转换为浮点数的值。错误详情：{e}")
    exit()

# 绘制影响因子分布图
plt.figure(figsize=(10, 6))
sns.histplot(df['影响因子'], bins=10, kde=True)
plt.title("文献影响因子分布")
plt.xlabel("影响因子")
plt.ylabel("频数")
plt.show()