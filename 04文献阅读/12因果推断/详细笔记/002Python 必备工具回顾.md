# 0.2 Python 必备工具回顾（快速复习 · 实战版）

## 0.2.1 `pandas` 数据操作（基于乳腺癌数据集）

> 数据集：`sklearn.datasets.load_breast_cancer()`
> 任务：把数据转成 DataFrame，做筛选、分组统计、交叉表与缺失/类型检查
> 因果视角小提示：此处仅做 EDA 练习；真正的因果分析中要注意“暴露/治疗→结局”的时间顺序与混杂控制，这里不涉及干预变量

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1) 载入并转换为 DataFrame
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')  # 1=malignant(恶性), 0=benign(良性)
df = pd.concat([X, y], axis=1)

# 2) 快速查看结构与类型
print("形状:", df.shape)
print(df.head(3))
print("\n信息(info)：")
print(df.info())

# 3) 缺失值检查（该数据集通常没有缺失，这里演示写法）
print("\n缺失值统计：")
print(df.isnull().sum().sort_values(ascending=False).head())

# 4) 简单筛选：示例——筛选“恶性”样本中 mean radius > 20 的病例
df_sel = df[(df['target'] == 1) & (df['mean radius'] > 20)]
print("\n筛选后样本量:", df_sel.shape[0])

# 5) 分组统计：按 target（良/恶性）查看若干关键特征的均值
cols_focus = ['mean radius', 'mean texture', 'mean symmetry']
group_mean = df.groupby('target')[cols_focus].mean()
print("\n按 target 分组的均值：")
print(group_mean)

# 6) 交叉表（若有离散特征可做，这里先构造一个简化离散特征以示范）
#   将 'mean radius' 二分为大/小（仅演示用，真实分析请谨慎离散化）
df['radius_bin'] = (df['mean radius'] > df['mean radius'].median()).astype(int)
crosstab = pd.crosstab(index=df['target'], columns=df['radius_bin'], margins=True)
print("\n交叉表(target vs radius_bin)：")
print(crosstab)

# 7) 描述性统计
print("\n描述性统计（部分列）：")
print(df[cols_focus + ['target']].describe())
```

---

## 0.2.2 可视化（直方图 / 箱线图 / 相关性热图）

> 工具：`matplotlib` + `seaborn`
> 小提示：可视化时请不要把**结局变量**当作“特征”参与建模；这里仅用于 EDA 对比和理解数据分布

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1) 直方图：mean radius 分布（按 target 分层）
plt.figure()
sns.histplot(data=df, x='mean radius', hue='target', kde=True, element='step', stat='density')
plt.title('Distribution of Mean Radius by Target (0=Benign, 1=Malignant)')
plt.show()

# 2) 箱线图：不同 target 的 mean radius
plt.figure()
sns.boxplot(data=df, x='target', y='mean radius')
plt.title('Mean Radius by Target')
plt.show()

# 3) 相关性热图（选部分特征以避免图过密）
subset_cols = ['mean radius','mean texture','mean perimeter','mean area','mean smoothness']
plt.figure()
sns.heatmap(df[subset_cols].corr(), annot=True, fmt='.2f')
plt.title('Correlation Heatmap (Selected Features)')
plt.show()
```

> 因果视角思考：
>
> * 直方图/箱线图只能显示**关联**与**分布差异**，不能证明因果。
> * 若要研究“某治疗是否导致结局改善”，需要明确定义**治疗（暴露）**变量、**结局**变量，并处理**混杂**（下一阶段会做）。
> * 这一步的价值是：找到可能的混杂候选（如年龄、基线指标）与变量分布异常，为后续因果建模做准备。

---

## 0.2.3 补充练习：使用“糖尿病进展”数据集（回归场景）

> 数据集：`sklearn.datasets.load_diabetes()`（目标是一个连续变量——疾病进展指标）
> 练习：基本整理、几何分布图、变量与目标的关系散点

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

# 1) 载入并转换
ddata = load_diabetes()
Xd = pd.DataFrame(ddata.data, columns=ddata.feature_names)
yd = pd.Series(ddata.target, name='disease_progression')  # 连续结局
df_diab = pd.concat([Xd, yd], axis=1)

print("形状:", df_diab.shape)
print(df_diab.head())

# 2) 快速分布观察：结局变量直方图
plt.figure()
sns.histplot(df_diab['disease_progression'], kde=True)
plt.title('Distribution of Diabetes Progression')
plt.show()

# 3) 特征与结局的关系（举例：bmi 与 y）
plt.figure()
sns.scatterplot(data=df_diab, x='bmi', y='disease_progression')
plt.title('BMI vs Disease Progression')
plt.show()

# 4) 相关性热图（选取几个常见特征）
sel = ['bmi','bp','s1','s5','disease_progression']
plt.figure()
sns.heatmap(df_diab[sel].corr(), annot=True, fmt='.2f')
plt.title('Correlation (Selected Features)')
plt.show()
```

> 因果视角思考：
>
> * 这里的 `bmi→disease_progression` 关系仍然是**相关**层面的观察。
> * 如果你希望估计“降低 BMI 是否**导致**疾病进展减缓”，需要在纵向/干预数据上设定清晰的时间顺序与调整混杂（例如基线年龄、药物、并发症等），并使用因果方法（PSM/IPTW/双重稳健等）。

---

## 0.2.4 小结与常见踩坑

* **数据→DataFrame**：`sklearn` 的内置数据一般是 `numpy` 数组，记得加上列名再转成 `pandas`，便于后续分组与可视化。
* **缺失与类型**：医学数据真实场景常有缺失；此处内置数据几乎无缺失，练习时也要保留“缺失检查”的步骤与习惯。
* **分箱只是演示**：把连续变量粗暴二分仅用于教学；真实分析要谨慎，可能引入信息损失与偏倚。
* **EDA ≠ 因果**：本节的所有图表与统计都是“看数据”的第一步，不承担因果结论；下一步才会进入“处理-结局-混杂”的建模与估计。

---


配套代码:`002_Python_Primer_Med.ipynb`
