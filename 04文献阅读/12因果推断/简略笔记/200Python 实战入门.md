# 📓 阶段 2 学习笔记 —— Python 实战入门

## 🎯 学习目标

* 能用 Python 在数据上跑出 **ATE（平均因果效应）** 和 **CATE（条件/个体化因果效应）**
* 学会用 `DoWhy`、`EconML`、`causalml` 等工具
* 掌握医学常见场景（药物干预、生存率分析）的基础实现流程

---

## 2.1 使用 DoWhy 进行因果推断

### ① 加载内置 Lalonde 数据集

Lalonde 数据是因果推断教学的“Hello World”，相当于模拟了一次临床试验：**就业培训（干预） → 收入（结果）**。

```python
import dowhy.datasets
import pandas as pd

# 加载数据（包含 treatment, outcome, 混杂因子）
data = dowhy.datasets.lalonde_binary()

df = data["df"]
print(df.head())
```

* `treatment`：是否接受培训（模拟药物治疗）
* `re78`：1978 年收入（模拟临床结局）
* 其他：混杂变量（年龄、教育、种族等）

---

### ② 构建因果模型（DAG）

在 `DoWhy` 里，我们需要用 **因果图**（DAG）告诉模型：哪些变量是混杂因素。

```python
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="re78",
    common_causes=["age","educ","black","hispan","married","nodegree","re74","re75"]
)

model.view_model()  # 输出因果图
```

这一步相当于在临床研究中明确：哪些是**基线协变量**。

---

### ③ 估计治疗效果（ATE）

DoWhy 支持多种估计方法（回归、PSM、IPTW 等）。

```python
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)
print("ATE (PSM):", estimate.value)
```

输出即为 **平均因果效应（ATE）**。

---

### ④ 假设检验与敏感性分析

```python
refutation = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause"
)
print(refutation)
```

这一步类似医学里的 **敏感性分析**：检验结果是否稳健。

---

## 2.2 使用 EconML 进行个体化治疗效果估计（ITE/CATE）

在临床里，我们不仅要知道“平均效果”，还要知道**不同患者是否有差异化获益**。这时用 `EconML`。

### ① 加载 UCI Heart Disease 数据

```python
import pandas as pd

url = "https://raw.githubusercontent.com/selva86/datasets/master/Heart.csv"
df = pd.read_csv(url)
print(df.head())
```

假设：

* `treatment` = 是否服用某药物（这里可用 `AHD` 字段假设为治疗标志）
* `Chol`（胆固醇）、`Age`、`Sex` 等作为协变量

---

### ② T-learner / X-learner

```python
from econml.metalearners import TLearner, XLearner
from sklearn.ensemble import RandomForestRegressor

# 定义特征和变量
X = df[["Age","Sex","Chol","RestBP"]]
T = (df["AHD"]=="Yes").astype(int)  # 0/1 处理
Y = df["MaxHR"]

# T-learner
t_learner = TLearner(models=RandomForestRegressor())
t_learner.fit(Y, T, X=X)
cate_t = t_learner.effect(X)

print("个体化治疗效应前 5 个：", cate_t[:5])
```

* 输出的是每个患者的 **CATE**，即“在他的条件下，治疗 vs 不治疗的差异”。

---

### ③ 解释不同患者的效果

我们可以画出 CATE 的分布，看哪些人群更获益。

```python
import matplotlib.pyplot as plt
plt.hist(cate_t, bins=30)
plt.xlabel("Estimated CATE")
plt.ylabel("Patients")
plt.title("Distribution of Individual Treatment Effects")
plt.show()
```

这类似医学里的“**异质性治疗效果（HTE）**”。

---

## 2.3 倾向评分匹配（PSM）实战

### ① 模拟数据：二甲双胍治疗

```python
import numpy as np

np.random.seed(42)
n = 500
age = np.random.randint(40,70,n)
bmi = np.random.normal(28,5,n)
treatment = np.random.binomial(1, p=1/(1+np.exp(-(0.1*age-0.2*bmi))), size=n)

# 结局（血糖水平）
y = 10 - 0.5*treatment + 0.05*age + 0.1*bmi + np.random.normal(0,1,n)

df = pd.DataFrame({"age":age,"bmi":bmi,"treatment":treatment,"y":y})
print(df.head())
```

---

### ② 估计倾向评分 + 匹配

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# 倾向评分
logit = LogisticRegression()
logit.fit(df[["age","bmi"]], df["treatment"])
ps = logit.predict_proba(df[["age","bmi"]])[:,1]

df["ps"] = ps

# 最近邻匹配
treated = df[df.treatment==1]
control = df[df.treatment==0]

nn = NearestNeighbors(n_neighbors=1).fit(control[["ps"]])
_, idx = nn.kneighbors(treated[["ps"]])
matched_control = control.iloc[idx.flatten()]
matched = pd.concat([treated, matched_control])
```

---

### ③ 匹配前后平衡性

```python
import seaborn as sns

sns.kdeplot(df[df.treatment==1]["ps"], label="Treated", shade=True)
sns.kdeplot(df[df.treatment==0]["ps"], label="Control", shade=True)
plt.title("倾向评分分布（匹配前）")
plt.show()

sns.kdeplot(matched[matched.treatment==1]["ps"], label="Treated", shade=True)
sns.kdeplot(matched[matched.treatment==0]["ps"], label="Control", shade=True)
plt.title("倾向评分分布（匹配后）")
plt.show()
```

---

# ✅ 阶段 2 总结

1. **DoWhy** → 最适合入门，能跑 **ATE + 敏感性分析**
2. **EconML** → 适合研究 **CATE/个体化效应**，临床异质性分析很常用
3. **PSM 实战** → 医学研究最常见方法，要掌握“倾向评分分布对齐 + 平衡性诊断”

---

把这三部分整理成 **三个可运行 Notebook**（含完整代码+注释），直接放到下载区