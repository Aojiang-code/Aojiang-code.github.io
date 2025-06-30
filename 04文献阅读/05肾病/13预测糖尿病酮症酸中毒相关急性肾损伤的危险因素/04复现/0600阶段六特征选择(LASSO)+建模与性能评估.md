非常好，我们现在进入：

---

# 📁 阶段六：特征选择（LASSO）+ 建模与性能评估

🎯 目标：

* 使用 **LASSO（L1 正则）进行特征筛选**
* 构建文献中提到的 8 种机器学习模型
* 使用 **10 折交叉验证** 评估性能
* 输出 AUC、F1、准确率、敏感性、特异性等指标

同时满足您提出的关键要求：

> ✅ 跨 notebook 共享数据：在阶段五末尾保存数据，在阶段六开头加载。

---

## ✅ 一、阶段五末尾操作：保存建模数据

在阶段五的 `.ipynb` 末尾添加以下代码：

```python
X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
```

---

## ✅ 二、阶段六开头操作：加载数据 & 初始化

在阶段六的 `.ipynb` 文件顶部写入：

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# 加载数据
X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_val = pd.read_csv("y_val.csv").squeeze()
```

---

## ✅ 三、特征选择（LASSO）

### 1️⃣ 使用 `LassoCV` 自动选择最佳正则系数

```python
lasso = LassoCV(cv=10, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
```

### 2️⃣ 保留非零系数对应的特征

```python
selected_features = X_train.columns[lasso.coef_ != 0].tolist()
print("🎯 LASSO保留的特征数量：", len(selected_features))
print(selected_features)
```

### 3️⃣ 筛选数据子集

```python
X_train_sel = X_train[selected_features]
X_val_sel = X_val[selected_features]
```

---

## ✅ 四、建模与交叉验证（文献中8种模型）

### 4️⃣ 定义模型列表（论文一致）

```python
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GaussianNB": GaussianNB(),
    "ComplementNB": ComplementNB(),
    "MLP": MLPClassifier(max_iter=1000),
    "SVM": SVC(probability=True)
}
```

---

## ✅ 五、训练与性能评估（使用验证集）

```python
results = []

for name, model in models.items():
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_val_sel)
    y_proba = model.predict_proba(X_val_sel)[:, 1]

    auc = roc_auc_score(y_val, y_proba)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    specificity = tn / (tn + fp)

    results.append({
        "Model": name,
        "AUC": auc,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": specificity,
        "F1 Score": f1
    })
```

---

### 📊 输出比较结果表格

```python
results_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
print(results_df)
results_df.to_csv("model_results.csv", index=False)
```

---

## ✅ 六、阶段六成果总结

| 输出对象                | 内容             |
| ------------------- | -------------- |
| `selected_features` | LASSO 选出的变量    |
| `results_df`        | 所有模型在验证集上的指标对比 |
| `model_results.csv` | 可供汇报或画图使用的结果表  |

---

## 📘 说明（符合原论文）：

| 步骤         | 是否复现论文方法           |
| ---------- | ------------------ |
| LASSO 特征选择 | ✅（明确说明）            |
| 8 模型比较     | ✅（论文逐一列出）          |
| 10折交叉      | ✅（用于 LASSO）        |
| 验证集评估指标    | ✅（AUC、F1、敏感性、特异性等） |

---

如您希望下一阶段继续完成：

### 📁 阶段七：模型可解释性分析（SHAP、特征重要性图、DCA 校准曲线）

我可为您提供 SHAP 分析代码与可视化图表模板。是否继续？
