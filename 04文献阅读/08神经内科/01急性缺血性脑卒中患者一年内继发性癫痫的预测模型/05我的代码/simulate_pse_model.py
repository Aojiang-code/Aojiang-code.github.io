
"""
simulate_pse_model.py

用于模拟复现文献《Predictive models for secondary epilepsy in patients with acute ischemic stroke within one year》中的机器学习建模流程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 读取模拟数据
df = pd.read_csv("04文献阅读/08神经内科/01急性缺血性脑卒中患者一年内继发性癫痫的预测模型/03模拟数据/01data/simulated_pse_data.csv")

# 模拟标签生成（逻辑规则可替换为临床特征加权）
np.random.seed(42)
df["PSE"] = np.random.binomial(1, 0.043, df.shape[0])

# 特征和标签
X = df.drop("PSE", axis=1)
y = df["PSE"]

# 划分训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SMOTEENN处理不平衡
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# 特征筛选（LASSO回归）
lasso = LassoCV(cv=5, random_state=42).fit(X_resampled, y_resampled)
selected_features = X.columns[(lasso.coef_ != 0)]
X_resampled = X_resampled[selected_features]
X_test_selected = X_test[selected_features]

# 模型构建（以随机森林为例）
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
}
grid_search = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=5)
grid_search.fit(X_resampled, y_resampled)
best_rf = grid_search.best_estimator_

# 模型评估
y_pred = best_rf.predict(X_test_selected)
y_proba = best_rf.predict_proba(X_test_selected)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label="ROC curve (area = {:.2f})".format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")

# SHAP解释模型
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_selected)

shap.summary_plot(shap_values[1], X_test_selected, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
