# reproduce_cap_ctd_analysis.py

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# 设置随机种子，确保可复现性
np.random.seed(42)

# 模拟数据加载
data_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data\simulated_full_cap_ctd_data.csv"
df = pd.read_csv(data_path)

# ---------------- 数据预处理 ----------------
# 选择目标变量 (ICU入院) 与特征变量
X = df.drop(columns=['ICU_admission'])
y = df['ICU_admission']

# 将数据划分为训练集和测试集 (70% 训练集，30% 测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---------------- 特征选择 ----------------
# 单因素分析
# (这里只做一个简单的显著性检验，后续可加入更多特征选择方法)
from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(chi2, k=16).fit_transform(X_train, y_train)

# ---------------- 模型构建 ----------------
# 构建9个不同的模型
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# ---------------- 模型评估 ----------------
# 使用10折交叉验证评估模型
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    results[name] = cv_results
    print(f"{name} - Mean AUC: {cv_results.mean():.4f}")

# ---------------- 类别不平衡处理（SMOTE） ----------------
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ---------------- 模型调优（使用网格搜索调参） ----------------
param_grid = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},
    'Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]},
    'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
}

# 选择调优模型 (举例：随机森林调优)
grid_search = GridSearchCV(RandomForestClassifier(), param_grid['Random Forest'], cv=cv, scoring='roc_auc')
grid_search.fit(X_resampled, y_resampled)
print(f"Best Parameters for Random Forest: {grid_search.best_params_}")

# ---------------- 模型训练与预测 ----------------
best_rf_model = grid_search.best_estimator_

best_rf_model.fit(X_resampled, y_resampled)
y_pred = best_rf_model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Confusion Matrix: \n{conf_matrix}")

# ---------------- ROC 曲线绘制 ----------------
fpr, tpr, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# ---------------- 结果保存 ----------------
output_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "model_results.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"模型评估结果已保存至：{output_file}")
