import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score, f1_score,
    brier_score_loss, confusion_matrix, roc_curve
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置路径与读取数据
data_path = "04文献阅读/09ICU/01抗生素相关性腹泻/01模拟数据/data/aad_icu_all.csv"
df = pd.read_csv(data_path)

# 特征与标签分离
X = df.drop(columns=["AAD", "dataset"])
y = df["AAD"]

# 标准化处理（必要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO 特征选择
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
coef = pd.Series(lasso.coef_, index=df.drop(columns=["AAD", "dataset"]).columns)
selected_features = coef[coef != 0].index.tolist()

print("✅ LASSO 选中的变量：", selected_features)

# 使用选中的特征建模
X_selected = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)

# 五种模型
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(
        learning_rate=0.1, max_depth=3, n_estimators=20, use_label_encoder=False, eval_metric="logloss"
    )
}

# 模型评估函数
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Sensitivity": recall_score(y_test, y_pred),
        "Specificity": recall_score(y_test, y_pred, pos_label=0),
        "F1 Score": f1_score(y_test, y_pred),
        "Brier Score": brier_score_loss(y_test, y_prob)
    }
    return results, model

results_list = []
trained_models = {}

for name, model in models.items():
    res, fitted_model = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results_list.append(res)
    trained_models[name] = fitted_model

# 输出结果
results_df = pd.DataFrame(results_list)
print("\n📊 模型性能比较：")
print(results_df)

# 保存性能表格
output_path = "04文献阅读/09ICU/01抗生素相关性腹泻/01模拟数据/data"
results_df.to_csv(os.path.join(output_path, "model_evaluation.csv"), index=False)

# SHAP 可解释分析（仅对XGBoost）
xgb_model = trained_models["XGBoost"]
explainer = shap.Explainer(xgb_model, X_selected)
shap_values = explainer(X_selected)

# 变量重要性图
shap.summary_plot(shap_values, X_selected, show=False)
plt.title("SHAP Feature Importance - XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "shap_summary_plot.png"), dpi=300)
plt.close()

# 示例解释图
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP Waterfall Example (Sample 0)")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "shap_waterfall_sample0.png"), dpi=300)

print("✅ SHAP 可解释图已生成并保存")

# K折交叉验证
cv_scores = cross_val_score(xgb_model, X_selected, y, cv=5, scoring="roc_auc")
print(f"\n✅ K折交叉验证AUC均值：{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
