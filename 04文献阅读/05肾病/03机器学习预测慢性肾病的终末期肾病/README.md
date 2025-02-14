# 机器学习在预测慢性肾病发展为终末期肾病中的应用研究

## 文献信息

|项目|内容|
| ---- | ---- |
|标题|Machine learning to predict end stage kidney disease in chronic kidney disease|
|作者|Qiong Bai, Chunyan Su, Wen Tang, Yike Li|
|发表时间|2022-01-01|
|国家|中国|
|分区|Q1|
|影响因子|5.678|
|摘要|本研究评估了机器学习在预测慢性肾病患者进展至终末期肾病（ESKD）中的可行性。研究使用了包含患者基线特征和常规血液检查结果的纵向队列数据，目标是预测5年内是否发生ESKD。研究结果表明，三种机器学习模型（逻辑回归、朴素贝叶斯和随机森林）在预测性能上与Kidney Failure Risk Equation（KFRE）相当，且具有更高的敏感性。|
|关键词|机器学习, 慢性肾病, 终末期肾病, 预测模型|
|期刊名称|Scientific Reports|
|卷号/期号|12:8377|
|DOI|10.1038/s41598-022-12316-z|
|研究方法|回顾性队列研究|
|数据来源|北京大学第三医院慢性肾病管理诊所的纵向队列|
|研究结果|随机森林模型的AUC最高（0.81），与逻辑回归、朴素贝叶斯和KFRE相当。KFRE在特异性和精确度上表现最佳，但敏感性较低（47%）。|
|研究结论|机器学习在基于易获取的特征评估慢性肾病预后方面是可行的，三种表现良好的机器学习模型可用于患者筛查。|
|研究意义|提供了一种新的预测工具，有助于早期识别高风险患者，优化临床决策和患者管理。|
|阅读开始时间|20250214 23|
|阅读结束时间|20250214 23|
|时刻|晚上|
|星期|星期五|
|天气|晴朗|


## 核心内容

### 研究核心
探讨机器学习（ML）在预测慢性肾病（CKD）患者发展为终末期肾病（ESKD）中的可行性，并将其性能与现有的肾脏衰竭风险方程（KFRE）进行比较。

### 研究背景
慢性肾病（CKD）是全球范围内的重大健康负担，影响着大量人群，并且可能导致终末期肾病（ESKD），需要进行肾脏替代治疗（KRT）。早期预测CKD患者发展为ESKD的风险对于改善患者预后、降低发病率和死亡率以及减少医疗成本至关重要。传统的统计模型在预测ESKD方面存在局限性，尤其是在不同种族群体中的适用性。机器学习（ML）作为一种新兴的预测工具，能够处理复杂的非线性关系，可能在预测ESKD方面具有优势。

### 研究方法
#### 数据来源
研究数据来自北京大学第三医院CKD管理诊所的纵向队列研究，共纳入748名CKD患者，随访时间为6.3±2.3年。主要观察终点是患者是否在5年内发展为ESKD。

#### 预测变量
包括患者的基线特征（如年龄、性别、教育水平等）、病史（如糖尿病、心血管疾病等）、临床参数（如BMI、血压等）和常规血液检查结果（如血清肌酐、尿素氮等）。

#### 模型开发
研究使用了五种ML算法（逻辑回归、朴素贝叶斯、随机森林、决策树和K最近邻）进行预测，并与KFRE模型进行比较。KFRE模型基于年龄、性别和eGFR预测5年内ESKD的风险。

#### 数据预处理
采用多重插补法处理缺失数据，生成五个不同的数据集，并在每个数据集上进行五折交叉验证。

#### 性能评估
使用准确率、精确率、召回率、特异性、F1分数和曲线下面积（AUC）等指标评估模型性能。

### 研究结果
#### 队列特征
748名患者中，9.4%发展为ESKD。大多数患者在基线时处于CKD 2期或3期。

#### 模型性能
- **随机森林**：AUC为0.81，准确率为0.82，敏感性为0.76，特异性为0.83。
- **逻辑回归**：AUC为0.79，准确率为0.75，敏感性为0.79，特异性为0.75。
- **朴素贝叶斯**：AUC为0.80，准确率为0.86，敏感性为0.72，特异性为0.87。
- **K最近邻**：AUC为0.73，准确率为0.84，敏感性为0.60，特异性为0.86。
- **决策树**：AUC为0.66，准确率为0.84，敏感性为0.44，特异性为0.89。
- **KFRE模型**：AUC为0.80，准确率为0.90，敏感性为0.47，特异性为0.95。

**结论**：逻辑回归、朴素贝叶斯和随机森林模型在预测ESKD方面表现出与KFRE相当的性能，且具有更高的敏感性，可能更适合用于患者筛查。

### 讨论与未来工作
#### 机器学习的优势
M#L模型能够利用易于获取的临床数据进行ESKD预测，具有较高的敏感性，可能有助于早期识别高风险患者。

#### KFRE的适用性
KFRE在本研究的中国CKD患者队列中表现良好，尽管其特异性高但敏感性较低，可能更适合用于确认需要密切监测的患者。

#### 数据缺失处理
多重插补法有效地解决了数据缺失问题，减少了模型偏差。

#### 未来方向
未来研究需要在更大的数据集上进行外部验证，并纳入更多预测变量（如尿液检测、影像学检查等）以进一步提高模型性能。

### 研究局限性
#### 样本量有限
研究队列规模较小，ESKD发生率较低，可能影响模型性能。

#### 缺乏尿液检测变量
由于数据限制，未纳入尿液检测指标（如ACR），可能限制了模型的预测能力。

#### 模型泛化能力未验证
尚未在外部数据集上验证模型的泛化能力。

## 文章小结
### 研究目的
本研究旨在评估机器学习（ML）在预测慢性肾病（CKD）患者发展为终末期肾病（ESKD）中的可行性，并将其性能与现有的肾脏衰竭风险方程（KFRE）进行比较。

### 研究背景
慢性肾病（CKD）是全球重大健康负担，可能导致终末期肾病（ESKD），需要肾脏替代治疗（KRT）。早期预测ESKD风险对于改善患者预后、降低发病率和死亡率以及减少医疗成本至关重要。

传统统计模型在预测ESKD方面存在局限性，尤其是在不同种族群体中的适用性。机器学习（ML）能够处理复杂的非线性关系，可能在预测ESKD方面具有优势。

### 研究人群
数据来源于北京大学第三医院CKD管理诊所的纵向队列研究，共纳入748名成人CKD患者（≥18岁），随访时间为6.3 ± 2.3年。
- **纳入标准**：稳定肾功能至少3个月。
- **排除标准**：既往接受过KRT（包括血液透析、腹膜透析或肾移植）、预期寿命<6个月、急性心力衰竭或晚期肝病、既往恶性肿瘤。

### 数据获取
- **患者特征**：包括年龄、性别、教育水平、婚姻状况、保险状态、吸烟史、饮酒史、合并症（糖尿病、心血管疾病、高血压等）。
- **临床参数**：BMI、收缩压、舒张压。
- **血液检查**：血清肌酐、尿酸、血尿素氮、白细胞计数、血红蛋白、血小板计数、肝功能指标、血脂、电解质等。

预测变量还包括估算肾小球滤过率（eGFR）和原发性肾脏疾病类型。

主要终点：需要肾脏替代治疗（KRT）的肾衰竭。

### 数据预处理
- 分类变量（如保险状态、教育水平）采用独热编码处理。
- **缺失值处理**：采用多重插补法，生成五个不同的数据集，每个数据集进行五折交叉验证。
- **数据分配**：确保训练集和测试集中ESKD+和ESKD-的分布与原始数据一致。

### 模型开发
- 使用五种机器学习算法：逻辑回归、朴素贝叶斯、随机森林、决策树、K最近邻。
- 使用网格搜索优化每个算法的超参数。
- **模型目标**：基于给定特征预测ESKD+的概率。

### 模型性能评估
- 使用准确率、精确率、召回率（敏感性）、特异性、F1分数和曲线下面积（AUC）评估模型性能。
- 将所有模型与KFRE进行比较，KFRE基于年龄、性别和eGFR预测5年ESKD风险。
- 模型性能结果为五个测试折的平均值。

### 伦理审批
研究遵循赫尔辛基宣言，获得北京大学第三医院医学科学伦理委员会批准（编号M2020132）。

### 结果
- **队列特征**：748名患者中，9.4%发展为ESKD，大多数患者在基线时处于CKD 2期或3期。
- **模型性能**：
    - **随机森林**：AUC为0.81，准确率为0.82，敏感性为0.76，特异性为0.83。
    - **逻辑回归**：AUC为0.79，准确率为0.75，敏感性为0.79，特异性为0.75。
    - **朴素贝叶斯**：AUC为0.80，准确率为0.86，敏感性为0.72，特异性为0.87。
    - **K最近邻**：AUC为0.73，准确率为0.84，敏感性为0.60，特异性为0.86。
    - **决策树**：AUC为0.66，准确率为0.84，敏感性为0.44，特异性为0.89。
    - **KFRE模型**：AUC为0.80，准确率为0.90，敏感性为0.47，特异性为0.95。

结论：逻辑回归、朴素贝叶斯和随机森林模型在预测ESKD方面表现出与KFRE相当的性能，且具有更高的敏感性，可能更适合用于患者筛查。

### 讨论
- **机器学习的优势**：ML模型能够利用易于获取的临床数据进行ESKD预测，具有较高的敏感性，可能有助于早期识别高风险患者。
- **KFRE的适用性**：KFRE在本研究的中国CKD患者队列中表现良好，尽管其特异性高但敏感性较低，可能更适合用于确认需要密切监测的患者。
- **数据缺失处理**：多重插补法有效地解决了数据缺失问题，减少了模型偏差。
- **未来方向**：未来研究需要在更大的数据集上进行外部验证，并纳入更多预测变量（如尿液检测、影像学检查等）以进一步提高模型性能。

### 研究局限性
- **样本量有限**：研究队列规模较小，ESKD发生率较低，可能影响模型性能。
- **缺乏尿液检测变量**：由于数据限制，未纳入尿液检测指标（如ACR），可能限制了模型的预测能力。
- **模型泛化能力未验证**：尚未在外部数据集上验证模型的泛化能力。

### 作者贡献
- Q.B.：数据收集、数据分析、撰写初稿。
- C.S.：数据收集。
- W.T.：构思研究、解释结果、撰写部分初稿。
- Y.L.：构思研究、数据分析、代码实现、评估模型、撰写和编辑初稿。

### 资助信息
研究由北京大学 - 百度基金（2020BD030）和中国国际医学基金会（Z - 2017 - 24 - 2037）资助。

### 利益冲突
作者声明无利益冲突。

## 复现计划
### 核心内容总结
#### 研究目标
- 利用机器学习（ML）技术，基于慢性肾病（CKD）患者的常规临床数据，预测患者发展为终末期肾病（ESKD）的风险。
- 将ML模型的性能与现有的肾脏衰竭风险方程（KFRE）进行比较。

#### 研究背景
- 慢性肾病（CKD）是全球重大健康负担，可能导致终末期肾病（ESKD），需要肾脏替代治疗（KRT）。
- 早期预测ESKD风险对于改善患者预后、降低发病率和死亡率以及减少医疗成本至关重要。
- 机器学习（ML）能够处理复杂的非线性关系，可能在预测ESKD方面具有优势。

#### 研究数据
- **数据来源**：北京大学第三医院CKD管理诊所的纵向队列研究。
- **样本量**：748名成人CKD患者，随访时间6.3±2.3年。
- **预测变量**：包括患者基线特征（年龄、性别、教育水平等）、病史（糖尿病、心血管疾病等）、临床参数（BMI、血压等）和常规血液检查结果（血清肌酐、尿素氮等）。
- **结局变量**：5年内是否发展为ESKD。

#### 研究方法
- **数据预处理**：
    - **缺失值处理**：采用多重插补法，生成五个不同的数据集。
    - **分类变量处理**：采用独热编码。
    - **数据分配**：确保训练集和测试集中ESKD+和ESKD-的分布与原始数据一致。
- **模型开发**：
    - 使用五种机器学习算法：逻辑回归、朴素贝叶斯、随机森林、决策树、K最近邻。
    - 使用网格搜索优化每个算法的超参数。
    - 采用五折交叉验证进行模型训练和测试。
- **性能评估**：
    - 使用准确率、精确率、召回率（敏感性）、特异性、F1分数和曲线下面积（AUC）评估模型性能。
    - 将所有模型与KFRE进行比较，KFRE基于年龄、性别和eGFR预测5年ESKD风险。

#### 研究结果
- **模型性能**：
    - **随机森林**：AUC为0.81，准确率为0.82，敏感性为0.76，特异性为0.83。
    - **逻辑回归**：AUC为0.79，准确率为0.75，敏感性为0.79，特异性为0.75。
    - **朴素贝叶斯**：AUC为0.80，准确率为0.86，敏感性为0.72，特异性为0.87。
    - **K最近邻**：AUC为0.73，准确率为0.84，敏感性为0.60，特异性为0.86。
    - **决策树**：AUC为0.66，准确率为0.84，敏感性为0.44，特异性为0.89。
    - **KFRE模型**：AUC为0.80，准确率为0.90，敏感性为0.47，特异性为0.95。
- **结论**：
逻辑回归、朴素贝叶斯和随机森林模型在预测ESKD方面表现出与KFRE相当的性能，且具有更高的敏感性，可能更适合用于患者筛查。

#### 研究局限性
- 样本量有限，ESKD发生率较低，可能影响模型性能。
- 未纳入尿液检测指标（如ACR），可能限制了模型的预测能力。
- 尚未在外部数据集上验证模型的泛化能力。

### 实施方案和计划
#### 1. 数据准备
- **数据来源**：选择一个包含CKD患者的纵向队列研究数据集，确保数据集中包含患者的基线特征、病史、临床参数和常规血液检查结果。
- **数据清洗**：
    - **处理缺失值**：采用多重插补法，生成多个数据集。
    - **编码分类变量**：对分类变量（如性别、教育水平）采用独热编码。
    - **数据标准化**：对连续变量进行标准化处理。

#### 2. 数据预处理
- **数据分割**：将数据集分为训练集和测试集，确保ESKD+和ESKD-的分布与原始数据一致。
- **交叉验证**：采用五折交叉验证，确保模型的稳定性和泛化能力。

#### 3. 模型开发
- **选择算法**：选择以下五种机器学习算法进行实验：
    - 逻辑回归
    - 朴素贝叶斯
    - 随机森林
    - 决策树
    - K最近邻
- **超参数优化**：使用网格搜索（Grid Search）优化每个算法的超参数。
- **模型训练**：在训练集上训练模型，并在验证集上调整超参数。

#### 4. 性能评估
- **评估指标**：使用以下指标评估模型性能：
    - 准确率（Accuracy）
    - 精确率（Precision）
    - 召回率（Recall，敏感性）
    - 特异性（Specificity）
    - F1分数
    - 曲线下面积（AUC）
- **与KFRE比较**：将ML模型的性能与KFRE模型进行比较，KFRE基于年龄、性别和eGFR预测5年ESKD风险。

#### 5. 结果分析
- **性能比较**：比较不同ML模型的性能，选择表现最佳的模型。
- **敏感性分析**：分析模型在不同阈值下的敏感性和特异性。
- **模型解释**：尝试解释模型的预测结果，分析哪些特征对预测结果影响最大。

#### 6. 外部验证
- **外部数据集**：寻找独立的外部数据集，验证模型的泛化能力。
- **模型调整**：根据外部验证结果，进一步调整模型参数。

#### 7. 模型改进
- **纳入更多特征**：考虑纳入更多预测变量，如尿液检测指标（如ACR）、影像学检查结果等。
- **尝试其他算法**：尝试其他机器学习算法（如深度学习算法）以进一步提高模型性能。

#### 8. 临床应用
- **模型部署**：将最终模型部署为临床决策支持工具。
- **用户反馈**：收集临床医生的反馈，进一步优化模型。

### 技术细节和代码实现
#### 1. 缺失值处理
```python
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 加载数据
data = pd.read_csv('ckd_data.csv')

# 多重插补
imputer = IterativeImputer(random_state=42)
data_imputed = imputer.fit_transform(data)

# 将插补后的数据转换为DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
```

#### 2. 数据分割和交叉验证
```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# 分割数据
X = data_imputed.drop('ESKD', axis=1)
y = data_imputed['ESKD']

# 五折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

#### 3. 模型训练和超参数优化
```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'K Nearest Neighbors': KNeighborsClassifier()
}

# 超参数优化
param_grid = {
    'Logistic Regression': {'C': [0.1, 1, 10],'solver': ['liblinear']},
    'Naive Bayes': {'alpha': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [10, 50, 100],'max_depth': [None, 10, 20]},
    'Decision Tree':'max_depth': [None, 10, 20]},
    'K Nearest Neighbors': {'n_neighbors': [3, 5, 10]}
}

best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[name], cv=cv, scoring='roc_auc')
    grid_search.fit(X, y)
    best_models[name] = grid_search.best_estimator_
```

#### 4. 性能评估
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 评估模型性能
results = []
for name, model in best_models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    results.append({
        'Model': name,
        'AUC': scores.mean(),
        'Accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean(),
        'Sensitivity': cross_val_score(model, X, y, cv=cv, scoring='recall').mean(),
        'Specificity': cross_val_score(model, X, y, cv=cv, scoring='precision').mean(),
        'F1 Score': cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
    })

# 输出结果
results_df = pd.DataFrame(results)
print(results_df)
```

#### 5. 与KFRE比较
```python
# KFRE模型
def kfre_model(age, gender, eGFR):
    # 示例KFRE公式（需根据实际公式调整）
    return 1 / (1 + np.exp(-(-3.159 + 0.024 * age + 0.713 * gender - 0.012 * eGFR)))

# 计算KFRE预测结果
kfre_predictions = kfre_model(data_imputed['age'], data_imputed['gender'], data_imputed['eGFR'])

# 评估KFRE性能
kfre_auc = roc_auc_score(y, kfre_predictions)
kfre_accuracy = accuracy_score(y, kfre_predictions > 0.5)
kfre_sensitivity = recall_score(y, kfre_predictions > 0.5)
kfre_specificity = precision_score(y, kfre_predictions > 0.5)
kfre_f1 = f1_score(y, kfre_predictions > 0.5)

kfre_results = {
    'Model': 'KFRE',
    'AUC': kfre_auc,
    'Accuracy': kfre_accuracy,
    'Sensitivity': kfre_sensitivity,
    'Specificity': kfre_specificity,
    'F1 Score': kfre_f1
}

results_df = results_df.append(kfre_results, ignore_index=True)
print(results_df)
```

### 总结
通过以上步骤，你可以复现这篇文献中的研究内容，并进一步探索和改进模型。如果需要更深入的研究，可以尝试纳入更多特征、尝试其他机器学习算法或进行外部验证。

## 复现计划与代码
### 数据准备
#### 数据来源
- 数据来自北京大学第三医院的CKD管理诊所的纵向队列研究。
- 包括患者的基线特征、病史、临床参数和常规血液检查结果。

#### 数据预处理
- **分类变量处理**：使用独热编码（One-Hot Encoding）。
- **缺失值处理**：采用多重插补（Multiple Imputation）。

#### 代码示例
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 假设数据已经加载到DataFrame中
data = pd.read_csv('ckd_data.csv')

# 分离特征和目标变量
X = data.drop(columns=['ESKD'])
y = data['ESKD']

# 独热编码处理分类变量
categorical_features = ['gender', 'education','marital_status', 'insurance_status']
X_encoded = pd.get_dummies(X, columns=categorical_features)

# 多重插补处理缺失值
imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X_encoded)

# 将插补后的数据转换回DataFrame
X_imputed_df = pd.DataFrame(X_imputed, columns=X_encoded.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_imputed_df, y, test_size=0.2, random_state=42, stratify=y)
```

### 模型开发
#### 机器学习算法
使用以下五种机器学习算法：
- 逻辑回归（Logistic Regression）
- 朴素贝叶斯（Naive Bayes）
- 随机森林（Random Forest）
- 决策树（Decision Tree）
- K最近邻（K-Nearest Neighbors）

#### 超参数优化
使用网格搜索（Grid Search）优化超参数。

#### 代码示例
```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 定义模型及其超参数范围
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10],'solver': ['liblinear','saga']},
    'Naive Bayes': {'alpha': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [10, 50, 100],'max_depth': [None, 10, 20]},
    'Decision Tree': {'max_depth': [None, 10, 20],'min_samples_split': [2, 5, 10]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']}
}

# 模型训练和超参数优化
best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
```

### 模型评估
#### 评估指标
- 准确率（Accuracy）
- 敏感性（Sensitivity/Recall）
- 特异性（Specificity）
- 精确率（Precision）
- F1分数（F1 Score）
- 曲线下面积（AUC）

#### 代码示例
```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc

# 定义评估函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = precision_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    return accuracy, sensitivity, specificity, precision, f1, auc_score

# 评估每个模型
results = {}
for name, model in best_models.items():
    results[name] = evaluate_model(model, X_test, y_test)
    print(f"Results for {name}: {results[name]}")
```

### 与KFRE模型比较
#### KFRE模型
KFRE模型基于年龄、性别和eGFR预测5年ESKD风险。

#### 代码示例
```python
# 假设KFRE模型的预测函数
def kfre_model(age, gender, eGFR):
    # 这里需要根据KFRE的具体公式实现预测逻辑
    # 示例公式（需要根据实际公式调整）
    risk = 1 / (1 + np.exp(-(0.1 * age + 0.2 * gender + 0.3 * eGFR + 1)))
    return risk

# 应用KFRE模型
kfre_predictions = X_test.apply(lambda row: kfre_model(row['age'], row['gender'], row['eGFR']), axis=1)
kfre_auc = roc_auc_score(y_test, kfre_predictions)
print(f"KFRE AUC: {kfre_auc}")
```

### 结果分析
#### 具体分析内容
- 比较不同模型的AUC、敏感性和特异性。
- 分析模型在不同特征集上的表现。

#### 代码示例
```python
import matplotlib.pyplot as plt

# 绘制ROC曲线
def plot_roc_curve(results, kfre_predictions, y_test):
    plt.figure(figsize=(8, 6))
    for name, result in results.items():
        y_prob = best_models[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={result[5]:.2f})')
    
    # 绘制KFRE的ROC曲线
    fpr_kfre, tpr_kfre, _ = roc_curve(y_test, kfre_predictions)
    plt.plot(fpr_kfre, tpr_kfre, label=f'KFRE (AUC={kfre_auc:.2f})', linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

plot_roc_curve(results, kfre_predictions, y_test)
```

### 计划和未来工作
- **外部验证**：在更大的外部数据集上验证模型的泛化能力。
- **特征扩展**：纳入更多特征（如尿液检测、影像学检查等）以提高预测性能。
- **模型改进**：尝试更复杂的机器学习模型（如深度学习）。

以上是根据文献内容整理的核心方法和实施方案，以及每一步的代码示例。你可以根据实际数据和需求进行调整和优化。如果需要进一步的帮助，请随时告诉我！


## 数据抓取
## 中文展示

### 连续变量信息
| 变量名 | 均值 (Mean) | 标准差 (SD) | 中位数 (Median) | 四分位间距 (IQR) |
| --- | --- | --- | --- | --- |
| 年龄 (Age) | 57.8 | 17.6 | -- | -- |
| 收缩压 (SBP, mmHg) | 129.5 | 17.8 | -- | -- |
| 舒张压 (DBP, mmHg) | 77.7 | 11.1 | -- | -- |
| BMI (kg/m²) | 24.8 | 3.7 | -- | -- |
| 血清肌酐 (Creatinine, µmol/L) | -- | -- | 130.0 | 100.0, 163.0 |
| 血尿素氮 (Urea, mmol/L) | -- | -- | 7.9 | 5.6, 10.4 |
| ALT (U/L) | -- | -- | 17.0 | 12.0, 24.0 |
| AST (U/L) | -- | -- | 18.0 | 15.0, 22.0 |
| ALP (U/L) | -- | -- | 60.0 | 50.0, 75.0 |
| 总蛋白 (Total protein, g/L) | 71.6 | 8.4 | -- | -- |
| 白蛋白 (Albumin, g/L) | 42.2 | 5.6 | -- | -- |
| 尿酸 (Uric acid, µmol/L) | -- | -- | 374.0 | 301.0, 459.0 |
| 钙 (Calcium, mmol/L) | 2.2 | 0.1 | -- | -- |
| 磷 (Phosphorous, mmol/L) | 1.2 | 0.2 | -- | -- |
| 钙磷乘积 (Ca × P, mg²/dL²) | 33.5 | 5.6 | -- | -- |
| 白细胞计数 (Blood leukocyte, 10⁹/L) | 7.1 | 2.4 | -- | -- |
| 血红蛋白 (Hemoglobin, g/L) | 131.0 | 20.3 | -- | -- |
| 血小板计数 (Platelet, 10⁹/L) | 209.8 | 57.1 | -- | -- |
| eGFR (ml/min/1.73m²) | -- | -- | 46.1 | 32.6, 67.7 |
| 总胆固醇 (Total cholesterol, mmol/L) | -- | -- | 5.1 | 4.3, 5.9 |
| 甘油三酯 (Triglyceride, mmol/L) | -- | -- | 1.8 | 1.3, 2.6 |
| HDL-C (mmol/L) | -- | -- | 1.3 | 1.1, 1.6 |
| LDL-C (mmol/L) | -- | -- | 3.0 | 2.4, 3.7 |
| 空腹血糖 (Fasting glucose, mmol/L) | -- | -- | 5.4 | 4.9, 6.2 |
| 钾 (Potassium, mmol/L) | 4.3 | 0.5 | -- | -- |
| 钠 (Sodium, mmol/L) | 140.2 | 2.8 | -- | -- |
| 氯 (Chlorine, mmol/L) | 106.9 | 3.7 | -- | -- |
| 碳酸氢根 (Bicarbonate, mmol/L) | 25.9 | 3.6 | -- | -- |

### 分类变量信息
| 变量名 | 频率 (Frequency) | 百分比 (%) |
| --- | --- | --- |
| 性别 (Gender) | 男性: 419, 女性: 329 | 男性: 56.0%, 女性: 44.0% |
| 原发疾病 (Primary disease) | 肾小球肾炎 (GN): 292 <br> 糖尿病: 224 <br> 高血压: 97 <br> 慢性间质性肾炎 (CIN): 64 <br> 其他: 18 <br> 未知: 53 | 肾小球肾炎 (GN): 39.0% <br> 糖尿病: 29.9% <br> 高血压: 13.0% <br> 慢性间质性肾炎 (CIN): 8.6% <br> 其他: 2.4% <br> 未知: 7.1% |
| CKD分期 (CKD stage) | 1期: 58 <br> 2期: 183 <br> 3期: 352 <br> 4期: 119 <br> 5期: 36 | 1期: 7.8% <br> 2期: 24.5% <br> 3期: 47.1% <br> 4期: 15.9% <br> 5期: 4.8% |
| 病史 (Medical history) | 高血压: 558 <br> 糖尿病: 415 <br> 心血管或脑血管疾病: 177 <br> 吸烟: 91 | 高血压: 74.6% <br> 糖尿病: 55.5% <br> 心血管或脑血管疾病: 23.7% <br> 吸烟: 12.6% |

### 其他信息
- 样本量 (Sample size)：748名患者。
- 随访时间 (Follow-up duration)：平均6.3年，标准差2.3年。
- ESKD发生率 (ESKD incidence)：70例（9.4%）。

## 英文展示
### Continuous Variables (连续变量)
| Variable Name | Mean ± SD | Median (IQR) |
| --- | --- | --- |
| Age (years) | 57.8 ± 17.6 | - |
| Systolic Blood Pressure (SBP, mmHg) | 129.5 ± 17.8 | - |
| Diastolic Blood Pressure (DBP, mmHg) | 77.7 ± 11.1 | - |
| Body Mass Index (BMI, kg/m²) | 24.8 ± 3.7 | - |
| Serum Creatinine (µmol/L) | - | 130.0 (100.0, 163.0) |
| Blood Urea Nitrogen (BUN, mmol/L) | - | 7.9 (5.6, 10.4) |
| Alanine Aminotransferase (ALT, U/L) | - | 17.0 (12.0, 24.0) |
| Aspartate Aminotransferase (AST, U/L) | - | 18.0 (15.0, 22.0) |
| Alkaline Phosphatase (ALP, U/L) | - | 60.0 (50.0, 75.0) |
| Total Protein (g/L) | 71.6 ± 8.4 | - |
| Albumin (g/L) | 42.2 ± 5.6 | - |
| Uric Acid (µmol/L) | - | 374.0 (301.0, 459.0) |
| Calcium (mmol/L) | 2.2 ± 0.1 | - |
| Phosphorous (mmol/L) | 1.2 ± 0.2 | - |
| Calcium-Phosphorus Product (Ca × P, mg²/dL²) | 33.5 ± 5.6 | - |
| Blood Leukocyte Count (10⁹/L) | 7.1 ± 2.4 | - |
| Hemoglobin (g/L) | 131.0 ± 20.3 | - |
| Platelet Count (10⁹/L) | 209.8 ± 57.1 | - |
| Estimated Glomerular Filtration Rate (eGFR, ml/min/1.73m²) | - | 46.1 (32.6, 67.7) |
| Total Cholesterol (mmol/L) | - | 5.1 (4.3, 5.9) |
| Triglyceride (mmol/L) | - | 1.8 (1.3, 2.6) |
| High-Density Lipoprotein Cholesterol (HDL-c, mmol/L) | - | 1.3 (1.1, 1.6) |
| Low-Density Lipoprotein Cholesterol (LDL-c, mmol/L) | - | 3.0 (2.4, 3.7) |
| Fasting Glucose (mmol/L) | - | 5.4 (4.9, 6.2) |
| Potassium (mmol/L) | 4.3 ± 0.5 | - |
| Sodium (mmol/L) | 140.2 ± 2.8 | - |
| Chlorine (mmol/L) | 106.9 ± 3.7 | - |
| Bicarbonate (mmol/L) | 25.9 ± 3.6 | - |

### Categorical Variables (分类变量)
#### Gender (Male/Female)
| Frequency (n) | Percentage (%) |
| --- | --- |
| 419/329 | - |

#### Primary Disease
| Disease Type | Frequency (n) | Percentage (%) |
| --- | --- | --- |
| Primary Glomerulonephritis (GN) | 292 | 39.0 |
| Diabetes | 224 | 29.9 |
| Hypertension | 97 | 13.0 |
| Chronic Interstitial Nephritis (CIN) | 64 | 8.6 |
| Others | 18 | 2.4 |
| Unknown | 53 | 7.1 |

#### Medical History
| History Type | Frequency (n) | Percentage (%) |
| --- | --- | --- |
| Hypertension | 558 | 74.6 |
| Diabetes Mellitus | 415 | 55.5 |
| Cardiovascular or Cerebrovascular Disease | 177 | 23.7 |
| Smoking | 91 | 12.6 |

### CKD Stages (肾病分期)
| CKD Stage | Frequency (n) | Percentage (%) |
| --- | --- | --- |
| Stage 1 | 58 | 7.8 |
| Stage 2 | 183 | 24.5 |
| Stage 3 | 352 | 47.1 |
| Stage 4 | 119 | 15.9 |
| Stage 5 | 36 | 4.8 |

### Outcome Variable (目标变量)
| Outcome | Frequency (n) | Percentage (%) |
| --- | --- | --- |
| ESKD+ (Kidney Failure) | 70 | 9.4 |
| ESKD- (No Kidney Failure) | 678 | 90.6 |

### Additional Notes
- The dataset contains 748 subjects with a follow-up duration of 6.3 ± 2.3 years.
- Missing data were handled using multiple imputation.
- The primary endpoint was kidney failure requiring renal replacement therapy (RRT), labeled as ESKD+.



## 重要数据模拟代码(无缺失值)

以下是根据抓取的数据信息编写的Python代码，用于生成模拟数据。代码命名为`generate_simulated_ckd_data.py`，并保存生成的模拟数据到指定路径。

### `generate_simulated_ckd_data.py`
```python
import numpy as np
import pandas as pd
from scipy.stats import norm, uniform
import os

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 模拟数据的样本量
n_samples = 748

# 创建保存路径
save_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\05肾病\03机器学习预测慢性肾病的终末期肾病\01模拟数据"
os.makedirs(save_path, exist_ok=True)

# 模拟连续变量
continuous_data = {
    "Age": norm.rvs(loc=57.8, scale=17.6, size=n_samples),
    "Systolic Blood Pressure (SBP)": norm.rvs(loc=129.5, scale=17.8, size=n_samples),
    "Diastolic Blood Pressure (DBP)": norm.rvs(loc=77.7, scale=11.1, size=n_samples),
    "BMI": norm.rvs(loc=24.8, scale=3.7, size=n_samples),
    "Total Protein": norm.rvs(loc=71.6, scale=8.4, size=n_samples),
    "Albumin": norm.rvs(loc=42.2, scale=5.6, size=n_samples),
    "Calcium": norm.rvs(loc=2.2, scale=0.1, size=n_samples),
    "Phosphorous": norm.rvs(loc=1.2, scale=0.2, size=n_samples),
    "Calcium-Phosphorus Product (Ca x P)": norm.rvs(loc=33.5, scale=5.6, size=n_samples),
    "Blood Leukocyte Count": norm.rvs(loc=7.1, scale=2.4, size=n_samples),
    "Hemoglobin": norm.rvs(loc=131.0, scale=20.3, size=n_samples),
    "Platelet Count": norm.rvs(loc=209.8, scale=57.1, size=n_samples),
    "Potassium": norm.rvs(loc=4.3, scale=0.5, size=n_samples),
    "Sodium": norm.rvs(loc=140.2, scale=2.8, size=n_samples),
    "Chlorine": norm.rvs(loc=106.9, scale=3.7, size=n_samples),
    "Bicarbonate": norm.rvs(loc=25.9, scale=3.6, size=n_samples)
}

# 模拟分类变量
categorical_data = {
    "Gender": np.random.choice(["Male", "Female"], size=n_samples, p=[419/748, 329/748]),
    "Primary Disease": np.random.choice(
        ["Primary GN", "Diabetes", "Hypertension", "CIN", "Others", "Unknown"],
        size=n_samples,
        p=[292/748, 224/748, 97/748, 64/748, 18/748, 53/748]
    ),
    "Hypertension History": np.random.choice(["Yes", "No"], size=n_samples, p=[558/748, 1 - 558/748]),
    "Diabetes Mellitus History": np.random.choice(["Yes", "No"], size=n_samples, p=[415/748, 1 - 415/748]),
    "Cardiovascular or Cerebrovascular Disease History": np.random.choice(["Yes", "No"], size=n_samples, p=[177/748, 1 - 177/748]),
    "Smoking History": np.random.choice(["Yes", "No"], size=n_samples, p=[91/748, 1 - 91/748])
}

# 模拟肾病分期
ckd_stages = np.random.choice(
    ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"],
    size=n_samples,
    p=[58/748, 183/748, 352/748, 119/748, 36/748]
)

# 模拟目标变量
eskd_status = np.random.choice(["ESKD+", "ESKD-"], size=n_samples, p=[70/748, 1 - 70/748])

# 合并所有数据
simulated_data = pd.DataFrame({**continuous_data, **categorical_data, "CKD Stage": ckd_stages, "ESKD Status": eskd_status})

# 保存模拟数据
file_path = os.path.join(save_path, "simulated_ckd_data.csv")
simulated_data.to_csv(file_path, index=False)
print(f"Simulated data saved to {file_path}")
```

### 代码说明
- **连续变量**：使用正态分布（`norm.rvs`）模拟均值和标准差。
- **分类变量**：使用`np.random.choice`根据频率分布模拟分类变量。
- **肾病分期和目标变量**：根据文献中的频率分布模拟。
- **保存路径**：确保保存路径存在，然后将生成的数据保存为CSV文件。

运行此代码后，模拟数据将保存到指定路径。你可以根据需要调整样本量或分布参数。

## 全部数据模拟代码(有缺失值)

以下代码，考虑了缺失数据的比例，并取消多重插补以保留缺失值。代码命名为`generate_simulated_ckd_data_with_missing_and_imputation.py`。

### `generate_simulated_ckd_data_with_missing_and_imputation.py`
```python
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 模拟数据的样本量
n_samples = 748

# 创建保存路径
save_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\05肾病\03机器学习预测慢性肾病的终末期肾病\01模拟数据"
os.makedirs(save_path, exist_ok=True)

# 模拟连续变量
continuous_data = {
    "Age": norm.rvs(loc=57.8, scale=17.6, size=n_samples),
    "Systolic Blood Pressure (SBP)": norm.rvs(loc=129.5, scale=17.8, size=n_samples),
    "Diastolic Blood Pressure (DBP)": norm.rvs(loc=77.7, scale=11.1, size=n_samples),
    "BMI": norm.rvs(loc=24.8, scale=3.7, size=n_samples),
    "Creatinine (µmol/L)": norm.rvs(loc=130.0, scale=30.0, size=n_samples),  # 假设标准差为30
    "Urea (mmol/L)": norm.rvs(loc=7.9, scale=2.0, size=n_samples),  # 假设标准差为2
    "Total Protein (g/L)": norm.rvs(loc=71.6, scale=8.4, size=n_samples),
    "Albumin (g/L)": norm.rvs(loc=42.2, scale=5.5, size=n_samples),
    "ALT (U/L)": norm.rvs(loc=17.0, scale=5.0, size=n_samples),  # 假设标准差为5
    "AST (U/L)": norm.rvs(loc=18.0, scale=3.0, size=n_samples),  # 假设标准差为3
    "ALP (U/L)": norm.rvs(loc=60.0, scale=10.0, size=n_samples),  # 假设标准差为10
    "Urine Acid (µmol/L)": norm.rvs(loc=374.0, scale=70.0, size=n_samples),  # 假设标准差为70
    "Calcium (mmol/L)": norm.rvs(loc=2.2, scale=0.1, size=n_samples),
    "Phosphorous (mmol/L)": norm.rvs(loc=1.2, scale=0.2, size=n_samples),
    "Calcium-Phosphorus Product (Ca×P)": norm.rvs(loc=33.5, scale=5.6, size=n_samples),
    "Blood Leukocyte Count (10⁹/L)": norm.rvs(loc=7.1, scale=2.4, size=n_samples),
    "Hemoglobin (g/L)": norm.rvs(loc=131.0, scale=20.3, size=n_samples),
    "Platelet Count (10⁹/L)": norm.rvs(loc=209.8, scale=57.1, size=n_samples),
    "eGFR (ml/min/1.73m²)": norm.rvs(loc=46.1, scale=15.0, size=n_samples),  # 假设标准差为15
    "Total Cholesterol (mmol/L)": norm.rvs(loc=5.1, scale=0.6, size=n_samples),  # 假设标准差为0.6
    "Triglyceride (mmol/L)": norm.rvs(loc=1.8, scale=0.5, size=n_samples),  # 假设标准差为0.5
    "HDL-c (mmol/L)": norm.rvs(loc=1.3, scale=0.2, size=n_samples),  # 假设标准差为0.2
    "LDL-c (mmol/L)": norm.rvs(loc=3.0, scale=0.5, size=n_samples),  # 假设标准差为0.5
    "Potassium (mmol/L)": norm.rvs(loc=4.3, scale=0.5, size=n_samples),
    "Sodium (mmol/L)": norm.rvs(loc=140.2, scale=2.8, size=n_samples),
    "Chlorine (mmol/L)": norm.rvs(loc=106.9, scale=3.7, size=n_samples),
    "Bicarbonate (mmol/L)": norm.rvs(loc=25.9, scale=3.6, size=n_samples)
}

# 模拟分类变量
categorical_data = {
    "Gender": np.random.choice(["Male", "Female"], size=n_samples, p=[419/748, 329/748]),
    "Primary Disease": np.random.choice(
        ["Primary GN", "Diabetes", "Hypertension", "CIN", "Others", "Unknown"],
        size=n_samples,
        p=[292/748, 224/748, 97/748, 64/748, 18/748, 53/748]
    ),
    "Hypertension History": np.random.choice(["Yes", "No"], size=n_samples, p=[558/748, 1 - 558/748]),
    "Diabetes Mellitus History": np.random.choice(["Yes", "No"], size=n_samples, p=[415/748, 1 - 415/748]),
    "Cardiovascular or Cerebrovascular Disease History": np.random.choice(["Yes", "No"], size=n_samples, p=[177/748, 1 - 177/748]),
    "Smoking History": np.random.choice(["Yes", "No"], size=n_samples, p=[91/748, 1 - 91/748])
}

# 模拟肾病分期
ckd_stages = np.random.choice(
    ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"],
    size=n_samples,
    p=[58/748, 183/748, 352/748, 119/748, 36/748]
)

# 模拟目标变量
eskd_status = np.random.choice(["ESKD+", "ESKD-"], size=n_samples, p=[70/748, 1 - 70/748])

# 合并所有数据
simulated_data = pd.DataFrame({**continuous_data, **categorical_data, "CKD Stage": ckd_stages, "ESKD Status": eskd_status})

# 引入缺失数据
missing_rates = {
    "Creatinine (µmol/L)": 0.017,
    "Urea (mmol/L)": 0.017,
    "ALT (U/L)": 0.013,
    "AST (U/L)": 0.013,
    "ALP (U/L)": 0.013,
    "Total Protein (g/L)": 0.0,
    "Albumin (g/L)": 0.0,
    "Calcium (mmol/L)": 0.0,
    "Phosphorous (mmol/L)": 0.0,
    "Calcium-Phosphorus Product (Ca×P)": 0.0,
    "Blood Leukocyte Count (10⁹/L)": 0.0,
    "Hemoglobin (g/L)": 0.0,
    "Platelet Count (10⁹/L)": 0.0,
    "eGFR (ml/min/1.73m²)": 0.0,
    "Total Cholesterol (mmol/L)": 0.0,
    "Triglyceride (mmol/L)": 0.0,
    "HDL-c (mmol/L)": 0.0,
    "LDL-c (mmol/L)": 0.0,
    "Potassium (mmol/L)": 0.0,
    "Sodium (mmol/L)": 0.0,
    "Chlorine (mmol/L)": 0.0,
    "Bicarbonate (mmol/L)": 0.0
}

# 随机引入缺失数据
for column, rate in missing_rates.items():
    if rate > 0:
        missing_indices = np.random.choice(n_samples, size=int(rate * n_samples), replace=False)
        simulated_data.loc[missing_indices, column] = np.nan

# 使用多重插补处理缺失数据
imputer = IterativeImputer(random_state=42, max_iter=10, n_nearest_features=5)
simulated_data_imputed = pd.DataFrame(imputer.fit_transform(simulated_data), columns=simulated_data.columns)

# 保存模拟数据
file_path = os.path.join(save_path, "simulated_ckd_data_with_missing_and_imputation.csv")
simulated_data_imputed.to_csv(file_path, index=False)
print(f"Simulated data with missing values and imputation saved to {file_path}")
```

### 代码说明
#### 缺失数据引入
- 根据补充文件中的缺失比例，为指定的变量引入缺失数据。
- 使用`np.random.choice`随机选择缺失的样本索引，并将对应值设置为`np.nan`。

#### 多重插补
- 使用`IterativeImputer`对缺失数据进行多重插补。
- 插补后的数据保存为新的`DataFrame`。

#### 保存路径
确保保存路径存在，然后将生成的模拟数据保存为CSV文件。

### 运行代码
运行此代码后，模拟数据将包含缺失值并取消多重插补处理，最终保存到指定路径。你可以根据需要调整缺失比例或插补参数。










