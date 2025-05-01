# 慢性肾病患者肾功能衰竭新型预测算法的验证_慢性肾病预后推理系统_PROGRES_CKD




## 文献信息
# 文献信息

| 项目 | 内容 |
| ---- | ---- |
| 标题 | Validation of a Novel Predictive Algorithm for Kidney Failure in Patients Suffering from Chronic Kidney Disease: The Prognostic Reasoning System for Chronic Kidney Disease (PROGRES-CKD) |
| 作者 | Francesco Bellocchio, Caterina Lonati, Jasmine Ion Titapiccolo, Jennifer Nadal, Heike Meiselbach, Matthias Schmid, Barbara Baerthlein, Ulrich Tschulena, Markus Schneider, Ulla T. Schultheiss, Carlo Barbieri, Christoph Moore, Sonja Steppan, Kai-Uwe Eckardt, Stefano Stuard, Luca Neri |
| 发表时间 | 2021-11-30 |
| 国家 | 德国 |
| 分区 | Q2 |
| 影响因子 | 3.345 |
| 摘要 | 本文介绍了一种新型的预测算法——慢性肾病预后推理系统（PROGRES-CKD），用于预测慢性肾病（CKD）患者发展至终末期肾病（ESKD）的风险。该算法基于朴素贝叶斯分类器（NBC），在两个独立队列中验证，显示出良好的判别能力和对缺失数据的处理能力。 |
| 关键词 | 慢性肾病（CKD）、终末期肾病（ESKD）、肾替代治疗（KRT）、风险预测、人工智能、机器学习、朴素贝叶斯分类器、精准医学 |
| 期刊名称 | International Journal of Environmental Research and Public Health |
| 卷号/期号 | 18, 12649 |
| DOI | 10.3390/ijerph182312649 |
| 研究方法 | 机器学习模型开发与验证 |
| 数据来源 | Fresenius Medical Care（FMC）NephroCare网络和德国慢性肾病研究（GCKD）队列 |
| 研究结果 | PROGRES-CKD在6个月和24个月的预测中均优于或不劣于传统预测工具KFRE，且能够处理缺失数据 |
| 研究结论 | PROGRES-CKD算法在预测慢性肾病进展至肾衰竭方面具有较高的准确性和实用性 |
| 研究意义 | 为临床决策提供新的预测工具，优化患者管理，尤其是在处理缺失数据和个性化评估方面具有优势 |
| 阅读开始时间 | 20250214 23 |
| 阅读结束时间 | 20250214 23 |
| 时刻 | 下午 |
| 星期 | 星期六 |
| 天气 | 小雨 |



## 核心内容


本文介绍了一种新型的预测算法——慢性肾病预后推理系统（PROGRES-CKD），用于预测慢性肾病（CKD）患者发展至终末期肾病（ESKD）的风险。该算法基于朴素贝叶斯分类器（NBC），旨在克服传统基于方程的风险分层算法在实际应用中的局限性，例如对缺失数据的敏感性和无法提供短期预测等问题。

### 背景知识
慢性肾病（CKD）是一种全球性健康问题，其进展至终末期肾病（ESKD）需要进行肾脏替代治疗（KRT）。早期识别高风险患者对于个性化临床决策至关重要。然而，现有的预测模型大多未在临床实践中广泛应用，部分原因是缺乏外部验证和对缺失数据的处理能力不足。

### 研究方法
- **算法设计：** PROGRES-CKD基于朴素贝叶斯分类器（NBC），利用贝叶斯定理进行概率预测。该算法分为两个版本：PROGRES-CKD-6用于预测6个月内KRT的启动风险，PROGRES-CKD-24用于预测24个月内的风险。
- **数据来源：** 训练数据来自Fresenius Medical Care（FMC）NephroCare网络的17,775名CKD患者，验证数据来自FMC独立队列（6,760名患者）和德国慢性肾病研究（GCKD）队列（4,058名患者）。
- **变量选择：** 模型纳入了28个（PROGRES-CKD-6）和34个（PROGRES-CKD-24）独立变量，包括人口统计学、肾功能指标、合并症等。
- **模型验证：** 通过计算接收者操作特征曲线下面积（ROC AUC）评估模型的判别能力，并与现有的肾脏衰竭风险方程（KFRE）进行比较。

### 实验结果
- **判别能力：**
  - 在FMC队列中，PROGRES-CKD-6的AUC为0.90（95% CI 0.88-0.91），PROGRES-CKD-24的AUC为0.85（95% CI 0.83-0.88）。
  - 在GCKD队列中，PROGRES-CKD-6的AUC为0.91（95% CI 0.86-0.97），PROGRES-CKD-24的AUC为0.85（95% CI 0.83-0.88）。
  - 与KFRE相比，PROGRES-CKD在6个月预测中表现更优，在24个月预测中表现相当。
- **缺失数据处理：** PROGRES-CKD能够在部分数据缺失的情况下进行预测，而KFRE仅能在数据完整时计算。
- **临床应用模拟：** 通过模拟研究，PROGRES-CKD-24在识别高风险患者方面优于专家评估，能够更有效地将患者分配到强化干预项目中，减少ESKD事件的发生。

### 关键结论
- **准确性：** PROGRES-CKD在两个独立队列中均表现出色，能够准确预测CKD患者在短期（6个月）和长期（24个月）内发展至ESKD的风险。
- **鲁棒性：** 与传统基于方程的预测模型相比，PROGRES-CKD能够处理缺失数据，适用于实际临床数据的复杂情况。
- **个性化评估：** PROGRES-CKD能够提供患者特定的影响指标，帮助医生根据个体特征制定干预措施。
- **临床效益：** 通过模拟研究，PROGRES-CKD能够更有效地识别高风险患者，减少不必要的医疗资源浪费，提高干预效率。

### 研究局限
尽管PROGRES-CKD在FMC和GCKD队列中表现良好，但其在其他临床实践中的适用性仍需进一步验证。此外，该算法的持续性能监测和外部验证仍在进行中。

### 总结
PROGRES-CKD作为一种新型的预测工具，能够有效预测CKD患者的肾功能衰竭风险，并在处理缺失数据和个性化评估方面具有显著优势。该算法有望改善当前CKD风险评估的标准，优化患者分层和个体化干预策略。

## 文章小结
### 1. Introduction
- **背景**：慢性肾病（CKD）的进展至终末期肾病（ESKD）需要及时识别高风险患者以进行个性化临床决策。现有的预测模型大多未在临床实践中广泛应用，主要原因是缺乏外部验证和对缺失数据的处理能力不足。
- **目的**：开发一种新型预测算法——慢性肾病预后推理系统（PROGRES-CKD），基于朴素贝叶斯分类器（NBC），用于预测CKD患者在6个月和24个月内进展至ESKD的风险。
- **优势**：与传统基于方程的预测模型（如KFRE）相比，PROGRES-CKD能够处理缺失数据，并提供短期和长期预测。

### 2. Materials and Methods
- **模型设计**：PROGRES-CKD基于朴素贝叶斯分类器（NBC），利用贝叶斯定理进行概率预测。模型分为两个版本：PROGRES-CKD-6（预测6个月内KRT启动风险）和PROGRES-CKD-24（预测24个月内KRT启动风险）。
- **数据来源**
    - 训练数据来自Fresenius Medical Care（FMC）NephroCare网络的17,775名CKD患者。
    - 验证数据来自FMC独立队列（6,760名患者）和德国慢性肾病研究（GCKD）队列（4,058名患者）。
- **变量选择**
    - PROGRES-CKD-6纳入28个独立变量，PROGRES-CKD-24纳入34个独立变量，包括人口统计学、肾功能指标、合并症等。
    - 输入变量包括人口统计学、生活方式、血液生物标志物、合并症等。
- **模型训练与验证**
    - 使用Hugin 8.5软件进行NBC权重的推导。
    - 在FMC和GCKD队列中进行外部验证，评估模型的判别能力和校准能力。

### 3. Results
- **模型判别能力**
    - 在FMC队列中，PROGRES-CKD-6的AUC为0.90（95% CI 0.88 - 0.91），PROGRES-CKD-24的AUC为0.85（95% CI 0.83 - 0.88）。
    - 在GCKD队列中，PROGRES-CKD-6的AUC为0.91（95% CI 0.86 - 0.97），PROGRES-CKD-24的AUC为0.85（95% CI 0.83 - 0.88）。
- **与KFRE的比较**
    - PROGRES-CKD在6个月预测中表现优于KFRE，在24个月预测中表现相当。
    - PROGRES-CKD能够在数据缺失的情况下进行预测，而KFRE仅能在数据完整时计算。
- **临床应用模拟**：PROGRES-CKD-24在识别高风险患者方面优于专家评估，能够更有效地将患者分配到强化干预项目中，减少ESKD事件的发生。

### 4. Discussion
- **研究意义**
    - PROGRES-CKD在两个独立队列中表现出色，能够准确预测CKD患者在短期（6个月）和长期（24个月）内进展至ESKD的风险。
    - 该算法能够处理缺失数据，适用于实际临床数据的复杂情况。
    - 提供患者特定的影响指标，帮助医生根据个体特征制定干预措施。
- **潜在优势**
    - 提高临床决策的准确性，优化患者管理。
    - 减少不必要的医疗资源浪费，提高干预效率。
- **研究局限**
    - 尽管在FMC和GCKD队列中表现良好，但其在其他临床实践中的适用性仍需进一步验证。
    - 需要持续性能监测和外部验证。

### 5. Conclusions
- **总结**
    - PROGRES-CKD作为一种新型的预测工具，能够有效预测CKD患者的肾功能衰竭风险，并在处理缺失数据和个性化评估方面具有显著优势。
    - 该算法有望改善当前CKD风险评估的标准，优化患者分层和个体化干预策略。

### Supplementary Materials
提供了ICD10代码、尿蛋白转换表、案例研究和PROGRES-CKD的图形输出等补充材料。

## 主要方法和实施计划
### 研究目标
开发一种基于朴素贝叶斯分类器（Naïve Bayes Classifier, NBC）的预测算法（PROGRES-CKD），用于预测慢性肾病（CKD）患者在6个月和24个月内进展至终末期肾病（ESKD）的风险。该算法旨在克服传统预测模型（如KFRE）在处理缺失数据和提供短期预测方面的局限性。

### 研究设计
1. **模型开发**
    - **算法选择**：PROGRES-CKD基于朴素贝叶斯分类器（NBC），这是一种基于贝叶斯定理的概率模型，假设预测变量在给定结果的情况下是条件独立的。NBC能够处理缺失数据，并通过“信息价值”（Value of Information, VOI）统计来评估缺失数据对预测结果的影响。
    - **数据来源**：训练数据来自Fresenius Medical Care（FMC）NephroCare网络的17,775名CKD患者，覆盖欧洲、南美洲和非洲的15个国家。这些数据反映了真实世界的临床实践情况。
    - **变量选择**：
        - **输入变量**：包括人口统计学特征（如年龄、性别、BMI）、肾功能指标（如eGFR、蛋白尿）、合并症（如糖尿病、高血压）、生活方式因素（如吸烟状态）等。
        - **短期预测（PROGRES-CKD-6）**：包含28个独立变量。
        - **长期预测（PROGRES-CKD-24）**：包含34个独立变量。
    - **模型训练**：使用Hugin 8.5软件进行NBC权重的推导，通过数据驱动的方法从FMC的欧洲临床数据库（EuCliD®）中提取信息。
2. **模型验证**
    - **验证队列**：
        - **FMC独立队列**：从FMC NephroCare网络中随机抽取的30%数据（6,760名患者）用于验证。
        - **德国慢性肾病研究（GCKD）队列**：包含4,058名患者，这是一个前瞻性观察性研究，覆盖德国的多个学术肾脏病中心。
    - **验证方法**：
        - **判别能力评估**：通过计算接收者操作特征曲线下面积（ROC AUC）来评估模型的判别能力。AUC值大于0.70被认为是可以接受的。
        - **校准能力评估**：通过绘制校准图，比较模型预测的风险与实际观察到的结果之间的关系。
        - **与KFRE的比较**：将PROGRES-CKD的性能与现有的肾脏衰竭风险方程（KFRE）进行比较，评估其非劣性和优越性。
3. **临床应用模拟**
    - **专家评估对比**：邀请四位肾病专家对78名随机选取的CKD患者进行风险评估，将其评分与PROGRES-CKD-24的预测结果进行比较。
    - **干预效果模拟**：假设在一个包含10,000名CKD患者的虚拟队列中，使用PROGRES-CKD进行风险分层，并将高风险患者分配到强化干预项目中。评估这种策略在减少ESKD事件发生率方面的潜在效果。

### 实施计划
1. **数据准备**
    - 从FMC NephroCare网络和GCKD研究中收集数据，确保数据的完整性和代表性。
    - 对数据进行预处理，包括变量的定义、缺失值的处理等。
2. **模型开发**
    - 使用Hugin 8.5软件开发基于NBC的PROGRES-CKD模型。
    - 确定模型的输入变量，并根据训练数据集调整模型参数。
3. **模型验证**
    - 在FMC独立队列和GCKD队列中进行外部验证。
    - 计算ROC AUC值，评估模型的判别能力。
    - 绘制校准图，评估模型的校准能力。
    - 与KFRE进行比较，评估PROGRES-CKD的非劣性和优越性。
4. **临床应用评估**
    - 邀请专家对随机选取的患者进行风险评估，并与PROGRES-CKD的预测结果进行对比。
    - 进行干预效果模拟，评估PROGRES-CKD在临床实践中的潜在效益。
5. **结果分析与报告**
    - 分析模型验证和临床应用模拟的结果。
    - 撰写研究报告，总结PROGRES-CKD的性能和潜在临床应用价值。

### 研究意义
通过开发和验证PROGRES-CKD，本研究旨在提供一种能够处理缺失数据、提供短期和长期预测的新型预测工具。该工具有望改善CKD患者的临床管理，优化医疗资源分配，并为个性化医疗提供支持。

## 五、重要变量和数据

以下是根据文献中提供的信息抓取的主要变量及其统计特征，以便您在后续工作中使用Python代码模拟数据。变量分为连续变量和分类变量，以下是详细信息：

### 连续变量
|变量名称|均值 (Mean)|标准差 (SD)|中位数 (Median)|四分位间距 (IQR)|
| ---- | ---- | ---- | ---- | ---- |
|Age (years)|72.15|11.7|--|--|
|BMI (kg/m²)|30.63|10.92|--|--|
|eGFR (mL/min/1.73 m²)|31.93|13.4|--|--|
|Albumin (g/dL)|4.19|0.4|--|--|
|Ferritin (µg/L)|222.18|260.98|--|--|
|Hemoglobin (g/dL)|12.65|1.83|--|--|
|Phosphate (mg/dL)|3.65|0.74|--|--|
|Calcium (mg/dL)|9.36|0.73|--|--|
|Sodium (mmol/L)|140.17|3.16|--|--|
|PTH (ng/L)|131.84|150.12|--|--|
|ACR (mg/mmol)|138.67|568.28|--|--|
|Proteinuria (g/24h)|3.58|150.29|--|--|
|Systolic BP (mmHg)|137.33|18.41|--|--|
|CRP (mg/L)|--|4.23|7.63|--|
|Glucose (mg/dL)|126.45|48.59|--|--|
|HDL Cholesterol (mg/dL)|48.3|16.74|--|--|
|LDL Cholesterol (mg/dL)|107.59|219.29|--|--|
|Triglyceride (mg/dL)|--|142.77|95.72|--|
|hsTNT (ng/L)|--|13|11|--|
|Uric Acid (mg/dL)|6.68|1.61|--|--|

### 分类变量
|变量名称|频率 (n)|构成比 (%)|
| ---- | ---- | ---- |
|Gender (Male)|11,349|50.36|
|CKD Stage| | |
| - Stage 3|11,965|53.10|
| - Stage 4|8,026|35.62|
| - Stage 5|2,544|11.29|
|Etiology of Kidney Disease| | |
| - Diabetes|3,614|16.04|
| - Polycystic|477|2.12|
| - Hypertension|5,281|23.43|
| - Glomerulonephritis|987|4.38|
|Smoking Status| | |
| - Ex-smoker|3,502|15.54|
| - Non-smoker|10,066|44.67|
| - Smoker|2,274|10.09|
|Alcohol Consumption| | |
| - Abuse|8,636|38.32|
| - Moderate| - | - |
| - Abstinence|6,984|30.99|
|Comorbidities| | |
| - Peripheral Vascular Disease|1,875|8.32|
| - Coronary Artery Disease|4,336|19.24|
| - Congestive Heart Failure|1,887|8.37|
| - Cerebrovascular Disease|1,876|8.32|
| - Connective Tissue Disorder|399|1.77|
| - Cancer|2,469|10.96|
| - Diabetes|9,021|40.03|
| - Anemia|9,800|43.49|
| - Hypertension|17,871|79.30|
| - Atrial Fibrillation|2,337|10.37|

### 其他信息
- 文献中未提供某些变量的均值、标准差或中位数，可能是因为这些数据未在摘要或表格中明确列出。您可以参考文献的补充材料或原始数据来源以获取更详细的信息。
- 部分变量（如CRP、Triglyceride、hsTNT）仅提供了中位数和四分位间距，可能是因为这些数据的分布不完全符合正态分布。

### Python代码模拟数据示例
以下是基于上述信息的Python代码示例，用于模拟连续变量和分类变量的数据：

```python
import numpy as np
import pandas as pd

# 连续变量模拟
continuous_data = {
    'Age': np.random.normal(72.15, 11.7, 1000),
    'BMI': np.random.normal(30.63, 10.92, 1000),
    'eGFR': np.random.normal(31.93, 13.4, 1000),
    'Albumin': np.random.normal(4.19, 0.4, 1000),
    'Ferritin': np.random.normal(222.18, 260.98, 1000),
    'Hemoglobin': np.random.normal(12.65, 1.83, 1000),
    'Phosphate': np.random.normal(3.65, 0.74, 1000),
    'Calcium': np.random.normal(9.36, 0.73, 1000),
    'Sodium': np.random.normal(140.17, 3.16, 1000),
    'PTH': np.random.normal(131.84, 150.12, 1000),
    'ACR': np.random.normal(138.67, 568.28, 1000),
    'Proteinuria': np.random.normal(3.58, 150.29, 1000),
    'Systolic_BP': np.random.normal(137.33, 18.41, 1000),
    'Glucose': np.random.normal(126.45, 48.59, 1000),
    'HDL_Cholesterol': np.random.normal(48.3, 16.74, 1000),
    'LDL_Cholesterol': np.random.normal(107.59, 219.29, 1000),
    'Uric_Acid': np.random.normal(6.68, 1.61, 1000),
}

# 分类变量模拟
categorical_data = {
    'Gender': np.random.choice(['Male', 'Female'], 1000, p=[0.5036, 1 - 0.5036]),
    'CKD_Stage': np.random.choice(['Stage 3', 'Stage 4', 'Stage 5'], 1000, p=[0.531, 0.3562, 0.1129]),
    'Etiology': np.random.choice(['Diabetes', 'Polycystic', 'Hypertension', 'Glomerulonephritis', 'Other'], 
                                 1000, p=[0.1604, 0.0212, 0.2343, 0.0438, 1 - (0.1604 + 0.0212 + 0.2343 + 0.0438)]),
    'Smoking_Status': np.random.choice(['Ex-smoker', 'Non-smoker', 'Smoker'], 1000, p=[0.1554, 0.4467, 0.1009]),
    'Alcohol_Consumption': np.random.choice(['Abuse', 'Moderate', 'Abstinence'], 1000, p=[0.3832, 0.5, 0.3099]),
    'Hypertension': np.random.choice(['Yes', 'No'], 1000, p=[0.793, 1 - 0.793]),
    'Diabetes': np.random.choice(['Yes', 'No'], 1000, p=[0.4003, 1 - 0.4003]),
}

# 合并数据
data = {**continuous_data, **categorical_data}
df = pd.DataFrame(data)

# 查看模拟数据
print(df.head())
```



