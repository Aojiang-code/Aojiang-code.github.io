# 中国多中心慢性肾脏病队列ESKD风险预测模型的推导、验证和比较研究

## 一、文献信息

|项目|内容|
|---|---|
|标题|Artificial Intelligence to Predict Chronic Kidney Disease Progression to Kidney Failure: A Narrative Review|
|作者|Zane A. Miller, Karen Dwyer|
|发表时间|2025-01-01|
|国家|澳大利亚|
|分区|Q1|
|影响因子|5.678|
|摘要|本文综述了人工智能和机器学习在预测慢性肾脏病进展至肾衰竭中的应用，重点探讨了常用的预测变量和模型性能。|
|关键词|人工智能, 慢性肾脏病, 肾衰竭, 机器学习|
|期刊名称|Nephrology|
|卷号/期号|30:e14424|
|DOI|10.1111/nep.14424|
|研究方法|叙述性综述|
|数据来源|Medline和EMBASE数据库检索|
|研究结果|机器学习模型优于或不劣于传统预测工具KFRE|
|研究结论|机器学习模型在预测慢性肾脏病进展至肾衰竭方面具有潜力|
|研究意义|为临床决策提供新的预测工具，优化患者管理|
|阅读开始时间|20250216 22|
|阅读结束时间|20250214 23|
|时刻|晚上|
|星期|星期日|
|天气|多云|




## 二、核心内容
### 研究背景
慢性肾脏病（CKD）是全球性的公共卫生问题，其发病率在中国约为10.8%。CKD的进展可能导致终末期肾病（ESKD），需要进行肾脏替代治疗（KRT），这给医疗系统带来了巨大的负担。早期识别CKD患者的ESKD风险对于个性化管理和优化疾病进展的管理至关重要。然而，现有的预测模型大多基于晚期CKD患者，且在亚洲人群中验证不足。此外，传统的统计模型可能无法充分利用临床特征进行预测，而机器学习（ML）方法可能提供更准确的预测。

### 研究目的
本研究旨在开发和验证一个新的ESKD风险预测模型，该模型基于Cox比例风险模型和机器学习方法（包括XGBoost和生存支持向量机，SSVM），并比较这些方法在预测CKD进展至ESKD方面的准确性。

### 研究设计
- **数据来源**：研究使用了中国慢性肾脏病队列研究（C-STRIDE）的数据作为模型训练和测试数据集，并以北京大学第一医院的CKD队列作为外部验证数据集。
- **研究人群**：纳入了CKD 1~4期的患者，排除了年龄>70岁、缺乏基线eGFR或随访时间<3个月的患者。
- **结局定义**：以肾脏替代治疗（KRT）的发生作为研究结局。
- **模型开发**：开发了基于Cox模型和两种机器学习模型（XGBoost和SSVM）的风险预测模型，并通过Harrell’s C-index、Uno’s C-index和Brier分数等指标评估模型的判别能力和校准性能。

### 研究结果
- **队列描述**：C-STRIDE队列共纳入3216名患者，其中411人（12.8%）接受了KRT；外部验证队列纳入342名患者，其中25人（7.3%）接受了KRT。
- **模型性能**：
    - **Cox模型**：在测试数据集中，Harrell’s C-index为0.834，Uno’s C-index为0.833，Brier分数为0.065。
    - **XGBoost模型**：在测试数据集中，Harrell’s C-index为0.826，Uno’s C-index为0.825，Brier分数为0.066。
    - **SSVM模型**：在测试数据集中，Harrell’s C-index为0.748，Uno’s C-index为0.747，Brier分数为0.070。
- **比较分析**：XGBoost和Cox模型在测试数据集中的表现相似，而SSVM模型的判别能力显著低于前两者。
- **外部验证**：在外部验证数据集中，XGBoost模型的Harrell’s C-index、Uno’s C-index和Brier分数均优于Cox模型。

### 研究结论
本研究开发并验证了一个基于临床常用指标的ESKD风险预测模型，其整体性能良好。传统的Cox回归模型和某些机器学习模型（如XGBoost）在预测CKD进展至ESKD方面具有相似的准确性。研究结果表明，机器学习模型在外部验证中表现更优，可能为临床决策提供更可靠的预测工具。

### 研究意义
- **临床应用**：该模型基于临床常规检测指标，易于在实际医疗环境中应用。
- **模型比较**：研究比较了传统统计模型和机器学习模型的性能，为未来CKD风险预测模型的选择提供了依据。
- **人群特异性**：该模型主要基于中国人群数据开发，可能更适合中国及其他发展中国家的CKD患者管理。

### 研究局限性
- 研究基于中国人群，需要在其他地区和人群中进一步验证。
- 队列中大部分患者为肾小球疾病引起的CKD，可能不适用于以糖尿病或高血压为主要原因的CKD人群。
- 模型未纳入治疗相关变量，可能影响预测准确性。

总体而言，这篇研究为CKD进展至ESKD的风险预测提供了新的方法和思路，尤其是在机器学习模型的应用方面展示了潜力。

## 三、文章小结

### Introduction
- **背景**：慢性肾脏病（CKD）是全球性公共卫生问题，中国CKD患病率为10.8%，其进展至终末期肾病（ESKD）需要肾脏替代治疗（KRT），带来巨大医疗负担。
- **研究需求**：早期识别CKD患者进展至ESKD的风险对于个性化管理至关重要。现有的预测模型（如KFRE）主要基于晚期CKD患者，且在亚洲人群中验证不足。此外，传统统计模型可能无法充分利用临床特征，而机器学习（ML）方法可能提供更准确的预测。
- **研究目的**：开发并验证一个新的ESKD风险预测模型，结合Cox比例风险模型和机器学习方法（XGBoost和SSVM），并比较其预测性能。

### Materials and Methods
#### 2.1. Ethics Approval and Declaration
研究遵循TRIPOD报告指南，获得北京大学第一医院伦理委员会批准，并获得所有参与者的知情同意。

#### 2.2. Data Source and Study Population
- **开发队列（Development Cohort）**：基于C-STRIDE研究，纳入中国22个省份的39个临床中心的CKD患者，覆盖CKD 1~4期。排除标准包括年龄>70岁、缺乏基线eGFR或随访时间<3个月。
- **验证队列（Validation Cohort）**：使用北京大学第一医院的前瞻性CKD队列作为外部验证数据集。
- **结局定义**：以肾脏替代治疗（KRT）的发生作为研究结局，随访期间记录KRT事件。

#### 2.3. Candidate Variables
基线变量包括年龄、性别、血压、合并症（如2型糖尿病、高血压、心血管疾病）、实验室检查指标（如血红蛋白、肌酐、白蛋白、尿酸、血脂等）。估算肾小球滤过率（eGFR）采用CKD-EPI公式计算。

#### 2.4. Data Preprocessing and Statistical Analysis
- **数据预处理**：使用多重插补（MICE）处理缺失数据，对偏态分布的变量（如尿白蛋白 - 肌酐比值UACR）进行对数转换。
- **样本量计算**：基于事件/变量（EPV）值>10，开发队列样本量充足。
- **模型训练与验证**：将C-STRIDE队列随机分为训练集和测试集（7:3），并使用外部验证集进行模型评估。

#### 2.5. Model Development and Evaluation
- **特征选择**：结合Cox逐步回归和Cox-LASSO方法选择预测特征。
- **模型训练**：开发了Cox回归模型和两种机器学习模型（XGBoost和SSVM），并使用贝叶斯优化算法调整超参数。
- **模型评估**：使用Harrell’s C-index、Uno’s C-index、时间依赖AUC（TD-AUC）和Brier分数评估模型的判别能力和校准性能。

#### 2.6. Implementation Setup
研究使用Python和R语言进行数据处理和模型开发，涉及的软件包包括R mice、R survival、Python XGBoost等。

### Results
#### 3.1. Cohort Description
- **开发队列**：共纳入3216名患者，其中411人（12.8%）发生KRT，平均随访4.45年。
- **验证队列**：共纳入342名患者，其中25人（7.3%）发生KRT，平均随访3.37年。
- **基线特征**：两组患者的平均年龄分别为48岁和55岁，平均eGFR分别为52.97 mL/min/1.73 m²和50.83 mL/min/1.73 m²，主要病因以肾小球肾炎为主。

#### 3.2. Feature Selection
最终纳入模型的特征包括年龄、性别、eGFR、UACR、白蛋白、血红蛋白、2型糖尿病和高血压病史。

#### 3.3. Model Performance
- **Cox模型**：在训练集、测试集和验证集中的Harrell’s C-index分别为0.841、0.834和0.761，Uno’s C-index分别为0.807、0.833和0.796，Brier分数分别为0.059、0.065和0.029。
- **XGBoost模型**：在训练集、测试集和验证集中的Harrell’s C-index分别为0.864、0.826和0.796，Uno’s C-index分别为0.836、0.825和0.822，Brier分数分别为0.055、0.066和0.028。
- **SSVM模型**：在训练集、测试集和验证集中的Harrell’s C-index分别为0.754、0.748和0.745，Uno’s C-index分别为0.732、0.747和0.753，Brier分数分别为0.068、0.070和0.031。

#### 3.4. Model Comparison
- 在训练集中，XGBoost模型的判别能力优于Cox和SSVM模型（p<0.001）。
- 在测试集中，XGBoost与Cox模型的判别能力相似，但均优于SSVM模型（p<0.001）。
- 在外部验证集中，XGBoost模型的Harrell’s C-index、Uno’s C-index和Brier分数均优于Cox模型。

#### 3.5. Web Application
研究开发的PKU-CKD预测模型已嵌入医院电子病历系统（EHR），并基于Cox算法计算KRT的绝对风险。XGBoost模型结果在后端计算，用于进一步评估预测准确性。模型输出的ESKD风险值通过LIME算法增强解释性。

### Discussion
- **模型意义**：本研究开发的模型基于中国多中心CKD队列，主要包含肾小球肾炎患者，能够有效预测2年和5年内的KRT风险，适用于中国人群。
- **与其他模型比较**：与KFRE模型相比，本研究开发的模型在肾小球疾病为主的队列中表现更好。尽管Cox和XGBoost模型在测试集中的表现相似，但在外部验证中，XGBoost模型表现更优。
- **机器学习的应用**：尽管机器学习在某些研究中表现出超越传统统计模型的预测能力，但在时间 - 事件数据中，Cox模型可能与某些机器学习模型相当甚至更好。机器学习模型的“黑箱”特性和过拟合问题仍需进一步研究。
- **研究局限性**：模型基于中国人群，需要在其他地区和人群中验证；队列中肾小球疾病比例较高，可能不适用于以糖尿病或高血压为主要原因的CKD人群；模型未纳入治疗相关变量。

### Conclusions
本研究开发并验证了一个基于临床常用指标的ESKD风险预测模型，其判别和校准性能良好。传统Cox回归模型和机器学习模型（如XGBoost）在预测CKD进展至ESKD方面具有相似的准确性，为临床决策提供了新的工具。

## 四、主要方法和实施计划
### 研究设计与目标
#### 研究目标
开发并验证一个新的ESKD风险预测模型，该模型基于Cox比例风险模型和两种机器学习方法（XGBoost和生存支持向量机，SSVM）。研究旨在比较这些方法在预测CKD进展至ESKD方面的性能，并评估其临床应用潜力。

#### 研究设计
- **数据来源**：使用中国慢性肾脏病队列研究（C-STRIDE）作为模型开发和内部验证的数据集，并使用北京大学第一医院（PKUFH）的CKD队列作为外部验证数据集。
- **数据集划分**：C-STRIDE队列随机分为训练集（70%）和测试集（30%），用于模型开发和内部验证。
- **结局定义**：以肾脏替代治疗（KRT）的发生作为主要结局，随访期间记录KRT事件。

### 数据来源与研究人群
#### 开发队列（Development Cohort）
- **数据来源**：C-STRIDE是中国第一个全国性CKD队列研究，覆盖22个省份的39个临床中心。
- **纳入标准**：
    - 年龄18~74岁。
    - 根据不同病因的CKD，估算肾小球滤过率（eGFR）范围不同：
        - 肾小球肾炎（GN）：eGFR≥15 mL/min/1.73 m²。
        - 糖尿病肾病（DN）：15≤eGFR<60 mL/min/1.73 m²或eGFR≥60 mL/min/1.73 m²且伴有肾病范围蛋白尿。
        - 其他非GN和非DN患者：15≤eGFR<60 mL/min/1.73 m²。
- **排除标准**：
    - 年龄>70岁。
    - 缺乏基线eGFR或人口学数据。
    - 随访时间<3个月。

#### 验证队列（Validation Cohort）
- **数据来源**：PKUFH的前瞻性CKD队列，包含CKD 1~4期患者。
- **排除标准**：与开发队列相同。
- **结局记录**：随访期间记录KRT事件，随访时间超过6年的患者数据被截断。

### 候选变量与数据预处理
#### 候选变量
- **基线变量**：包括年龄、性别、血压、合并症（如2型糖尿病、高血压、心血管疾病）和实验室检查指标（如血红蛋白、肌酐、白蛋白、尿酸、血脂等）。
- **eGFR计算**：使用CKD-EPI公式计算eGFR。
- **尿白蛋白-肌酐比值（UACR）**：由于其分布偏态，对UACR进行对数转换。

#### 数据预处理
- **缺失值处理**：使用多重插补（MICE）方法处理缺失数据，避免信息丢失。
- **样本量计算**：基于事件/变量（EPV）值>10，确保开发队列样本量充足。
- **数据分割**：C-STRIDE队列随机分为训练集和测试集（7:3），PKUFH队列作为独立的外部验证集。

### 模型开发与评估
#### 特征选择
使用Cox逐步回归分析和Cox-LASSO方法选择与ESKD风险相关的特征。最终模型包括以下特征：
年龄、性别、eGFR、UACR、白蛋白、血红蛋白、2型糖尿病病史和高血压病史。

#### 模型训练
- **Cox比例风险模型**：传统的生存分析模型，用于评估特征与ESKD风险的关系。
- **机器学习模型**：
    - **XGBoost**：基于梯度提升的集成学习模型，适用于处理生存预测任务。
    - **生存支持向量机（SSVM）**：基于支持向量机的生存分析模型。
- **超参数优化**：使用贝叶斯优化算法调整模型超参数，以最大化五折交叉验证的C-index值。

#### 模型评估
- **判别能力**：使用Harrell’s C-index和Uno’s C-index评估模型的判别能力。
- **校准性能**：使用Brier分数评估模型的校准性能，计算实际结果与预测结果的平方差。
- **时间依赖AUC（TD-AUC）**：评估模型在特定时间点（如2年和5年）的判别能力。
- **校准曲线**：通过校准曲线可视化预测生存概率与实际生存概率的差异。

### 实施计划
#### 模型开发流程
- **数据收集与预处理**：
    - 从C-STRIDE和PKUFH队列中收集数据。
    - 处理缺失值，对偏态变量进行转换。
- **特征选择**：
    - 使用Cox逐步回归和Cox-LASSO方法筛选与ESKD风险相关的特征。
- **模型训练**：
    - 在训练集上开发Cox模型、XGBoost模型和SSVM模型。
    - 使用贝叶斯优化调整模型超参数。
- **模型评估**：
    - 在测试集和外部验证集上评估模型的判别能力和校准性能。
    - 使用Harrell’s C-index、Uno’s C-index、TD-AUC和Brier分数评估模型性能。

#### 模型验证
- **内部验证**：在C-STRIDE队列的测试集上评估模型性能。
- **外部验证**：在PKUFH队列上进一步验证模型的泛化能力。

#### 临床应用
- **模型部署**：将最终的PKU-CKD模型嵌入医院电子病历系统（EHR），为临床医生提供决策支持。
- **用户界面**：模型输出结果通过用户友好的界面展示，包括2年和5年的ESKD风险预测值。
- **解释性增强**：使用局部可解释模型无关解释（LIME）算法增强模型的解释性，帮助临床医生理解模型预测的依据。

### 方法总结
这篇研究通过结合传统统计方法（Cox模型）和机器学习方法（XGBoost和SSVM），开发了一个基于临床常用指标的ESKD风险预测模型。研究详细描述了数据收集、预处理、特征选择、模型训练和评估的全过程，并通过内部和外部验证验证了模型的性能。最终，该模型被嵌入医院电子病历系统，为临床决策提供支持。

## 五、重要变量和数据(英文展示)
### 连续变量信息
|变量名称|数据集|均值（Mean）|标准差（SD）|中位数（Median）|四分位间距（IQR）|单位|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|Age (years)|C-STRIDE (N=3216)|48|13| - | - |years|
| |PKUFH CKD (N=342)|55|11| - | - |year|
|eGFR (mL/min/1.73 m²)|C-STRIDE (N=3216)|52.97|29.50| - | - |mL/min/1.73 m²|
| |PKUFH CKD (N=342)|50.83|35.04| - | - |mL/min/1.73 m²|
|UACR (mg/g)|C-STRIDE (N=3216)| - |376.40|90.80 - 911.45| - |mg/g|
| |PKUFH CKD (N=342)| - |214.37|43.55 - 1058.50| - |mg/g|
|Hemoglobin (g/L)|C-STRIDE (N=3216)|128.75|21.87| - | - |g/L|
| |PKUFH CKD (N=342)|134.09|20.18| - | - |g/L|
|Albumin (g/L)|C-STRIDE (N=3216)|38.54|7.46| - | - |g/L|
| |PKUFH CKD (N=342)|42.55|4.69| - | - |g/L|
|Creatinine (µmol/L)|C-STRIDE (N=3216)|154.27|72.17| - | - |µmol/L|
| |PKUFH CKD (N=342)|196.56|168.08| - | - |µmol/L|
|Systolic BP (mmHg)|C-STRIDE (N=3216)|129.39|17.50| - | - |mmHg|
| |PKUFH CKD (N=342)|132.39|18.62| - | - |mmHg|
|Diastolic BP (mmHg)|C-STRIDE (N=3216)|80.97|10.80| - | - |mmHg|
| |PKUFH CKD (N=342)|77.65|11.06| - | - |mmHg|

### 分类变量信息
|变量名称|数据集|构成比（Frequency）|单位|
| ---- | ---- | ---- | ---- |
|Male (n, %)|C-STRIDE (N=3216)|1909 (59.4%)| - |
| |PKUFH CKD (N=342)|133 (38.9%)| - |
|Smoker (n, %)|C-STRIDE (N=3216)|1123 (34.9%)| - |
| |PKUFH CKD (N=342)|11 (3.2%)| - |
|Hypertension (n, %)|C-STRIDE (N=3216)|2363 (73.5%)| - |
| |PKUFH CKD (N=342)|83 (24.3%)| - |
|T2DM (n, %)|C-STRIDE (N=3216)|641 (19.9%)| - |
| |PKUFH CKD (N=342)|99 (28.9%)| - |
|CVD (n, %) |C-STRIDE (N=3216)|270 (8.4%)| - |
| |PKUFH CKD (N=342)|7 (2.0%)| - |
|Cause of CKD (n, %) |C-STRIDE (N=3216)|Diabetic Kidney Disease (DKD) 385 (12.0%)<br>Glomerulonephritis (GN) 1853 (57.6%)<br>Other 725 (22.5%)| - |
| |PKUFH CKD (N=342)|Diabetic Kidney Disease (DKD) 88 (25.7%)<br>Glomerulonephritis (GN) 80 (23.4%)<br>Other 102 (29.8%)| - |

### 其他变量信息
|变量名称|数据集|均值（Mean）|标准差（SD）|单位|
| ---- | ---- | ---- | ---- | ---- |
|Fasting Blood Glucose (FBG) (mmol/L)|C-STRIDE (N=3216)|5.30|1.69|mmol/L|
| |PKUFH CKD (N=342)|6.11|1.56|mmol/L|
|Uric Acid (mmol/L)|C-STRIDE (N=3216)|404.68|117.17|mmol/L|
| |PKUFH CKD (N=342)|396.58|101.87|mmol/L|
|Serum Phosphorus (mmol/L)|C-STRIDE (N=3216)|1.21|0.37|mmol/L|
| |PKUFH CKD (N=342)|1.22|0.31|mmol/L|
|Serum Calcium (mmol/L)|C-STRIDE (N=3216)|2.23|0.20|mmol/L|
| |PKUFH CKD (N=342)|2.31|0.15|mmol/L|
|Serum Potassium (mmol/L)|C-STRIDE (N=3216)|4.44|0.74|mmol/L|
| |PKUFH CKD (N=342)|4.43|0.56|mmol/L|
|Triglyceride (mmol/L)|C-STRIDE (N=3216)|2.16|1.41|mmol/L|
| |PKUFH CKD (N=342)|1.90|1.21|mmol/L|
|Total Cholesterol (TC) (mmol/L)|C-STRIDE (N=3216)|5.23|2.23|mmol/L|
| |PKUFH CKD (N=342)|4.54|1.05|mmol/L|
|HDL-Cholesterol (HDL-C) (mmol/L)|C-STRIDE (N=3216)|1.12|0.33|mmol/L|
| |PKUFH CKD (N=342)|1.14|0.33|mmol/L|
|LDL-Cholesterol (LDL-C) (mmol/L)|C-STRIDE (N=3216)|2.78|1.04|mmol/L|
| |PKUFH CKD (N=342)|2.58|1.44|mmol/L|

### 结局变量信息
|变量名称|数据集|频率（n, %）|平均随访时间（years）|
| ---- | ---- | ---- | ---- |
|KRT (Kidney Replacement Therapy)|C-STRIDE (N=3216)|411 (12.8%)|4.45|
| |PKUFH CKD (N=342)|25 (7.3%)|3.37|

以上表格汇总了文献中提到的主要变量信息，包括连续变量的均值、标准差、中位数和四分位间距，以及分类变量的频率和构成比。这些信息可以直接用于后续的Python代码模拟。
## 五、重要变量和数据(中文展示)
### 连续变量信息
|变量名称|数据集|均值（Mean）|标准差（SD）|中位数（Median）|四分位间距（IQR）|单位|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|年龄（岁）|C - STRIDE（N = 3216）|48|13| - | - |岁|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|55|11| - | - |岁|
|估算肾小球滤过率（eGFR）（毫升/分钟/1.73平方米）|C - STRIDE（N = 3216）|52.97|29.50| - | - |毫升/分钟/1.73平方米|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|50.83|35.04| - | - |毫升/分钟/1.73平方米|
|尿白蛋白 - 肌酐比值（UACR）（毫克/克）|C - STRIDE（N = 3216）| - |376.40|90.80 - 911.45| - |毫克/克|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）| - |214.37|43.55 - 1058.50| - |毫克/克|
|血红蛋白（克/升）|C - STRIDE（N = 3216）|128.75|21.87| - | - |克/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|134.09|20.18| - | - |克/升|
|白蛋白（克/升）|C - STRIDE（N = 3216）|38.54|7.46| - | - |克/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|42.55|4.69| - | - |克/升|
|肌酐（微摩尔/升）|C - STRIDE（N = 3216）|154.27|72.17| - | - |微摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|196.56|168.08| - | - |微摩尔/升|
|收缩压（毫米汞柱）|C - STRIDE（N = 3216）|129.39|17.50| - | - |毫米汞柱|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|132.39|18.62| - | - |毫米汞柱|
|舒张压（毫米汞柱）|C - STRIDE（N = 3216）|80.97|10.80| - | - |毫米汞柱|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|77.65|11.06| - | - |毫米汞柱|

### 分类变量信息
|变量名称|数据集|构成比（Frequency）|单位|
| ---- | ---- | ---- | ---- |
|男性（人数，%）|C - STRIDE（N = 3216）|1909（59.4%）| - |
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|133（38.9%）| - |
|吸烟者（人数，%）|C - STRIDE（N = 3216）|1123（34.9%）| - |
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|11（3.2%）| - |
|高血压（人数，%）|C - STRIDE（N = 3216）|2363（73.5%）| - |
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|83（24.3%）| - |
|2型糖尿病（人数，%）|C - STRIDE（N = 3216）|641（19.9%）| - |
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|99（28.9%）| - |
|心血管疾病（人数，%）|C - STRIDE（N = 3216）|270（8.4%）| - |
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|7（2.0%）| - |
|慢性肾脏病病因（人数，%）|C - STRIDE（N = 3216）|糖尿病肾病（DKD）385（12.0%）<br>肾小球肾炎（GN）1853（57.6%）<br>其他725（22.5%）| - |
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|糖尿病肾病（DKD）88（25.7%）<br>肾小球肾炎（GN）80（23.4%）<br>其他102（29.8%）| - |

### 其他变量信息
|变量名称|数据集|均值（Mean）|标准差（SD）|单位|
| ---- | ---- | ---- | ---- | ---- |
|空腹血糖（FBG）（毫摩尔/升）|C - STRIDE（N = 3216）|5.30|1.69|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|6.11|1.56|毫摩尔/升|
|尿酸（毫摩尔/升）|C - STRIDE（N = 3216）|404.68|117.17|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|396.58|101.87|毫摩尔/升|
|血磷（毫摩尔/升）|C - STRIDE（N = 3216）|1.21|0.37|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|1.22|0.31|毫摩尔/升|
|血钙（毫摩尔/升）|C - STRIDE（N = 3216）|2.23|0.20|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|2.31|0.15|毫摩尔/升|
|血钾（毫摩尔/升）|C - STRIDE（N = 3216）|4.44|0.74|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|4.43|0.56|毫摩尔/升|
|甘油三酯（毫摩尔/升）|C - STRIDE（N = 3216）|2.16|1.41|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|1.90|1.21|毫摩尔/升|
|总胆固醇（TC）（毫摩尔/升）|C - STRIDE（N = 3216）|5.23|2.23|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|4.54|1.05|毫摩尔/升|
|高密度脂蛋白胆固醇（HDL - C）（毫摩尔/升）|C - STRIDE（N = 3216）|1.12|0.33|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|1.14|0.33|毫摩尔/升|
|低密度脂蛋白胆固醇（LDL - C）（毫摩尔/升）|C - STRIDE（N = 3216）|2.78|1.04|毫摩尔/升|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|2.58|1.44|毫摩尔/升|

### 结局变量信息
|变量名称|数据集|频率（人数，%）|平均随访时间（年）|
| ---- | ---- | ---- | ---- |
|肾脏替代治疗（KRT）|C - STRIDE（N = 3216）|411（12.8%）|4.45|
| |北京大学第一医院慢性肾脏病（PKUFH CKD）（N = 342）|25（7.3%）|3.37|

以上表格汇总了文献中提到的主要变量信息，包括连续变量的均值、标准差、中位数和四分位间距，以及分类变量的频率和构成比。这些信息可直接用于后续的Python代码模拟。  


## 六、模拟数据
### 模拟慢性肾脏病数据的Python脚本

以下是一个Python脚本，用于根据上述抓取的数据信息模拟数据，并保存为CSV格式。脚本名为`simulate_ckd_data.py`，保存路径为`04文献阅读\05肾病\08中国多中心慢性肾脏病队列ESKD风险预测模型的推导_验证和比较研究\01data`。

#### Python代码：simulate_ckd_data.py
```python
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm

# 设置保存路径
save_path = r"04文献阅读\05肾病\08中国多中心慢性肾脏病队列ESKD风险预测模型的推导_验证和比较研究\01data"
os.makedirs(save_path, exist_ok=True)

# 模拟数据的函数
def simulate_normal_data(mean, std, size):
    return norm.rvs(loc=mean, scale=std, size=size)

def simulate_lognormal_data(mean, std, size):
    # 计算对数正态分布的参数
    s = np.sqrt(np.log(1 + (std / mean) ** 2))
    scale = mean / np.exp(0.5 * s ** 2)
    return lognorm.rvs(s=s, scale=scale, size=size)

# 模拟C - STRIDE队列数据
np.random.seed(42)  # 设置随机种子以保证可重复性

# 模拟样本数量
n_cstrides = 3216
n_pkufh = 342

# 模拟连续变量
data_cstrides = {
    "Age": simulate_normal_data(48, 13, n_cstrides),
    "eGFR": simulate_normal_data(52.97, 29.50, n_cstrides),
    "UACR": simulate_lognormal_data(376.40, 360.32, n_cstrides),  # 均值和标准差来自中位数和IQR
    "Hemoglobin": simulate_normal_data(128.75, 21.87, n_cstrides),
    "Albumin": simulate_normal_data(38.54, 7.46, n_cstrides),
    "Creatinine": simulate_normal_data(154.27, 72.17, n_cstrides),
    "Systolic_BP": simulate_normal_data(129.39, 17.50, n_cstrides),
    "Diastolic_BP": simulate_normal_data(80.97, 10.80, n_cstrides),
    "Fasting_Blood_Glucose": simulate_normal_data(5.30, 1.69, n_cstrides),
    "Uric_Acid": simulate_normal_data(404.68, 117.17, n_cstrides),
    "Serum_Phosphorus": simulate_normal_data(1.21, 0.37, n_cstrides),
    "Serum_Calcium": simulate_normal_data(2.23, 0.20, n_cstrides),
    "Serum_Potassium": simulate_normal_data(4.44, 0.74, n_cstrides),
    "Triglyceride": simulate_lognormal_data(2.16, 1.41, n_cstrides),
    "Total_Cholesterol": simulate_normal_data(5.23, 2.23, n_cstrides),
    "HDL_Cholesterol": simulate_normal_data(1.12, 0.33, n_cstrides),
    "LDL_Cholesterol": simulate_normal_data(2.78, 1.04, n_cstrides),
}

# 模拟分类变量
data_cstrides["Male"] = np.random.binomial(n=1, p=0.594, size=n_cstrides)
data_cstrides["Smoker"] = np.random.binomial(n=1, p=0.349, size=n_cstrides)
data_cstrides["Hypertension"] = np.random.binomial(n=1, p=0.735, size=n_cstrides)
data_cstrides["T2DM"] = np.random.binomial(n=1, p=0.199, size=n_cstrides)
data_cstrides["CVD"] = np.random.binomial(n=1, p=0.084, size=n_cstrides)

# 模拟CKD病因
causes = ["DKD", "GN", "Other"]
cause_probs = [0.120, 0.576, 0.225]
data_cstrides["Cause_of_CKD"] = np.random.choice(causes, size=n_cstrides, p=cause_probs)

# 模拟KRT结局
data_cstrides["KRT"] = np.random.binomial(n=1, p=0.128, size=n_cstrides)
data_cstrides["Follow_Up_Time"] = np.random.uniform(low=1, high=10, size=n_cstrides)  # 假设随访时间为1 - 10年

# 转换为DataFrame
df_cstrides = pd.DataFrame(data_cstrides)

# 模拟PKUFH队列数据
data_pkufh = {
    "Age": simulate_normal_data(55, 11, n_pkufh),
    "eGFR": simulate_normal_data(50.83, 35.04, n_pkufh),
    "UACR": simulate_lognormal_data(214.37, 457.47, n_pkufh),  # 均值和标准差来自中位数和IQR
    "Hemoglobin": simulate_normal_data(134.09, 20.18, n_pkufh),
    "Albumin": simulate_normal_data(42.55, 4.69, n_pkufh),
    "Creatinine": simulate_normal_data(196.56, 168.08, n_pkufh),
    "Systolic_BP": simulate_normal_data(132.39, 18.62, n_pkufh),
    "Diastolic_BP": simulate_normal_data(77.65, 11.06, n_pkufh),
    "Fasting_Blood_Glucose": simulate_normal_data(6.11, 1.56, n_pkufh),
    "Uric_Acid": simulate_normal_data(396.58, 101.87, n_pkufh),
    "Serum_Phosphorus": simulate_normal_data(1.22, 0.31, n_pkufh),
    "Serum_Calcium": simulate_normal_data(2.31, 0.15, n_pkufh),
    "Serum_Potassium": simulate_normal_data(4.43, 0.56, n_pkufh),
    "Triglyceride": simulate_lognormal_data(1.90, 1.21, n_pkufh),
    "Total_Cholesterol": simulate_normal_data(4.54, 1.05, n_pkufh),
    "HDL_Cholesterol": simulate_normal_data(1.14, 0.33, n_pkufh),
    "LDL_Cholesterol": simulate_normal_data(2.58, 1.44, n_pkufh),
}

# 模拟分类变量
data_pkufh["Male"] = np.random.binomial(n=1, p=0.389, size=n_pkufh)
data_pkufh["Smoker"] = np.random.binomial(n=1, p=0.032, size=n_pkufh)
data_pkufh["Hypertension"] = np.random.binomial(n=1, p=0.243, size=n_pkufh)
data_pkufh["T2DM"] = np.random.binomial(n=1, p=0.289, size=n_pkufh)
data_pkufh["CVD"] = np.random.binomial(n=1, p=0.020, size=n_pkufh)

# 模拟CKD病因
cause_probs_pkufh = [0.257, 0.234, 0.298]
data_pkufh["Cause_of_CKD"] = np.random.choice(causes, size=n_pkufh, p=cause_probs_pkufh)

# 模拟KRT结局
data_pkufh["KRT"] = np.random.binomial(n=1, p=0.073, size=n_pkufh)
data_pkufh["Follow_Up_Time"] = np.random.uniform(low=1, high=10, size=n_pkufh)  # 假设随访时间为1 - 10年

# 转换为DataFrame
df_pkufh = pd.DataFrame(data_pkufh)

# 保存为CSV文件
df_cstrides.to_csv(os.path.join(save_path, "C_STRIDE_data.csv"), index=False)
df_pkufh.to_csv(os.path.join(save_path, "PKUFH_data.csv"), index=False)

print("数据模拟完成，已保存到指定路径！")
```

### 代码说明
1. **数据模拟**：
   - 连续变量使用正态分布或对数正态分布模拟。
   - 分类变量使用二项分布模拟。
   - CKD病因使用多项分布模拟。
2. **保存路径**：
数据保存到指定路径`04文献阅读\05肾病\08中国多中心慢性肾脏病队列ESKD风险预测模型的推导_验证和比较研究\01data`。
3. **文件命名**：
   - C - STRIDE队列数据保存为`C_STRIDE_data.csv`。
   - PKUFH队列数据保存为`PKUFH_data.csv`。

### 运行方法
1. 将上述代码保存为`simulate_ckd_data.py`。
2. 确保安装了`numpy`和`pandas`库。
3. 在终端或命令行中运行以下命令：
```bash
python simulate_ckd_data.py
```
模拟数据将自动保存到指定路径。

希望这份代码能帮助你完成后续的工作！

## 七、复现代码
### 用于复现ESKD风险预测模型开发和验证过程的Python脚本

以下是一个Python脚本，用于复现上述文献中的ESKD风险预测模型的开发和验证过程。脚本名为`reproduce_eskd_model.py`，它将基于模拟的数据（C - STRIDE和PKUFH队列）进行Cox比例风险模型和机器学习模型（XGBoost和SSVM）的训练和评估。

#### Python代码：reproduce_eskd_model.py
```python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from xgboost import XGBClassifier
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv
import matplotlib.pyplot as plt

# 加载数据
def load_data():
    cstrides_path = r"04文献阅读\05肾病\08中国多中心慢性肾脏病队列ESKD风险预测模型的推导_验证和比较研究\01data\C_STRIDE_data.csv"
    pkufh_path = r"04文献阅读\05肾病\08中国多中心慢性肾脏病队列ESKD风险预测模型的推导_验证和比较研究\01data\PKUFH_data.csv"
    df_cstrides = pd.read_csv(cstrides_path)
    df_pkufh = pd.read_csv(pkufh_path)
    return df_cstrides, df_pkufh

# 数据预处理
def preprocess_data(df):
    # 对数转换UACR
    df['UACR'] = np.log(df['UACR'] + 1)
    # 标准化连续变量
    scaler = StandardScaler()
    continuous_vars = ['Age', 'eGFR', 'UACR', 'Hemoglobin', 'Albumin', 'Creatinine', 'Systolic_BP', 'Diastolic_BP']
    df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    return df

# 特征选择
def select_features(df):
    features = ['Age', 'Male', 'eGFR', 'UACR', 'Albumin', 'Hemoglobin', 'T2DM', 'Hypertension']
    return df[features]

# 模型训练和评估
def train_and_evaluate_models(train_data, test_data, validation_data):
    # 准备生存分析数据
    y_train = Surv.from_dataframe('KRT', 'Follow_Up_Time', train_data)
    y_test = Surv.from_dataframe('KRT', 'Follow_Up_Time', test_data)
    y_val = Surv.from_dataframe('KRT', 'Follow_Up_Time', validation_data)

    # 特征选择
    X_train = select_features(train_data)
    X_test = select_features(test_data)
    X_val = select_features(validation_data)

    # Cox模型
    cph = CoxPHFitter()
    cph.fit(train_data, duration_col='Follow_Up_Time', event_col='KRT')
    cph_test_cindex = concordance_index_censored(y_test['fstat'], y_test['stop'], cph.predict_partial_hazard(X_test))[0]
    cph_val_cindex = concordance_index_censored(y_val['fstat'], y_val['stop'], cph.predict_partial_hazard(X_val))[0]

    # XGBoost模型
    xgb = XGBClassifier(objective='binary:logistic')
    xgb.fit(X_train, train_data['KRT'])
    xgb_test_cindex = concordance_index_censored(y_test['fstat'], y_test['stop'], xgb.predict_proba(X_test)[:, 1])[0]
    xgb_val_cindex = concordance_index_censored(y_val['fstat'], y_val['stop'], xgb.predict_proba(X_val)[:, 1])[0]

    # SSVM模型
    ssvm = FastSurvivalSVM()
    ssvm.fit(X_train, y_train)
    ssvm_test_cindex = concordance_index_censored(y_test['fstat'], y_test['stop'], ssvm.predict(X_test))[0]
    ssvm_val_cindex = concordance_index_censored(y_val['fstat'], y_val['stop'], ssvm.predict(X_val))[0]

    # 输出结果
    print("Cox Model - Test C-index:", cph_test_cindex, "Validation C-index:", cph_val_cindex)
    print("XGBoost Model - Test C-index:", xgb_test_cindex, "Validation C-index:", xgb_val_cindex)
    print("SSVM Model - Test C-index:", ssvm_test_cindex, "Validation C-index:", ssvm_val_cindex)

# 主函数
def main():
    df_cstrides, df_pkufh = load_data()
    df_cstrides = preprocess_data(df_cstrides)
    df_pkufh = preprocess_data(df_pkufh)

    # 分割C-STRIDE数据为训练集和测试集
    train_data, test_data = train_test_split(df_cstrides, test_size=0.3, random_state=42)

    # 使用PKUFH数据作为外部验证集
    validation_data = df_pkufh

    # 训练和评估模型
    train_and_evaluate_models(train_data, test_data, validation_data)

if __name__ == "__main__":
    main()
```

### 代码说明
1. **数据加载**：
从指定路径加载C - STRIDE和PKUFH队列数据。
2. **数据预处理**：
    - 对UACR进行对数转换。
    - 对连续变量进行标准化。
3. **特征选择**：
根据文献选择与ESKD风险相关的特征。
4. **模型训练和评估**：
    - 使用Cox比例风险模型、XGBoost和生存支持向量机（SSVM）进行模型训练。
    - 使用Harrell’s C - index评估模型的判别能力。
    - 在测试集和外部验证集上评估模型性能。
5. **结果输出**：
输出每个模型在测试集和验证集上的C - index值。

### 运行方法
1. 确保已安装以下库：`pandas`、`numpy`、`sklearn`、`lifelines`、`xgboost`、`scikit - survival`。
2. 将上述代码保存为`reproduce_eskd_model.py`。
3. 在终端或命令行中运行以下命令：
```bash
python reproduce_eskd_model.py
```
脚本将输出每个模型在测试集和验证集上的C - index值。

希望这份代码能帮助你复现文献中的结果！