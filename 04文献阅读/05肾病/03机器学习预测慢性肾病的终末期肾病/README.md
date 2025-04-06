# 机器学习在预测慢性肾病发展为终末期肾病中的应用研究
Machine learning to predict end stage kidney disease in chronic kidney disease

## 文献信息


期刊名称：Frontiers in Medicine
影响因子：3.10
JCR分区：Q1
中科院分区(2025)：医学3区
小类：医学：内科3区
中科院分区(2023)：医学3区
小类：医学：内科3区
OPEN ACCESS：99.44%
出版周期：暂无数据
是否综述：否
预警等级：2021中科院国际期刊低度预警，多地黑名单预警（2023湘雅医学院、长江大学低风险预警、江苏省肿瘤医院低危、浙江工商大学、安徽省立医院）
年度|影响因子|发文量|自引率
2023 | 3.10 | 2539 | 3.2%
2022 | 3.90 | 3950 | 5.1%
2021 | 5.06 | 2676 | 4.3%
2020 | 5.09 | 982 | 2.2%
2019 | 3.90 | 285 | 1.3%


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
ML模型能够利用易于获取的临床数据进行ESKD预测，具有较高的敏感性，可能有助于早期识别高风险患者。

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

## 主要数据(英文展示)

# 重要变量信息
以下是以Markdown表格形式呈现的根据文献内容抓取的重要变量信息，包括变量名、均值、方差、中位数、分类变量的频率等信息，以便后续使用Python模拟数据生成。

## 1. Continuous Variables (连续变量)
| Variable Name | Mean ± SD | Median (IQR) |
| ---- | ---- | ---- |
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

## 2. Categorical Variables (分类变量)
### Gender (Male/Female)
| Gender | Frequency (n) | Percentage (%) |
| ---- | ---- | ---- |
| Male | 419 | - |
| Female | 329 | - |

### Primary Disease
| Primary Disease | Frequency (n) | Percentage (%) |
| ---- | ---- | ---- |
| Primary Glomerulonephritis (GN) | 292 | 39.0 |
| Diabetes | 224 | 29.9 |
| Hypertension | 97 | 13.0 |
| Chronic Interstitial Nephritis (CIN) | 64 | 8.6 |
| Others | 18 | 2.4 |
| Unknown | 53 | 7.1 |

### Medical History
| Medical History | Frequency (n) | Percentage (%) |
| ---- | ---- | ---- |
| Hypertension | 558 | 74.6 |
| Diabetes Mellitus | 415 | 55.5 |
| Cardiovascular or Cerebrovascular Disease | 177 | 23.7 |
| Smoking | 91 | 12.6 |

## 3. CKD Stages (肾病分期)
| CKD Stage | Frequency (n) | Percentage (%) |
| ---- | ---- | ---- |
| Stage 1 | 58 | 7.8 |
| Stage 2 | 183 | 24.5 |
| Stage 3 | 352 | 47.1 |
| Stage 4 | 119 | 15.9 |
| Stage 5 | 36 | 4.8 |

## 4. Outcome Variable (目标变量)
| Outcome | Frequency (n) | Percentage (%) |
| ---- | ---- | ---- |
| ESKD+ (Kidney Failure) | 70 | 9.4 |
| ESKD- (No Kidney Failure) | 678 | 90.6 |

## 5. Additional Notes
- The dataset contains 748 subjects with a follow-up duration of 6.3 ± 2.3 years.
- Missing data were handled using multiple imputation.
- The primary endpoint was kidney failure requiring renal replacement therapy (RRT), labeled as ESKD+.

## 主要数据(中文展示)

以下是抓取的重要变量信息，以中文形式展示，以便后续使用Python模拟数据生成：

### 连续变量（Continuous Variables）
|变量名称|均值 ± 标准差|中位数（四分位间距）|
| ---- | ---- | ---- |
|年龄（岁）|57.8 ± 17.6| - |
|收缩压（SBP，mmHg）|129.5 ± 17.8| - |
|舒张压（DBP，mmHg）|77.7 ± 11.1| - |
|体质指数（BMI，kg/m²）|24.8 ± 3.7| - |
|血清肌酐（µmol/L）| - |130.0 (100.0, 163.0)|
|血尿素氮（BUN，mmol/L）| - |7.9 (5.6, 10.4)|
|丙氨酸氨基转移酶（ALT，U/L）| - |17.0 (12.0, 24.0)|
|天门冬氨酸氨基转移酶（AST，U/L）| - |18.0 (15.0, 22.0)|
|碱性磷酸酶（ALP，U/L）| - |60.0 (50.0, 75.0)|
|总蛋白（g/L）|71.6 ± 8.4| - |
|白蛋白（g/L）|42.2 ± 5.6| - |
|尿酸（µmol/L）| - |374.0 (301.0, 459.0)|
|钙（mmol/L）|2.2 ± 0.1| - |
|磷（mmol/L）|1.2 ± 0.2| - |
|钙磷乘积（Ca × P，mg²/dL²）|33.5 ± 5.6| - |
|白细胞计数（10⁹/L）|7.1 ± 2.4| - |
|血红蛋白（g/L）|131.0 ± 20.3| - |
|血小板计数（10⁹/L）|209.8 ± 57.1| - |
|估算肾小球滤过率（eGFR，ml/min/1.73m²）| - |46.1 (32.6, 67.7)|
|总胆固醇（mmol/L）| - |5.1 (4.3, 5.9)|
|甘油三酯（mmol/L）| - |1.8 (1.3, 2.6)|
|高密度脂蛋白胆固醇（HDL-c，mmol/L）| - |1.3 (1.1, 1.6)|
|低密度脂蛋白胆固醇（LDL-c，mmol/L）| - |3.0 (2.4, 3.7)|
|空腹血糖（mmol/L）| - |5.4 (4.9, 6.2)|
|钾（mmol/L）|4.3 ± 0.5| - |
|钠（mmol/L）|140.2 ± 2.8| - |
|氯（mmol/L）|106.9 ± 3.7| - |
|碳酸氢盐（mmol/L）|25.9 ± 3.6| - |

### 分类变量（Categorical Variables）
#### 性别（男/女）
|变量名称|频数（n）|百分比（%）|
| ---- | ---- | ---- |
|男|419| - |
|女|329| - |

#### 原发疾病
|变量名称|频数（n）|百分比（%）|
| ---- | ---- | ---- |
|原发性肾小球肾炎（GN）|292|39.0|
|糖尿病|224|29.9|
|高血压|97|13.0|
|慢性间质性肾炎（CIN）|64|8.6|
|其他|18|2.4|
|未知|53|7.1|

#### 病史
|变量名称|频数（n）|百分比（%）|
| ---- | ---- | ---- |
|高血压|558|74.6|
|糖尿病|415|55.5|
|心血管或脑血管疾病|177|23.7|
|吸烟|91|12.6|

### 肾病分期（CKD Stages）
|肾病分期|频数（n）|百分比（%）|
| ---- | ---- | ---- |
|1期|58|7.8|
|2期|183|24.5|
|3期|352|47.1|
|4期|119|15.9|
|5期|36|4.8|

###  目标变量（Outcome Variable）
|结局|频数（n）|百分比（%）|
| ---- | ---- | ---- |
|ESKD+（肾衰竭）|70|9.4|
|ESKD-（无肾衰竭）|678|90.6|

### 附加说明
- 数据集包含748名受试者，随访时间为6.3 ± 2.3年。
- 缺失数据通过多重插补处理。
- 主要终点是需要肾脏替代治疗（RRT）的肾衰竭，标记为ESKD+。


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

## 上述两端模拟数据代码的区别


### 第一段代码：`generate_simulated_ckd_data.py`
#### 功能
- 生成模拟数据，包括连续变量、分类变量、肾病分期和目标变量。
- 不包含缺失数据的引入，所有变量的值都是完整的。
- 保存生成的数据到指定路径。

#### 适用场景
适用于需要完整数据集的场景，例如初步模型训练或验证，而不涉及缺失数据的处理。

#### 核心逻辑
- 使用正态分布（`norm.rvs`）生成连续变量。
- 使用`np.random.choice`生成分类变量。
- 合并所有数据并保存为CSV文件。

### 第二段代码：`generate_simulated_ckd_data_with_missing_and_imputation.py`
#### 功能
- 生成模拟数据，包括连续变量、分类变量、肾病分期和目标变量。
- 引入缺失数据，根据指定的缺失比例随机设置某些变量的值为`np.nan`。
- 可以使用多重插补（`IterativeImputer`）处理缺失数据，生成完整的数据集。
- 保存处理后的数据到指定路径。

#### 适用场景
- 适用于需要模拟真实世界数据的场景，尤其是数据可能包含缺失值的情况。
- 适合进行缺失数据处理方法的研究或验证。

#### 核心逻辑
- 与第一段代码类似，生成连续变量和分类变量。

#### 额外步骤
- **引入缺失数据**：根据指定的缺失比例随机设置某些变量的值为`np.nan`。
- **使用`IterativeImputer`进行多重插补**，处理缺失值。
- **保存处理后的数据**。

## 具体区别
### 缺失数据的引入
- **第一段代码**：不引入缺失数据。
- **第二段代码**：引入缺失数据，并根据指定比例随机设置某些变量的值为`np.nan`。

### 多重插补
- **第一段代码**：不进行多重插补。
- **第二段代码**：使用`IterativeImputer`进行多重插补，处理缺失数据。

### 保存的数据内容
- **第一段代码**：保存完整的模拟数据。
- **第二段代码**：保存经过多重插补处理后的模拟数据。

### 总结
- 如果你需要一个完整的数据集，没有缺失值，使用第一段代码。
- 如果你需要一个更接近真实世界的数据集，包含缺失值并进行多重插补处理，使用第二段代码。

# 原文

Machine learning to predict end stage kidney disease in chronic kidney disease
机器学习预测慢性肾脏病中的终末期肾脏疾病

## Abstract  摘要
The purpose of this study was to assess the feasibility of machine learning (ML) in predicting the risk of end-stage kidney disease (ESKD) from patients with chronic kidney disease (CKD). Data were obtained from a longitudinal CKD cohort. Predictor variables included patients’ baseline characteristics and routine blood test results. The outcome of interest was the presence or absence of ESKD by the end of 5 years. Missing data were imputed using multiple imputation. Five ML algorithms, including logistic regression, naïve Bayes, random forest, decision tree, and K-nearest neighbors were trained and tested using fivefold cross-validation. The performance of each model was compared to that of the Kidney Failure Risk Equation (KFRE). The dataset contained 748 CKD patients recruited between April 2006 and March 2008, with the follow-up time of 6.3 ± 2.3 years. ESKD was observed in 70 patients (9.4%). Three ML models, including the logistic regression, naïve Bayes and random forest, showed equivalent predictability and greater sensitivity compared to the KFRE. The KFRE had the highest accuracy, specificity, and precision. This study showed the feasibility of ML in evaluating the prognosis of CKD based on easily accessible features. Three ML models with adequate performance and sensitivity scores suggest a potential use for patient screenings. Future studies include external validation and improving the models with additional predictor variables.

本研究旨在评估机器学习（ML）在预测慢性肾脏病（CKD）患者终末期肾脏病（ESKD）风险方面的可行性。数据来自纵向 CKD 队列。预测变量包括患者的基线特征和常规血液检查结果。关注的结果是 5 年结束时是否存在 ESKD。使用多重插补对缺失数据进行插补。五种 ML 算法，包括逻辑回归，朴素贝叶斯，随机森林，决策树和 K-最近邻，使用五重交叉验证进行训练和测试。将每个模型的性能与肾衰竭风险方程（KFRE）的性能进行比较。该数据集包含 2006 年 4 月至 2008 年 3 月期间招募的 748 名 CKD 患者，随访时间为 6.3 ± 2.3 年。 70 例患者（9.4%）观察到 ESKD。三种 ML 模型，包括逻辑回归，朴素贝叶斯和随机森林，与 KFRE 相比，表现出相当的可预测性和更高的灵敏度。KFRE 具有最高的准确度、特异性和精密度。这项研究表明，ML 在评估 CKD 预后的基础上，易于访问的功能的可行性。三个 ML 模型具有足够的性能和灵敏度评分，表明可能用于患者筛查。未来的研究包括外部验证和改进模型与额外的预测变量。

## Subject terms: Nephrology, Kidney diseases, Chronic kidney disease, End-stage renal disease
主题术语： 肾脏病学、肾脏疾病、慢性肾脏疾病、终末期肾脏疾病


## Introduction  介绍
Chronic kidney disease (CKD) is a significant healthcare burden that affects billions of individuals worldwide[1,2] and makes a profound impact on global morbidity and mortality[3–5]. In the United States, approximately 11% of the population or 37 million people suffer from CKD that results in an annual Medicare cost of $84 billion[6]. The prevalence of this disease is estimated at 10.8% in China, affecting about 119.5 million people[7].

慢性肾脏疾病（CKD）是一种严重的医疗保健负担，影响全球数十亿人 [1，2]，并对全球发病率和死亡率产生深远影响 [3-5]。在美国，约有 11%的人口或 3700 万人患有 CKD，导致每年的医疗保险费用为 840 亿美元。 据估计，这种疾病在中国的患病率为 10.8%，影响约 1.195 亿人。

Gradual loss of the kidney function can lead to end stage kidney disease (ESKD) in CKD patients, precipitating the need for kidney replacement therapy (KRT). Timely intervention in those CKD patients who have a high risk of ESKD may not only improve these patients’ quality of life by delaying the disease progression, but also reduce the morbidity, mortality and healthcare costs resulting from KRT[8,9]. Because the disease progression is typically silent[10], a reliable prediction model for risk of ESKD at the early stage of CKD can be clinically essential. Such a model is expected to facilitate physicians in making personalized treatment decisions for high-risk patients, thereby improving the overall prognosis and reducing the economic burden of this disease.

肾功能的逐渐丧失可导致 CKD 患者的终末期肾病（ESKD），从而促使需要肾脏替代治疗（KRT）。对 ESKD 高风险 CKD 患者进行及时干预不仅可以通过延迟疾病进展改善这些患者的生活质量，还可以降低 KRT 导致的发病率、死亡率和医疗费用 [8，9]。由于疾病进展通常是无症状的 [10]，因此在 CKD 早期阶段 ESKD 风险的可靠预测模型在临床上是必不可少的。这种模型有望帮助医生为高危患者做出个性化的治疗决策，从而改善整体预后并减轻这种疾病的经济负担。

A few statistical models were developed to predict the likelihood of ESKD based on certain variables, including age, gender, lab results, and most commonly, the estimated glomerular filtration rate (eGFR) and albuminuria[11,12]. Although some of these models demonstrated adequate predictability in patients of a specific race, typically Caucasians[13–15], literature on their generalizability in other ethnic groups, such as Chinese, remains scarce[13,16]. In addition, models based on non-urine variables, such as patients’ baseline characteristics and routine blood tests, have reportedly yield sufficient performance[17,18]. Therefore, it may be feasible to predict ESKD without urine tests, leading to a simplified model with equivalent reliability.

开发了一些统计模型来预测 ESKD 的可能性，基于某些变量，包括年龄，性别，实验室结果，最常见的是估计的肾小球滤过率（eGFR）和白蛋白尿 [11，12]。尽管这些模型中的一些在特定种族的患者中表现出足够的可预测性，通常是高加索人 [13-15]，但关于其在其他种族群体（如中国人）中的普遍性的文献仍然很少 [13，16]。此外，据报道，基于非尿液变量（如患者的基线特征和常规血液检查）的模型具有足够的性能 [17，18]。因此，它可能是可行的预测 ESKD 没有尿液测试，导致一个简化的模型具有同等的可靠性。

With the advent of the big data era, new methods became available in developing a predictive model that used to rely on traditional statistics. Machine learning (ML) is a subset of artificial intelligence (AI) that allows the computer to perform a specific task without explicit instructions. When used in predictive modeling, ML algorithm can be trained to capture the underlying patterns of the sample data and make predictions about the new data based on the acquired information[19]. Compared to traditional statistics, ML represents more sophisticated math functions and usually results in better performance in predicting an outcome that is determined by a large set of variables with non-linear, complex interactions[20]. ML has recently been applied in numerous studies and demonstrated high level of performance that surpassed traditional statistics and even humans[20–23].

随着大数据时代的到来，开发预测模型的新方法变得可用，而这些预测模型过去依赖于传统的统计数据。机器学习（ML）是人工智能（AI）的一个子集，允许计算机在没有明确指令的情况下执行特定任务。当在预测建模中使用时，ML 算法可以被训练以捕获样本数据的底层模式，并基于所获取的信息 19 对新数据进行预测。与传统统计相比，ML 代表了更复杂的数学函数，并且通常在预测由具有非线性复杂交互的大量变量确定的结果时具有更好的性能。ML 最近被应用于许多研究，并表现出超越传统统计甚至人类的高水平性能。

This article presents a proof-of-concept study with the major goal to establish ML models for predicting the risk of ESKD on a Chinese CKD dataset. The ML models were trained and tested based on easily obtainable variables, including the baseline characteristics and routine blood tests. Results obtained from this study suggest not only the feasibility of ML models in performing this clinically critical task, but also the potential in facilitating personalized medicine.

本文提出了一项概念验证研究，主要目标是建立 ML 模型，用于预测中国 CKD 数据集上 ESKD 的风险。ML 模型基于容易获得的变量进行训练和测试，包括基线特征和常规血液检查。从这项研究中获得的结果不仅表明了 ML 模型在执行这一临床关键任务方面的可行性，而且还表明了促进个性化医疗的潜力。

## Materials and methods  材料和方法
### Study population  研究人群
The data used for this retrospective work were obtained from a longitudinal cohort previously enrolled in an observational study[24,25]. The major inclusion criteria for the cohort were adult CKD patients (≥ 18 years old) with stable kidney functions for at least three months prior to recruitment. Patients were excluded if they had one or more of the following situations: (1) history of KRT in any form, including hemodialysis, peritoneal dialysis or kidney transplantation; (2) any other existing condition deemed physically unstable, including life expectancy < 6 months, acute heart failure, and advanced liver disease; (3) any pre-existing malignancy. All patients were recruited from the CKD management clinic of Peking University Third Hospital between April 2006 and March 2008. Written informed consent was obtained from all patients. They were treated according to routine clinical practice determined by the experienced nephrologists and observed until December 31st, 2015. Detailed information regarding patient recruitment and management protocol has been described in a previous publication[24].

本回顾性研究使用的数据来自于先前在一项观察性研究中招募的纵向队列 [24，2]5。该队列的主要入选标准是招募前至少 3 个月肾功能稳定的成人 CKD 患者（≥ 18 岁）。如果患者有以下一种或多种情况，则将其排除：（1）任何形式的 KRT 病史，包括血液透析、腹膜透析或肾移植;（2）任何其他被认为身体不稳定的现有状况，包括预期寿命< 6 个月、急性心力衰竭和晚期肝病;（3）任何预先存在的恶性肿瘤。所有患者均于 2006 年 4 月至 2008 年 3 月期间从北京大学第三医院 CKD 管理门诊招募。获得所有患者的书面知情同意书。 根据经验丰富的肾脏科医生确定的常规临床实践进行治疗，并观察至 2015 年 12 月 31 日。 关于患者招募和管理方案的详细信息已在先前的出版物 [24] 中描述。

### Data acquisition  数据采集
Patient characteristics included age, gender, education level, marriage status, and insurance status. Medical history comprised history of smoking, history of alcohol consumption, presence of each comorbid condition—diabetes, cardiovascular disease and hypertension. Clinical parameters contained body mass index (BMI), systolic pressure and diastolic pressure. Blood tests consisted of serum creatinine, uric acid, blood urea nitrogen, white blood cell count, hemoglobin, platelets count, alanine aminotransferase (ALT), aspartate aminotransferase (AST), total protein, albumin, alkaline phosphatase (ALP), high-density lipoprotein, low-density lipoprotein, triglycerides, total cholesterol, calcium, phosphorus, potassium, sodium, chloride, and bicarbonate. The estimated glomerular filtration rate and type of primary kidney disease were also used as predictors.

患者特征包括年龄、性别、教育水平、婚姻状况和保险状况。病史包括吸烟史、饮酒史、存在每种共病-糖尿病、心血管疾病和高血压。临床参数包括体重指数（BMI）、收缩压和舒张压。血液检查包括血清肌酐、尿酸、血尿素氮、白色血细胞计数、血红蛋白、血小板计数、丙氨酸氨基转移酶（ALT）、天冬氨酸氨基转移酶（AST）、总蛋白、白蛋白、碱性磷酸酶（ALP）、高密度脂蛋白、低密度脂蛋白、甘油三酯、总胆固醇、钙、磷、钾、钠、氯和碳酸氢盐。估计肾小球滤过率和原发性肾脏疾病的类型也被用作预测因子。

All baseline variables were obtained at the time of subject enrollment. The primary study end point was kidney failure which necessitated the use of any KRT. Subjects with the outcome of kidney failure were labeled as ESKD+, and the rest ESKD−. Patients who died before reaching the study end point or lost to follow up were discarded. Patients who developed ESKD after five years were labeled as ESKD−.

在受试者入组时获得所有基线变量。主要研究终点是肾衰竭，需要使用任何 KRT。结局为肾衰竭的受试者标记为 ESKD+，其余为 ESKD−。在达到研究终点前死亡或失访的患者被丢弃。5 年后发生 ESKD 的患者被标记为 ESKD−。

### Data preprocessing  数据预处理
All categorical variables, such as insurance status, education, and primary disease, were encoded using the one-hot approach. Any variable was removed from model development if the missing values were greater than 50%. Missing data were handled using multiple imputation with five times of repetition, leading to five slightly different imputed datasets where each of the missing values was randomly sampled from their predictive distribution based on the observed data. On each imputed set, all models were trained and tested using a fivefold cross validation method. To minimize selection bias, subject assignment to train/test folds was kept consistent across all imputed sets. Data were split in a stratified fashion to ensure the same distribution of the outcome classes (ESKD+ vs. ESKD−) in each subset as the entire set.

所有的分类变量，如保险状况，教育和原发疾病，都是使用独热方法编码的。如果缺失值大于 50%，则从模型开发中删除任何变量。使用重复 5 次的多重插补处理缺失数据，得到 5 个略有不同的插补数据集，其中每个缺失值均基于观察数据从其预测分布中随机采样。在每个插补集上，使用五重交叉验证方法训练和测试所有模型。为了最大限度地减少选择偏差，受试者对训练/测试倍数的分配在所有插补集中保持一致。以分层方式分割数据，以确保每个子集中结局类别（ESKD+ vs. ESKD-）的分布与整个数据集相同。

### Model development  模型开发
The model was trained to perform a binary classification task with the goal of generating the probability of ESKD+ based on the given features. Five ML algorithms were employed in this study, including logistic regression, naïve Bayes, random forest, decision tree, and K-nearest neighbors. Grid search was performed to obtain the best hyperparameter combination for each algorithm.

该模型经过训练以执行二进制分类任务，目标是根据给定的特征生成 ESKD+的概率。在这项研究中使用了五种 ML 算法，包括逻辑回归，朴素贝叶斯，随机森林，决策树和 K-最近邻。进行网格搜索，以获得每个算法的最佳超参数组合。

### Assessment of model performance    模型性能评估
The performance of a classifiers was measured using accuracy, precision, recall, specificity, F1 score and area under the curve (AUC), as recommended by guidelines for results reporting of clinical prediction models[26]. All classifiers developed in this study were further compared with the Kidney Failure Risk Equation (KFRE), which estimates the 5-year risk of ESKD based on patient’s age, gender, and eGFR[12]. The KFRE is currently the most widely used model in predicting CKD progression to ESKD. The reported outcome of a model represented the average performance of 5 test folds over all imputed sets.

使用准确度、精确度、召回率、特异性、F1 评分和曲线下面积（AUC）测量分类器的性能，如临床预测模型结果报告指南 [26] 所推荐的。将本研究中开发的所有分类器与肾衰竭风险方程（KFRE）进行进一步比较，KFRE 根据患者的年龄、性别和 eGFR12 估计 ESKD 的 5 年风险。KFRE 是目前预测 CKD 进展为 ESKD 的最广泛使用的模型。报告的模型结果代表了所有插补集上 5 个检验倍数的平均性能。

### Statistical analysis  统计分析
Basic descriptive statistics were applied as deemed appropriate. Results are expressed as frequencies and percentages for categorical variables; the mean ± standard deviation for continuous, normally distributed variables; and the median (interquartile range) for continuous variables that were not normally distributed. Patient characteristics were compared between the original dataset and the imputed sets using one-way analysis of variance (ANOVA). The AUC of each model was measured using the predicted probability. The optimal threshold of a classifier was determined based on the receiver operating characteristic (ROC) curve at the point with minimal distance to the upper left corner. For each ML model, this threshold was obtained during the training process and applied unchangeably to the test set. For the KFRE, the threshold was set at a default value of 0.5. Model development, performance evaluation and data analyses were all performed using Python[27]. The alpha level was set at 0.05.

适当时采用基本描述性统计量。结果表示为分类变量的频率和百分比;连续、正态分布变量的平均值±标准差;非正态分布的连续变量的中位数（四分位距）。使用单因素方差分析（ANOVA）比较原始数据集和插补集之间的患者特征。使用预测概率测量每个模型的 AUC。根据受试者工作特征（ROC）曲线，在距离左上角最小的点确定分类器的最佳阈值。对于每个 ML 模型，这个阈值是在训练过程中获得的，并不可更改地应用于测试集。对于 KFRE，阈值设置为默认值 0.5。模型开发、性能评估和数据分析都使用 Python[27] 进行。 α水平设定为0.05。

### Ethical approval  伦理批准
This research was conducted ethically in accordance with the World Medical Association Declaration of Helsinki. The study protocol has been approved by the Peking University Third Hospital Medical Science Research Ethics Committee on human research (No. M2020132).

本研究按照世界医学协会赫尔辛基宣言进行。本研究方案已获得北京大学第三医院医学科学研究伦理委员会关于人体研究的批准（编号：M2020132）。

## Results  结果
### Cohort characteristics  队列特征
The dataset contained a total of 748 subjects with the follow-up duration of 6.3 ± 2.3 years. The baseline characteristics are summarized in Table 1. Most patients were in stage 2 (24.5%) or 3 (47.1%) CKD at baseline. ESKD was observed in 70 patients (9.4%), all of whom subsequently received KRT, including hemodialysis in 49 patients, peritoneal dialysis in 17 and kidney transplantation in 4.
数据集共包含 748 例受试者，随访时间为 6.3 ± 2.3 年。基线特征总结见表 1 。大多数患者在基线时处于 CKD 2 期（24.5%）或 3 期（47.1%）。70 例患者（9.4%）观察到 ESKD，所有患者随后接受 KRT，包括 49 例血液透析，17 例腹膜透析和 4 例肾移植。










### Model performance  模型性能
Details of the five imputed sets are provided in the supplemental materials. There was no significant difference between the imputed sets and the original dataset in each variable where missing data were replaced by imputed values. The hyperparameter settings for each classifier are displayed in Table 2. The best overall performance, as measured by the AUC score, was achieved by the random forest algorithm (0.81, see Table 3). Nonetheless, this score and its 95% confidence interval had overlap with those of the other three models, including the logistic regression, naïve Bayes, and the KFRE (Fig. 1). Interestingly, the KFRE model that was based on 3 simple variables, demonstrated not only a comparable AUC score but also the highest accuracy, specificity, and precision. At the default threshold, however, the KFRE was one of the least sensitive models (47%).
5 个插补集的详细信息见 supplemental materials 。插补数据集和原始数据集之间的每个变量均无显著差异，其中缺失数据被插补值替代。每个分类器的超参数设置如表 2 所示。通过随机森林算法（0.81，见表 2#）获得了最佳总体性能（通过 AUC 评分测量）。尽管如此，该评分及其 95%置信区间与其他三种模型（包括逻辑回归、朴素贝叶斯和 KFRE）的评分及其 95%置信区间重叠（图 3#）。有趣的是，基于 3 个简单变量的 KFRE 模型不仅表现出相当的 AUC 评分，而且表现出最高的准确度、特异性和精密度。然而，在默认阈值下，KFRE 是最不敏感的模型之一（47%）。
















## Discussion  讨论
With extensive utilization of electronic health record and recent progress in ML research, AI is expanding its impact on healthcare and has gradually changed the way clinicians pursue for problem-solving[28]. Instead of adopting a theory-driven strategy that requires a preformed hypothesis from prior knowledge, training an ML model typically follows a data-driven approach that allows the model to learn from experience alone. Specifically, the model improves its performance iteratively on a training set by comparing the predictions to the ground truths and adjusting model parameters so as to minimize the distance between the predictions and the truths. In nephrology, ML has demonstrated promising performances in predicting acute kidney injury or time to allograft loss from clinical features[29,30], recognizing specific patterns in pathology slides[31,32], choosing an optimal dialysis prescription[33], or mining text in the electronic health record to find specific cases[34,35]. Additionally, a few recent studies were performed to predict the progression of CKD using ML methods. These models were developed to estimate the risk of short-term mortality following dialysis[36], calculate the future eGFR values[37], or assess the 24-h urinary protein levels[18]. To our best knowledge, there hasn’t been any attempt to apply ML methods to predict the occurrence of ESKD in CKD patients.

随着电子健康记录的广泛使用和机器学习研究的最新进展，人工智能正在扩大其对医疗保健的影响，并逐渐改变了临床医生寻求解决问题的方式。 训练 ML 模型通常遵循数据驱动的方法，允许模型仅从经验中学习，而不是采用理论驱动的策略，需要从先验知识中预先形成假设。具体而言，该模型通过将预测与地面事实进行比较并调整模型参数以最小化预测与事实之间的距离来迭代地提高其在训练集上的性能。 在肾脏病学中，ML 在从临床特征预测急性肾损伤或同种异体移植物丢失时间 29、30、识别病理切片中的特定模式 31、32、选择最佳透析处方 33 或挖掘电子健康记录中的文本以找到特定病例 34、35 方面表现出有前途的性能。此外，最近进行了一些研究，使用 ML 方法预测 CKD 的进展。开发这些模型是为了估计透析后短期死亡率的风险 36，计算未来的 eGFR 值 37，或评估 24 小时尿蛋白水平 18。据我们所知，还没有任何尝试应用 ML 方法来预测 CKD 患者 ESKD 的发生。

In the present study, a prediction model for ESKD in CKD patients was explored using ML techniques. Most classifiers demonstrated adequate performance based on easily accessible patient information that is convenient for clinical translation. In general, three ML models, including the logistic regression, naïve Bayes and random forest, showed non-inferior performance to the KFRE in this study. These findings imply ML as a feasible approach for predicting disease progression in CKD, which could potentially guide physicians in establishing personalized treatment plans for this condition at an early stage. These ML models with higher sensitivity scores may also be practically favored in patient screening over the KFRE.

在本研究中，使用 ML 技术探索了 CKD 患者 ESKD 的预测模型。大多数分类器基于便于临床翻译的易于访问的患者信息表现出足够的性能。总体而言，三种 ML 模型，包括逻辑回归，朴素贝叶斯和随机森林，在本研究中表现出不劣于 KFRE 的性能。这些发现意味着 ML 是预测 CKD 疾病进展的可行方法，这可能会指导医生在早期阶段为这种疾病制定个性化治疗计划。这些具有更高灵敏度评分的 ML 模型在患者筛选中也可能实际上优于 KFRE。

To our best understanding, this study was also the first to validate the KFRE in CKD patients of Mainland China. The KFRE was initially developed and validated using North American patients with CKD stage 3–5[12]. There were seven KFRE models that consisted of different combinations of predictor variables. The most commonly used KFRE included a 4-variable model (age, gender, eGFR and urine ACR) or an 8-variable model (age, gender, eGFR, urine ACR, serum calcium, phosphorous, bicarbonate, and albumin). Besides, there was a 3-variable model (age, gender, and eGFR) that required no urine ACR and still showed comparable performance to the other models in the original article. Despite its favorable performance in prediction for ESKD in patients of Western countries[14,15,38,39], the generalizability of KFRE in Asian population remained arguable following the suboptimal results revealed by some recent papers[13,40,41]. In the current study, the KFRE was validated in a Chinese cohort with CKD stage 1–5 and showed an AUC of 0.80. This result indicated the KFRE was adequately applicable to the Chinese CKD patients and even earlier disease stages. In particular, the high specificity score (0.95) may favor the use of this equation in ruling in patients who require close monitoring of disease progression. On the other hand, a low sensitivity (0.47) at the default threshold may suggest it may be less desirable than the other models for ruling out patients.

据我们所知，这项研究也是首次在中国大陆 CKD 患者中验证 KFRE。KFRE 最初是使用患有 CKD 3-5 期的北美患者开发和验证的。 有七个 KFRE 模型，包括不同的预测变量的组合。最常用的 KFRE 包括 4 变量模型（年龄、性别、eGFR 和尿 ACR）或 8 变量模型（年龄、性别、eGFR、尿 ACR、血清钙、磷、碳酸氢盐和白蛋白）。此外，有一个 3 变量模型（年龄、性别和 eGFR），不需要尿 ACR，仍然显示出与原始文章中其他模型相当的性能。 尽管 KFRE 在预测西方国家患者 ESKD 方面表现良好 [14，15，38，39]，但在亚洲人群中的普适性仍然是不可接受的，因为最近的一些论文揭示了次优结果 13，40，41。在当前研究中，KFRE 在 1-5 期 CKD 中国队列中得到验证，AUC 为 0.80。这一结果表明，KFRE 适用于中国 CKD 患者，甚至更早的疾病阶段。特别是，高特异性评分（0.95）可能有利于在需要密切监测疾病进展的患者中使用该方程。另一方面，在默认阈值处的低灵敏度（0.47）可能表明其可能不如用于排除患者的其他模型理想。

Urine test is a critical diagnostic approach for CKD. The level of albuminuria (i.e. ACR) has also been regarded as a major predictor for disease progression and therefore used by most prognostic models. However, quantitative testing for albuminuria is not always available in China especially in rural areas, which precludes clinicians from using most urine-based models for screening patients. In this regard, several simplified models were developed to predict CKD progression without the need of albuminuria. These models were based on patient characteristics (e.g. age, gender, BMI, comorbidity) and/or blood work (e.g. creatinine/eGFR, BUN), and still able to achieve an AUC of 0.87–0.8912,[18] or a sensitivity of 0.8837. Such performance was largely consistent with the findings of this study and comparable or even superior to some models incorporating urine tests[16,42]. Altogether, it suggested a reliable prediction for CKD progression may be obtained from routine clinical variables without urine measures. These models are expected to provide a more convenient screening tool for CKD patients in developing regions.

尿液检查是诊断 CKD 的重要方法。白蛋白尿水平（即 ACR）也被认为是疾病进展的主要预测因子，因此被大多数预后模型使用。然而，在中国，尤其是在农村地区，蛋白尿的定量检测并不总是可用的，这使得临床医生无法使用大多数基于尿液的模型来筛查患者。在这方面，开发了几种简化模型来预测 CKD 进展，而不需要蛋白尿。这些模型基于患者特征（例如，年龄、性别、BMI、合并症）和/或血液检查（例如，肌酐/eGFR、BUN），并且仍然能够实现 0.87-0.8912，18 的 AUC 或 0.8837 的灵敏度。这种性能与本研究的结果基本一致，并且与结合尿液测试的一些模型相当或甚至优于上级 16，42。 总而言之，这表明可以从常规临床变量中获得 CKD 进展的可靠预测，而无需尿液测量。这些模型有望为发展中地区的 CKD 患者提供更方便的筛查工具。

Missing data are such a common problem in ML research that they can potentially lead to a biased model and undermine the validity of study outcomes. Traditional methods to handle missing data include complete case analysis, missing indicator, single value imputation, sensitivity analyses, and model-based methods (e.g. mixed models or generalized estimating equations)[43–45]. In most scenarios, complete case analysis and single value imputation are favored by researchers primarily due to the ease of implementation[45–47]. However, these methods may be associated with significant drawbacks. For example, by excluding samples with missing data from analyses, complete case analysis can result in reduction of model power, overestimation of benefit and underestimation of harm[43,46]; Single value imputation replaces the missing data by a single value—typically the mean or mode of the complete cases, thereby increasing the homogeneity of data and overestimating the precision[43,48]. In this regard, multiple imputation solves these problems by generating several different plausible imputed datasets, which account for the uncertainty about the missing data and provide unbiased estimates of the true effect[49,50]. It is deemed effective regardless of the pattern of missingness[43,51]. Multiple imputation is now widely recognized as the standard method to deal with missing data in many areas of research[43,45]. In the current study, a 5-set multiple imputation method was employed to obtain reasonable variability of the imputed data. The performance of each model was analyzed on each imputed set and pooled for the final result. These procedures ensured that the model bias resulting from missing data was minimized. In the future, multiple imputation is expected to become a routine method for missing data handling in ML research, as the extra amount of computation associated with multiple imputation over those traditional methods can simply be fulfilled by the high level of computational power required by ML.

缺失数据是 ML 研究中的一个常见问题，它们可能会导致有偏见的模型并破坏研究结果的有效性。处理缺失数据的传统方法包括完整病例分析、缺失指标、单值插补、敏感性分析和基于模型的方法（例如混合模型或广义估计方程）43-45。在大多数情况下，完整的案例分析和单值插补受到研究人员的青睐，主要是因为易于实现 45-47。然而，这些方法可能与显著的缺点相关联。 例如，通过从分析中排除缺失数据的样本，完整的病例分析可能导致模型功效降低，高估获益和低估危害 43，46;单值插补用单个值（通常为完整病例的平均值或众数）替换缺失数据，从而增加数据的同质性并高估精度 43，48。在这方面，多重插补通过生成几个不同的合理插补数据集来解决这些问题，这些数据集解释了缺失数据的不确定性，并提供了真实效应的无偏估计 49，50。无论缺失的模式如何，它都被认为是有效的 43，51。 多重插补现在被广泛认为是标准的方法来处理缺失的数据在许多领域的研究 43，45。在本研究中，采用 5 集多重插补方法获得插补数据的合理变异性。在每个插补集上分析每个模型的性能，并合并最终结果。这些程序确保了缺失数据导致的模型偏倚最小化。在未来，多重插补有望成为 ML 研究中缺失数据处理的常规方法，因为与传统方法相比，与多重插补相关的额外计算量可以简单地通过 ML 所需的高水平计算能力来实现。

Although ML has been shown to outperform traditional statistics in a variety of tasks by virtue of the model complexity, some studies demonstrated no gain or even declination of performance compared to traditional regression methods[52,53]. In this study, the simple logistic regression model also yielded a comparable or even superior predictability for ESKD to other ML algorithms. The most likely explanation is that the current dataset only had a small sample size and limited numbers of predictor variables, and the ESKD+ cases were relatively rare. The lack of big data and imbalanced class distribution may have negative impact on the performance of complex ML algorithms, as they are typically data hungry[54]. On the other hand, this finding could imply simple interactions among the predictor variables. In other words, the risk of ESKD may be largely influenced by only a limited number of factors in an uncomplicated fashion, which is consistent with some previous findings[12,18,55]. The fact that the 3-variable KFRE, which is also a regression model, yielded equivalent outcomes to the best ML models in this study may further support this implication. It is therefore indicated that traditional regression models may continue to play a key role in disease risk prediction, especially when a small sample size, limited predictor variables, or an imbalanced dataset is encountered. The fact that some of the complex ML models are subject to the risk of overfitting and the lack of interpretability further favors the use of simple regression models, which can be translated to explainable equations.

尽管 ML 已经被证明在各种任务中由于模型复杂性而优于传统统计，但一些研究表明，与传统回归方法相比，ML 的性能没有提高甚至下降。 在这项研究中，简单逻辑回归模型也产生了与其他 ML 算法相当甚至上级的 ESKD 可预测性。最可能的解释是，当前数据集的样本量较小，预测变量数量有限，ESKD+病例相对较少。缺乏大数据和不平衡的类分布可能会对复杂 ML 算法的性能产生负面影响，因为它们通常是数据饥饿的。 另一方面，这一发现可能意味着预测变量之间存在简单的相互作用。 换句话说，ESKD 的风险可能在很大程度上仅受有限数量的因素以简单的方式影响，这与之前的一些发现一致 12，18，55。3 变量 KFRE 也是一种回归模型，在本研究中产生了与最佳 ML 模型相同的结果，这一事实可能进一步支持这一含义。因此，传统的回归模型可能继续在疾病风险预测中发挥关键作用，特别是当遇到小样本量，有限的预测变量或不平衡的数据集时。一些复杂的 ML 模型存在过度拟合的风险和缺乏可解释性，这一事实进一步有利于使用简单的回归模型，这些模型可以转换为可解释的方程。

Several limitations should be noted. First, this cohort consisted of less than 1000 subjects and ESKD only occurred in a small portion of them, both of which might have affected model performance as discussed earlier. Second, although this study aimed to assess the feasibility of a prediction model for ESKD without any urine variables, this was partially due to the lack of quantitative urine tests at our institute when this cohort was established. As spot urine tests become increasingly popular, urine features such as ACR will be as accessible and convenient as other lab tests. They are expected to play a critical role in more predictive models. Third, the KFRE was previously established on stages 3–5 CKD patients while the current cohort contained stages 1–5. This discrepancy may have affected the KFRE performance. Forth, the generalizability of this model has not been tested on any external data due to the lack of such resource in this early feasibility study. Therefore, additional efforts are required to improve and validate this model before any clinical translation. Finally, although a simple model without urine variables is feasible and convenient, model predictability may benefit from a greater variety of clinical features, such as urine tests, imaging, or biopsy. Future works should include training ML models with additional features using a large dataset, and validating them on external patients.

应注意几个限制。首先，该队列由不到 1000 例受试者组成，ESKD 仅发生在其中一小部分受试者中，两者都可能影响模型性能，如前所述。其次，虽然本研究旨在评估 ESKD 预测模型的可行性，但没有任何尿液变量，这部分是由于我们研究所在建立该队列时缺乏定量尿液检测。随着现场尿液测试变得越来越受欢迎，尿液功能，如 ACR 将作为其他实验室测试访问和方便。预计它们将在更具预测性的模型中发挥关键作用。第三，KFRE 先前是在 3-5 期 CKD 患者中建立的，而当前队列包含 1-5 期。这一差异可能影响了 KFRE 的业绩。第四，由于在早期可行性研究中缺乏此类资源，因此尚未在任何外部数据上测试该模型的通用性。 因此，在任何临床转化之前，需要额外的努力来改进和验证该模型。最后，虽然没有尿液变量的简单模型是可行和方便的，但模型的可预测性可能受益于更多种类的临床特征，如尿液检查，成像或活检。未来的工作应该包括使用大型数据集训练具有额外功能的 ML 模型，并在外部患者身上验证它们。

In conclusion, this study showed the feasibility of ML in evaluating the prognosis of CKD based on easily accessible features. Logistic regression, naïve Bayes and random forest demonstrated comparable predictability to the KFRE in this study. These ML models also had greater sensitivity scores that were potentially advantageous for patient screenings. Future studies include performing external validation and improving the model with additional predictor variables.

总之，本研究显示了 ML 在基于易于获得的特征评估 CKD 预后方面的可行性。在这项研究中，逻辑回归、朴素贝叶斯和随机森林表现出与 KFRE 相当的预测性。这些 ML 模型也具有更高的灵敏度评分，这对患者筛查可能是有利的。未来的研究包括进行外部验证和改进模型与额外的预测变量。

