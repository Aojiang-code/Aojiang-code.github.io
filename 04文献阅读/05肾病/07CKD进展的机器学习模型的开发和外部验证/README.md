# CKD进展的机器学习模型的开发和外部验证


## 一、文献信息

|项目|内容|
| ---- | ---- |
|标题|Development and External Validation of a Machine Learning Model for Progression of CKD|
|作者|Thomas Ferguson, Pietro Ravani, Manish M. Sood, Alix Clarke, Paul Komenda, Claudio Rigatto, Navdeep Tangri|
|发表时间|2022-05-13|
|国家|加拿大|
|分区|Q1|
|影响因子|5.678|
|摘要|本研究旨在开发和外部验证一种基于随机森林的机器学习模型，用于预测慢性肾脏病（CKD）的进展，包括估算肾小球滤过率（eGFR）下降40%或肾衰竭。研究利用人口基础队列数据，结合人口统计学和实验室数据，模型在内部测试和外部验证中均表现出良好的区分能力和校准性能。|
|关键词|CKD进展, 机器学习, 预测模型|
|期刊名称|Kidney International Reports|
|卷号/期号|7:1772–1781|
|DOI|10.1016/j.ekir.2022.05.004|
|研究方法|机器学习模型开发与外部验证|
|数据来源|加拿大曼尼托巴省和阿尔伯塔省的行政数据库|
|研究结果|模型在2年和5年的预测中AUC分别为0.88和0.84，外部验证中AUC分别为0.87和0.84，能够准确预测CKD进展事件。|
|研究结论|基于常规实验室数据的机器学习模型能够准确预测CKD进展，可用于临床和研究场景。|
|研究意义|为临床决策提供新的预测工具，优化患者管理，提高卫生系统效率|
|阅读开始时间|20250216 22|
|阅读结束时间|20250216 23|
|时刻|晚上|
|星期|星期日|
|天气|多云|



## 二、核心内容
这篇文献的核心内容是开发和外部验证一种基于随机森林的机器学习模型，用于预测慢性肾脏病（CKD）进展至肾衰竭或估算肾小球滤过率（eGFR）下降40%的风险。以下是文献的主要内容总结：

### 研究背景
慢性肾脏病（CKD）是全球性健康问题，影响超过8.5亿成年人，与高发病率、死亡率和医疗成本相关。尽管只有少数CKD患者最终发展为肾衰竭，但进展至更晚期CKD阶段的患者会带来显著的健康负担。准确预测CKD进展风险对于改善患者预后、优化治疗决策和提高卫生系统效率至关重要。然而，现有的预测工具（如Kidney Failure Risk Equation, KFRE）仅适用于晚期CKD（G3 - G5），且仅考虑需要透析的肾衰竭结局。因此，开发适用于所有CKD阶段（G1 - G5）的预测模型具有重要意义。

### 研究目的
本研究旨在开发和外部验证一种基于随机森林的机器学习模型，利用常规收集的实验室数据和人口统计学信息，预测CKD患者eGFR下降40%或肾衰竭的风险。

### 研究方法
- **数据来源**：
    - **开发队列**：来自加拿大曼尼托巴省的人口基础队列，时间跨度为2006年4月1日至2016年12月31日，共纳入77,196名患者。
    - **验证队列**：来自加拿大阿尔伯塔省的外部验证队列，共107,097名患者。
- **纳入标准**：估算肾小球滤过率（eGFR）>10 ml/min/1.73m²，且有尿白蛋白/肌酐比值（ACR）数据。
- **排除标准**：既往有肾衰竭（透析或移植）病史的患者。
- **预测变量**：
    - 主要变量包括年龄、性别、eGFR、尿ACR。
    - 还纳入了18项其他实验室指标，如血细胞计数、生化指标、肝酶等。
- **结局定义**：
    主要结局为eGFR下降40%或肾衰竭（定义为开始慢性透析、肾移植或eGFR<10 ml/min/1.73m²）。
- **模型开发与验证**：
    - 使用随机森林模型，基于R语言的Fast Unified Random Forest for Survival, Regression, and Classification包。
    - 数据分为训练集（70%）和测试集（30%），并在外部队列中进行验证。
    - 评估指标包括AUC、Brier评分和校准图。

### 研究结果
- **模型性能**：
    - **内部测试队列**：
        - 1年预测AUC为0.90，5年预测AUC为0.84。
        - Brier评分在1年和5年分别为0.02和0.07。
    - **外部验证队列**：
        - 1年预测AUC为0.87，5年预测AUC为0.84。
        - Brier评分在1年和5年分别为0.01和0.04。
    - 模型在校准图中表现良好，预测风险与实际风险高度一致。
- **风险分层**：
在内部和外部验证队列中，高风险（前10%、15%、20%）和低风险（后30%、45%、50%）患者的敏感性、特异性、阳性预测值（PPV）和阴性预测值（NPV）均表现出良好的预测能力。
例如，在2年预测中，前10%风险患者的敏感性为58%，特异性为92%，PPV为25%。
- **变量重要性**：
尿ACR、eGFR、尿素、血红蛋白、年龄、血清白蛋白等是模型中最具影响力的变量。

### 研究结论
基于常规实验室数据的随机森林模型能够准确预测CKD患者的eGFR下降40%或肾衰竭风险，且在内部测试和外部验证中均表现出良好的性能。该模型优于现有的传统预测工具（如KFRE），并具有临床和研究应用的潜力。未来需要在更多样化的患者群体中进一步验证模型的性能，并探索其在临床实践中的应用价值。

### 研究意义
- **临床意义**：
    - 该模型可帮助识别早期CKD患者中高风险进展的个体，从而优化治疗决策，延缓或预防肾衰竭的发生。
    - 为临床医生提供了一种基于常规实验室数据的预测工具，便于在电子健康记录或实验室信息系统中集成。
- **研究意义**：
    - 该模型可应用于临床试验中，帮助富集高风险人群，提高研究效率。
    - 为未来开发更精准的CKD预测工具提供了方法学基础。

### 研究局限性
- 模型开发和验证均基于加拿大人群，需在其他地区进一步验证。
- 模型依赖于尿蛋白定量数据，可能不适用于缺乏此类数据的地区。
- 目前缺乏在线计算器或电子健康记录集成工具，限制了其广泛应用。
- 未来需要进一步研究模型指导下的治疗策略的成本效益和安全性。

## 三、文章小结

### Introduction
- **背景**：慢性肾脏病（CKD）影响全球超过8.5亿人，与高发病率、死亡率和医疗成本相关。虽然只有少数CKD患者最终发展为肾衰竭，但进展至晚期CKD的患者会带来显著的健康负担。
- **研究目的**：开发并外部验证一种基于机器学习的预测模型，用于预测CKD患者的eGFR下降40%或肾衰竭的风险。该模型将基于常规实验室数据和人口统计学信息，适用于所有CKD阶段（G1 - G5）。
- **临床意义**：准确预测CKD进展风险有助于改善患者预后、优化治疗决策，并提高卫生系统效率。

### Methods
#### Study Population Development Cohort
- **数据来源**：开发队列来自加拿大曼尼托巴省的人口基础队列（2006年4月1日至2016年12月31日），共纳入77,196名患者。
- **纳入标准**：年龄≥18岁，有eGFR检测记录，且有尿ACR或尿蛋白/肌酐比值（PCR）数据。
- **排除标准**：既往有肾衰竭（透析或移植）病史的患者。
- **eGFR计算**：使用CKD - Epidemiology Collaboration方程计算eGFR。

#### Validation Cohort
- **数据来源**：验证队列来自加拿大阿尔伯塔省的行政数据库，共107,097名患者。
- **时间范围**：数据覆盖2009年4月1日至2016年12月31日。
- **数据限制**：由于部分实验室数据从2009年起才完全覆盖，因此验证队列的起始时间为2009年。

#### Variables
- **独立变量**：包括年龄、性别、eGFR、尿ACR等。此外，还纳入了18项其他实验室指标，如血细胞计数、生化指标、肝酶等。
- **数据处理**：对于缺失数据，使用Ishwaran等人的方法进行填补。
- **结局变量**：主要结局为eGFR下降40%或肾衰竭（定义为开始慢性透析、肾移植或eGFR<10 ml/min/1.73m²）。

#### Statistical Analysis
- **模型开发**：使用随机森林模型，基于R语言的Fast Unified Random Forest for Survival, Regression, and Classification包。
- **数据分割**：数据分为训练集（70%）和测试集（30%），并在外部队列中进行验证。
- **评估指标**：使用AUC、Brier评分和校准图评估模型性能。
- **敏感性分析**：与传统Cox比例风险模型（包括eGFR、尿ACR、糖尿病、高血压等变量）进行比较。

### Results
#### Cohort Selection
- **开发队列**：共77,196名患者，其中54,037名用于训练，23,159名用于测试。
- **验证队列**：共107,097名患者，随机抽样自321,396名患者。

#### Cohort Description
- **开发队列**：平均年龄59.3岁，平均eGFR为82.2 ml/min/1.73m²，中位尿ACR为1.1 mg/mmol，48%为男性，45%有糖尿病。
- **验证队列**：平均年龄55.5岁，平均eGFR为86.0 ml/min/1.73m²，中位尿ACR为0.8 mg/mmol，53%为男性，41%有糖尿病。

#### Model Performance in Internal Testing Cohort
- **AUC和Brier评分**：
    - 1年预测AUC为0.90，Brier评分为0.02。
    - 5年预测AUC为0.84，Brier评分为0.07。
- **校准和风险分层**：模型在校准图中表现良好，预测风险与实际风险高度一致。高风险患者（前10%、15%、20%）的敏感性、特异性、PPV和NPV表现出良好的预测能力。

#### Model Performance in External Validation
- **AUC和Brier评分**：
    - 1年预测AUC为0.87，Brier评分为0.01。
    - 5年预测AUC为0.84，Brier评分为0.04。
- **校准和风险分层**：外部验证队列中模型表现良好，预测风险与实际风险高度一致。

#### Sensitivity Analyses
- **比较模型**：随机森林模型优于传统Cox比例风险模型（C统计量0.84 vs. 0.78）。
- **数据限制**：即使仅使用指数日期前12个月的实验室数据，模型性能也未受影响。

### Discussion
- **研究意义**：开发的随机森林模型基于常规实验室数据，能够准确预测CKD进展，优于现有的传统预测工具（如KFRE）。该模型在临床和研究中具有重要应用价值。
- **临床应用**：该模型可帮助识别早期CKD患者中高风险进展的个体，从而优化治疗决策，延缓或预防肾衰竭的发生。
- **研究应用**：该模型可用于临床试验中富集高风险人群，提高研究效率。
- **局限性**：模型开发和验证均基于加拿大人群，需在其他地区进一步验证。此外，模型依赖于尿蛋白定量数据，可能不适用于缺乏此类数据的地区。
- **未来方向**：需要进一步研究模型指导下的治疗策略的成本效益和安全性，并开发在线计算器或电子健康记录集成工具以促进广泛应用。

### Conclusion
总结：基于常规实验室数据的随机森林模型能够准确预测CKD患者的eGFR下降40%或肾衰竭风险，且在内部测试和外部验证中均表现出良好的性能。该模型具有重要的临床和研究应用潜力，未来需进一步验证和推广。

### Disclosure
利益声明：作者报告了与研究相关的资助来源和潜在的利益冲突，包括与制药公司和研究机构的合作关系。

### Acknowledgments
致谢：感谢曼尼托巴省和阿尔伯塔省健康数据库的支持，以及相关研究资助机构的资助。

### Data Sharing
数据共享：研究数据由曼尼托巴省和阿尔伯塔省健康数据库提供，数据集以编码形式安全存储，仅对符合保密要求的研究人员开放。

### Author Contributions
作者贡献：所有作者均参与了研究设计、数据分析、论文撰写和修订，并对研究结果负责。

### Supplementary Material
补充材料：包括队列选择流程图、变量缺失情况、校准图、敏感性分析结果等。


## 四、主要方法和实施计划

### Methods

#### 1. Study Population Development Cohort
- **数据来源**：
  - 开发队列来自加拿大曼尼托巴省的人口基础队列，数据由曼尼托巴健康政策中心（Manitoba Centre for Health Policy）提供。
  - 研究时间跨度为2006年4月1日至2016年12月31日。
- **纳入标准**：
  - 年龄≥18岁。
  - 至少有一次门诊eGFR检测记录。
  - 至少有一次尿白蛋白/肌酐比值（ACR）或尿蛋白/肌酐比值（PCR）检测记录。
  - 在研究前至少有1年的曼尼托巴健康注册记录。
- **排除标准**：
  - 既往有肾衰竭病史（透析或移植）的患者。
- **eGFR计算**：
  - 使用CKD-Epidemiology Collaboration方程从血清肌酐检测结果中计算eGFR。

#### 2. Validation Cohort
- **数据来源**：
  - 验证队列来自加拿大阿尔伯塔省的行政数据库，数据由阿尔伯塔健康数据库提供。
  - 研究时间跨度为2009年4月1日至2016年12月31日。
- **纳入标准**：
  - 至少有一次可计算的eGFR记录。
  - 至少有一次ACR（或PCR）检测记录。
  - 在研究前至少有1年的阿尔伯塔健康注册记录。
- **数据限制**：
  - 由于部分实验室数据从2009年起才完全覆盖，因此验证队列的起始时间为2009年。
  - 随机抽取了验证队列的三分之一进行最终分析，以减少计算时间。

#### 3. Variables
- **独立变量**：
  - **主要变量**：年龄、性别、eGFR、尿ACR。
  - **其他实验室指标**：包括血细胞计数、生化指标、肝酶等，共18项实验室检测结果。
  - **数据处理**：对于缺失数据，使用Ishwaran等人的方法进行填补。
- **结局变量**：
  - **主要结局为eGFR下降40%或肾衰竭**：
    - **eGFR下降40%**：定义为从基线eGFR下降≥40%，需要第二次确认性检测结果（在首次检测后90天至2年内）。
    - **肾衰竭**：定义为开始慢性透析、肾移植或eGFR<10 ml/min/1.73m²。
  - **数据来源**：实验室数据、医疗索赔记录、住院记录等。

#### 4. Statistical Analysis
- **模型开发**：
  - 使用随机森林模型，基于R语言的Fast Unified Random Forest for Survival, Regression, and Classification包。
  - 数据分为训练集（70%）和测试集（30%），并在外部队列中进行验证。
- **模型评估**：
  - 使用AUC（曲线下面积）评估模型的区分能力。
  - 使用Brier评分评估模型的预测精度。
  - 使用校准图（calibration plot）评估模型的预测风险与实际风险的一致性。
- **敏感性分析**：
  - 与传统Cox比例风险模型进行比较，包括基于指南的模型（heatmap model）和临床模型（clinical model）。
  - 在外部验证队列中，仅使用指数日期前12个月的实验室数据进行模型验证。
- **变量重要性评估**：
  - 使用随机森林模型评估变量的重要性，重点关注对模型影响最大的5个变量。

### 实施计划

#### 数据收集与整理
- 从曼尼托巴省和阿尔伯塔省的行政数据库中提取人口统计学和实验室数据。
- 清洗数据，处理缺失值，确保数据质量。

#### 模型开发
- 使用随机森林算法开发预测模型。
- 将数据分为训练集和测试集，训练集用于模型训练，测试集用于初步验证。

#### 模型验证
- 在外部验证队列中验证模型性能，确保模型的泛化能力。
- 使用AUC、Brier评分和校准图评估模型的区分能力和预测精度。

#### 敏感性分析
- 与传统预测模型（如Cox比例风险模型）进行比较，评估随机森林模型的优越性。
- 在不同时间窗口（如指数日期前12个月）内验证模型性能。

#### 结果分析与报告
- 分析模型在不同时间点（1 - 5年）的预测性能。
- 评估模型在高风险和低风险人群中的敏感性、特异性、PPV和NPV。

#### 未来计划
- 在更多样化的患者群体中进一步验证模型。
- 开发在线计算器或电子健康记录集成工具，促进模型的广泛应用。

### 总结
这篇文献详细描述了开发和验证一个基于随机森林的机器学习模型的方法和实施计划。研究团队通过严格的数据收集、模型开发和验证流程，确保了模型的准确性和泛化能力。未来的研究将集中在进一步验证模型的性能，并探索其在临床实践中的应用价值。



## 五、重要变量和数据(英文展示)
以下是根据文献中提供的信息抓取的主要变量信息，包括连续变量的均值、方差、中位数，以及分类变量的构成比和频率。这些信息以Markdown表格的形式呈现，方便后续使用Python代码进行数据模拟。

### Continuous Variables（连续变量）
| Variable | Mean ± SD | Median (IQR) | Unit |
| ---- | ---- | ---- | ---- |
| Age | 59.3 ± 17.0 | - | Years |
| eGFR | 82.2 ± 27.1 | - | ml/min/1.73m² |
| Urea | 6.6 ± 4.0 | - | mmol/l |
| Serum Hemoglobin | 134 ± 19 | - | g/l |
| Glucose | 7.9 ± 4.1 | - | mmol/l |
| Serum Albumin | 37 ± 6 | - | g/l |
| Urine ACR | - | 1.1 (0.5–4.7) | mg/mmol |

### Categorical Variables（分类变量）
| Variable | Category | Frequency | Percentage |
| ---- | ---- | ---- | ---- |
| Sex | Male | 25,829 | 48% |
|  | Female | 28,208 | 52% |
| Diabetes | Yes | 24,460 | 45% |
|  | No | 29,577 | 55% |
| Hypertension | Yes | 37,701 | 70% |
|  | No | 16,336 | 30% |
| Congestive Heart Failure | Yes | 2,840 | 5% |
|  | No | 51,197 | 95% |
| Prior Stroke | Yes | 1,937 | 4% |
|  | No | 52,099 | 96% |
| Myocardial Infarction | Yes | 1,380 | 3% |
|  | No | 52,657 | 97% |

### Event Outcomes（事件结局）
| Outcome | Frequency | Percentage |
| ---- | ---- | ---- |
| 40% Decline in eGFR | 3,965 | 7.3% |
| Kidney Failure | 246 | 0.5% |
| Composite Outcome (40% Decline or Kidney Failure) | 4,211 | 7.8% |


## 五、重要变量和数据(中文展示)
以下是根据文献中提供的信息提取的主要变量信息，涵盖连续变量的均值、标准差、中位数，以及分类变量的构成比和频数。这些信息以Markdown表格的形式呈现，以便后续使用Python代码进行数据模拟。

### 连续变量
| 变量 | 均值 ± 标准差 | 中位数（四分位距） | 单位 |
| ---- | ---- | ---- | ---- |
| 年龄 | 59.3 ± 17.0 | - | 岁 |
| 估算肾小球滤过率（eGFR） | 82.2 ± 27.1 | - | 毫升/分钟/1.73平方米 |
| 尿素 | 6.6 ± 4.0 | - | 毫摩尔/升 |
| 血清血红蛋白 | 134 ± 19 | - | 克/升 |
| 葡萄糖 | 7.9 ± 4.1 | - | 毫摩尔/升 |
| 血清白蛋白 | 37 ± 6 | - | 克/升 |
| 尿白蛋白肌酐比（Urine ACR） | - | 1.1（0.5 - 4.7） | 毫克/毫摩尔 |

### 分类变量
| 变量 | 类别 | 频数 | 百分比 |
| ---- | ---- | ---- | ---- |
| 性别 | 男性 | 25,829 | 48% |
|  | 女性 | 28,208 | 52% |
| 糖尿病 | 是 | 24,460 | 45% |
|  | 否 | 29,577 | 55% |
| 高血压 | 是 | 37,701 | 70% |
|  | 否 | 16,336 | 30% |
| 充血性心力衰竭 | 是 | 2,840 | 5% |
|  | 否 | 51,197 | 95% |
| 既往中风 | 是 | 1,937 | 4% |
|  | 否 | 52,099 | 96% |
| 心肌梗死 | 是 | 1,380 | 3% |
|  | 否 | 52,657 | 97% |

### 事件结局
| 结局 | 频数 | 百分比 |
| ---- | ---- | ---- |
| 估算肾小球滤过率下降40% | 3,965 | 7.3% |
| 肾衰竭 | 246 | 0.5% |
| 复合结局（估算肾小球滤过率下降40% 或 肾衰竭） | 4,211 | 7.8% |


## 六、模拟数据1
以下是一个Python代码示例，用于根据上述抓取的变量信息模拟数据，并将其保存为CSV文件。代码使用 `pandas` 库来处理数据，并使用 `numpy` 库来生成随机数据。

```python
import pandas as pd
import numpy as np
import os

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 模拟数据的样本量
n_samples = 77196

# 连续变量的模拟
age = np.random.normal(loc=59.3, scale=17.0, size=n_samples)
eGFR = np.random.normal(loc=82.2, scale=27.1, size=n_samples)
urea = np.random.normal(loc=6.6, scale=4.0, size=n_samples)
serum_hemoglobin = np.random.normal(loc=134, scale=19, size=n_samples)
glucose = np.random.normal(loc=7.9, scale=4.1, size=n_samples)
serum_albumin = np.random.normal(loc=37, scale=6, size=n_samples)
urine_ACR = np.random.lognormal(mean=np.log(1.1), sigma=0.5, size=n_samples)  # 使用对数正态分布模拟偏态数据

# 分类变量的模拟
sex = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.48, 0.52])
diabetes = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.45, 0.55])
hypertension = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.70, 0.30])
congestive_heart_failure = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.05, 0.95])
prior_stroke = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.04, 0.96])
myocardial_infarction = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.03, 0.97])

# 事件结局的模拟
composite_outcome = np.random.choice([1, 0], size=n_samples, p=[0.078, 1 - 0.078])

# 创建DataFrame
data = pd.DataFrame({
    'Age': age,
    'eGFR': eGFR,
    'Urea': urea,
    'Serum_Hemoglobin': serum_hemoglobin,
    'Glucose': glucose,
    'Serum_Albumin': serum_albumin,
    'Urine_ACR': urine_ACR,
    'Sex': sex,
    'Diabetes': diabetes,
    'Hypertension': hypertension,
    'Congestive_Heart_Failure': congestive_heart_failure,
    'Prior_Stroke': prior_stroke,
    'Myocardial_Infarction': myocardial_infarction,
    'Composite_Outcome': composite_outcome
})

# 确保保存路径存在
save_path = r"04文献阅读\05肾病\07CKD进展的机器学习模型的开发和外部验证\01data"
os.makedirs(save_path, exist_ok=True)

# 保存为CSV文件
file_path = os.path.join(save_path, "simulated_ckd_data.csv")
data.to_csv(file_path, index=False)

print(f"数据已成功保存到路径：{file_path}")
```

### 代码说明
#### 连续变量
- 使用 `numpy.random.normal` 生成正态分布数据。
- 对于偏态分布（如尿白蛋白/肌酐比值，Urine_ACR），使用 `numpy.random.lognormal` 生成对数正态分布数据。

#### 分类变量
使用 `numpy.random.choice` 生成分类变量，指定每个类别的概率。

#### 事件结局
使用 `numpy.random.choice` 生成二分类结局，模拟复合结局（eGFR下降40% 或肾衰竭）的发生概率。

#### 保存路径
- 使用 `os.makedirs` 确保保存路径存在。
- 使用 `pandas.DataFrame.to_csv` 将数据保存为CSV文件。
### 运行代码
将上述代码保存为 `.py` 文件并运行，或者直接在Jupyter Notebook中运行。运行后，模拟数据将被保存到指定路径。

如果需要调整样本量或其他参数，请修改代码中的 `n_samples` 或其他相关参数。

## 六、模拟数据2
# 模拟数据并保存为CSV格式的Python脚本

以下是一个Python脚本，用于模拟上述数据并保存为CSV格式。脚本名称为 `simulate_ckd_data.py`。它将生成一个包含上述变量的模拟数据集，并将其保存到指定路径。

### simulate_ckd_data.py
```python
import pandas as pd
import numpy as np
import os

# 设置保存路径
save_path = r"04文献阅读\05肾病\07CKD进展的机器学习模型的开发和外部验证\01data"
os.makedirs(save_path, exist_ok=True)
file_name = os.path.join(save_path, "simulated_ckd_data.csv")

# 设置随机种子以确保可重复性
np.random.seed(42)

# 模拟数据的样本量
n_samples = 10000

# 模拟连续变量
age = np.random.normal(loc=59.3, scale=17.0, size=n_samples)
eGFR = np.random.normal(loc=82.2, scale=27.1, size=n_samples)
urea = np.random.normal(loc=6.6, scale=4.0, size=n_samples)
serum_hemoglobin = np.random.normal(loc=134, scale=19, size=n_samples)
glucose = np.random.normal(loc=7.9, scale=4.1, size=n_samples)
serum_albumin = np.random.normal(loc=37, scale=6, size=n_samples)
urine_ACR = np.random.lognormal(mean=np.log(1.1), sigma=0.6, size=n_samples)  # Log-normal distribution

# 模拟分类变量
sex = np.random.choice(["Male", "Female"], size=n_samples, p=[0.48, 0.52])
diabetes = np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55])
hypertension = np.random.choice(["Yes", "No"], size=n_samples, p=[0.70, 0.30])
congestive_heart_failure = np.random.choice(["Yes", "No"], size=n_samples, p=[0.05, 0.95])
prior_stroke = np.random.choice(["Yes", "No"], size=n_samples, p=[0.04, 0.96])
myocardial_infarction = np.random.choice(["Yes", "No"], size=n_samples, p=[0.03, 0.97])

# 模拟结局变量
composite_outcome = np.random.choice([1, 0], size=n_samples, p=[0.078, 1 - 0.078])

# 创建DataFrame
data = pd.DataFrame({
    "Age": age,
    "eGFR": eGFR,
    "Urea": urea,
    "Serum_Hemoglobin": serum_hemoglobin,
    "Glucose": glucose,
    "Serum_Albumin": serum_albumin,
    "Urine_ACR": urine_ACR,
    "Sex": sex,
    "Diabetes": diabetes,
    "Hypertension": hypertension,
    "Congestive_Heart_Failure": congestive_heart_failure,
    "Prior_Stroke": prior_stroke,
    "Myocardial_Infarction": myocardial_infarction,
    "Composite_Outcome": composite_outcome
})

# 保存为CSV文件
data.to_csv(file_name, index=False)
print(f"Simulated data saved to {file_name}")
```

### 代码说明
#### 路径设置
- 使用 `os.makedirs` 确保保存路径存在。
- 文件保存路径为 `04文献阅读\05肾病\07CKD进展的机器学习模型的开发和外部验证\01data`。

#### 数据模拟
- **连续变量**：（如年龄、eGFR等）使用正态分布或对数正态分布模拟。
- **分类变量**：（如性别、糖尿病等）使用 `np.random.choice` 模拟，根据文献中提供的比例分配概率。

#### 结局变量
综合结局（Composite Outcome）使用二项分布模拟，概率为7.8%。

#### 保存数据
使用 `pandas` 将数据保存为CSV文件。

### 运行脚本
将上述代码保存为 `simulate_ckd_data.py` 文件，并在Python环境中运行。运行后，模拟数据将保存到指定路径下的 `simulated_ckd_data.csv` 文件中。

如果需要调整样本量或其他参数，可以直接修改脚本中的 `n_samples` 或其他相关设置。

## 复现计划

### 为复现文献结果的Python脚本
为了复现文献中的结果，我将提供一个Python脚本，用于加载模拟数据、开发随机森林模型、进行内部验证和外部验证，并评估模型性能。脚本名称为 `reproduce_ckd_model.py`。

以下是完整的代码实现：
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, calibration_curve
import matplotlib.pyplot as plt

# 加载模拟数据
data_path = r"04文献阅读\05肾病\07CKD进展的机器学习模型的开发和外部验证\01data\simulated_ckd_data.csv"
data = pd.read_csv(data_path)

# 定义特征和目标变量
features = [
    "Age", "eGFR", "Urea", "Serum_Hemoglobin", "Glucose", "Serum_Albumin", "Urine_ACR",
    "Sex", "Diabetes", "Hypertension", "Congestive_Heart_Failure", "Prior_Stroke", "Myocardial_Infarction"
]

# 处理分类变量
data["Sex"] = data["Sex"].map({"Male": 1, "Female": 0})
data["Diabetes"] = data["Diabetes"].map({"Yes": 1, "No": 0})
data["Hypertension"] = data["Hypertension"].map({"Yes": 1, "No": 0})
data["Congestive_Heart_Failure"] = data["Congestive_Heart_Failure"].map({"Yes": 1, "No": 0})
data["Prior_Stroke"] = data["Prior_Stroke"].map({"Yes": 1, "No": 0})
data["Myocardial_Infarction"] = data["Myocardial_Infarction"].map({"Yes": 1, "No": 0})

X = data[features]
y = data["Composite_Outcome"]

# 数据分割：训练集（70%）和测试集（30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 开发随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 内部验证
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)
print(f"Internal Validation - AUC: {auc:.3f}, Brier Score: {brier:.3f}")

# 绘制校准图
def plot_calibration_curve(y_true, y_pred_proba, title):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label="Random Forest")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()
    plt.show()

plot_calibration_curve(y_test, y_pred_proba, "Calibration Curve - Internal Validation")

# 外部验证
# 假设外部验证数据集与内部测试集相似
y_ext_pred_proba = model.predict_proba(X_test)[:, 1]
ext_auc = roc_auc_score(y_test, y_ext_pred_proba)
ext_brier = brier_score_loss(y_test, y_ext_pred_proba)
print(f"External Validation - AUC: {ext_auc:.3f}, Brier Score: {ext_brier:.3f}")

plot_calibration_curve(y_test, y_ext_pred_proba, "Calibration Curve - External Validation")

# 变量重要性
importances = model.feature_importances_
feature_names = features
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.xticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx], rotation=90)
plt.xlabel("Relative Importance")
plt.title("Feature Importances")
plt.show()
```

### 代码说明
#### 数据加载
从指定路径加载模拟数据。
数据路径为 `04文献阅读\05肾病\07CKD进展的机器学习模型的开发和外部验证\01data\simulated_ckd_data.csv`。

#### 特征处理
将分类变量（如性别、糖尿病等）转换为数值变量。

#### 模型开发
使用随机森林分类器（RandomForestClassifier）开发模型。
数据分为训练集（70%）和测试集（30%）。

#### 内部验证
使用AUC和Brier评分评估模型性能。
绘制校准图以评估预测风险与实际风险的一致性。

#### 外部验证
假设外部验证数据集与内部测试集相似，使用相同的测试集进行外部验证。
评估AUC和Brier评分，并绘制校准图。

#### 变量重要性
使用随机森林模型评估变量的重要性，并绘制条形图。

### 运行脚本
将上述代码保存为 `reproduce_ckd_model.py` 文件，并在Python环境中运行。运行后，脚本将输出内部验证和外部验证的AUC和Brier评分，并绘制校准图和变量重要性图。
