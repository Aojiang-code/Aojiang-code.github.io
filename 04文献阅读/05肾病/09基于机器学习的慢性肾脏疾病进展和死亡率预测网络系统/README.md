# 基于机器学习的慢性肾脏疾病进展和死亡率预测网络系统


## 一、文献信息


|项目|内容|
| ---- | ---- |
|标题|Machine-learning-based Web system for the prediction of chronic kidney disease progression and mortality|
|作者|Eiichiro Kanda, Bogdan Iuliu Epureanu, Taiji Adachi, Naoki Kashihara|
|发表时间|2023-01-18|
|国家|日本|
|分区|Q1|
|影响因子|未明确提供（需查询期刊影响因子）|
|摘要|本研究开发了一种基于机器学习的网络系统，用于预测慢性肾脏病（CKD）患者的疾病进展和死亡率。通过开发16种机器学习模型，使用随机森林（RF）、梯度提升决策树（GB）和极端梯度提升（XG）算法，基于22个变量或精选变量预测主要结局（ESKD或死亡）。研究结果表明，该系统在临床实践中具有较高的预测准确性。|
|关键词|机器学习, 慢性肾脏病, 疾病进展, 死亡率, 风险预测|
|期刊名称|PLOS Digital Health|
|卷号/期号|2(1): e0000188|
|DOI|10.1371/journal.pdig.0000188|
|研究方法|机器学习模型开发与验证|
|数据来源|川崎医科大学医院电子病历数据（2014 - 2017）和3年队列研究数据（2018 - 2020）|
|研究结果|两种随机森林模型（22变量和8变量）表现出高准确性，C统计量分别为0.932和0.930，优于传统预测工具KFRE。|
|研究结论|基于机器学习的网络系统能够准确预测CKD患者的ESKD或死亡风险，适用于临床实践。|
|研究意义|为临床决策提供新的预测工具，优化患者管理，填补了AI研究与临床实践之间的空白。|
|阅读开始时间|20250217 22|
|阅读结束时间|20250217 23|
|时刻|晚上|
|星期|星期一|
|天气|小雨|



## 二、核心内容
### 基于机器学习的慢性肾脏病患者疾病进展和死亡率预测网络系统研究

本文介绍了一种基于机器学习的网络系统，用于预测慢性肾脏病（CKD）患者的疾病进展和死亡率。研究团队开发了多个机器学习模型，并将其应用于临床实践的网络平台，旨在提高对CKD患者风险的预测能力，尤其是对于高风险患者。

#### 背景知识
慢性肾脏病（CKD）患者面临较高的终末期肾病（ESKD）和死亡风险。准确预测这些风险对于临床治疗和资源分配至关重要。尽管已有多种风险预测模型，但目前尚无基于机器学习的模型被广泛应用于临床实践。主要原因包括：模型变量过多、预测准确性不足以及与医院电子病历（EMR）系统的兼容性问题。

#### 研究方法
1. **数据来源**
研究使用了日本川崎医科大学医院的电子病历数据（2014 - 2017年）开发机器学习模型，并使用2018 - 2020年的队列研究数据进行验证。研究纳入了3,714名患者的重复测量数据（共66,981条记录）。

2. **模型开发**
研究团队开发了16个基于随机森林（RF）、梯度提升决策树（GB）和极端梯度提升（XG）的机器学习模型，使用22个变量或精选变量预测CKD患者的主要结局（ESKD或死亡）。模型开发分为两部分：基于基线数据和时间序列数据的模型。

3. **变量选择**
选择的变量包括年龄、性别、糖尿病、高血压、心血管病史、估算肾小球滤过率（eGFR）、血清白蛋白、钠、钾、钙、磷、低密度脂蛋白、尿酸水平、白细胞计数、血红蛋白水平、尿蛋白/肌酐比（UPCR）以及相关药物使用情况。

4. **模型训练与选择**
使用网格搜索算法优化模型参数，并通过置换重要性选择变量。最终选择了两个RF模型（RF_time_all和RF_time_v8）用于进一步验证和网络系统的开发。

#### 实验结果
1. **模型性能评估**
在模型选择阶段，RF_time_all（22个变量）和RF_time_v8（8个变量）模型表现出较高的预测准确性，C统计量分别为0.932（95% CI: 0.916 - 0.948）和0.930（95% CI: 0.915 - 0.945）。这些模型的预测能力显著高于传统的肾脏衰竭风险方程（KFRE）。

2. **风险预测验证**
在队列研究数据中，RF_time_all和RF_time_v8模型的C统计量分别为0.932和0.930，显示出对不同肾功能、年龄和糖尿病状态患者的高适用性。高风险组患者的生存概率显著低于低风险组，风险比（HR）分别为104.9和90.9。

3. **网络系统开发**
研究团队开发了一个基于Web的风险预测系统，用户可以通过输入患者数据快速获得预测结果，并根据风险水平获得相应的建议。该系统可通过网址http://160.16.88.112:8000/ 访问。

#### 关键结论
1. 基于机器学习的Web系统能够准确预测CKD患者的ESKD或死亡风险，并为临床实践提供新的风险评估工具。
2. 该系统通过简化变量输入和优化模型性能，克服了以往AI模型在临床应用中的局限性。
3. 研究结果表明，机器学习模型可以作为CKD患者风险评估的新指标，并为高风险患者提供治疗建议。

#### 研究局限性
1. 数据来源于日本，可能对非亚洲患者的适用性有限。
2. 模型未包含生活方式、饮食、家族史等非EMR数据。
3. 未纳入新药物（如SGLT2抑制剂）的数据，因为这些药物在日本的应用时间较短。
4. 未来需要通过干预性研究验证该系统对CKD治疗的实际影响。

## 三、文章小结
### Abstract
- 研究开发了一种基于机器学习的网络系统，用于预测慢性肾脏病（CKD）患者的疾病进展和死亡率。
- 使用随机森林（RF）、梯度提升决策树（GB）和极端梯度提升（XG）算法开发了16个风险预测模型，基于22个变量或精选变量。
- 在3年队列研究数据中验证，两个RF模型（22变量和8变量）表现出高准确性，C统计量分别为0.932和0.930。
- 开发的Web系统可实现临床实践中对CKD患者风险的快速预测。

### Author summary
- CKD患者面临高风险的终末期肾病（ESKD）和死亡，开发AI模型有助于筛选高风险患者。
- 研究解决了AI模型在临床应用中的挑战，包括变量过多、预测准确性不足和与电子病历（EMR）系统的兼容性问题。
- 研究开发了基于少量常用变量的AI模型，并在不同肾功能、年龄和糖尿病状态的患者亚组中验证了其性能。
- 开发的Web系统填补了AI研究与临床实践之间的空白。

### Introduction
- CKD患者面临ESKD、心血管疾病和死亡的高风险，疾病进展会加重这些风险。
- CKD的治疗成本高，对患者造成身体和经济负担，控制风险因素（如糖尿病、高血压）有助于延缓进展。
- 准确预测患者肾功能和生存期对高风险患者诊断和治疗策略制定至关重要。
- 以往的机器学习模型因变量过多或准确性不足未能广泛应用于临床，本研究旨在开发适用于临床的机器学习模型。

### Results
#### Model selection
- 比较了不同模型的C统计量，RF_time_all和RF_time_v8模型在预测1年、2年和3年的主要结局（ESKD或死亡）时表现出高准确性。
- 在不同亚组（基于eGFR、年龄和糖尿病状态）中，这些模型也显示出高C统计量。
- 选择了RF_time_all和RF_time_v8模型用于进一步验证和Web系统开发。

#### Validation of the machine - learning models using CKD cohort study data
- 使用独立的3年队列研究数据验证了RF_time_all和RF_time_v8模型的性能。
- 两个模型的C统计量分别为0.932和0.930，显著高于传统肾脏衰竭风险方程（KFRE）。
- 高风险组患者的生存概率显著低于低风险组，风险比（HR）分别为104.9和90.9。

#### Implementation of the models in clinical settings
- 开发了一个基于Web的风险预测系统，用户可通过输入患者数据快速获得预测结果。
- 系统基于RF模型，可通过PC或智能手机访问，网址为http://160.16.88.112:8000/ 。

### Discussion
- 研究开发的基于机器学习的Web系统能够准确预测CKD患者的预后，并为临床实践提供新的风险评估工具。
- 该系统通过简化变量输入和优化模型性能，克服了以往AI模型在临床应用中的局限性。
- 研究结果表明，机器学习模型可以作为CKD患者风险评估的新指标，并为高风险患者提供治疗建议。
- 研究局限性包括数据来源的地域限制、未包含生活方式等非EMR数据、未纳入新药物数据等。

### Materials and methods
#### Study design and ethics
- 研究使用了川崎医科大学医院的电子病历数据（2014 - 2017年）开发模型，并使用2018 - 2020年的队列研究数据进行验证。
- 研究获得了伦理委员会的批准，免除了知情同意。

#### Step (1) Development of machine - learning models
- 研究纳入了3,714名患者的重复测量数据（66,981条记录），开发了基于基线数据和时间序列数据的机器学习模型。
- 选择了22个常用变量，包括临床指标和药物使用情况。
- 使用随机森林、梯度提升决策树和极端梯度提升算法开发模型。

#### Step (2) Model selection
- 使用747名患者的基线数据对模型进行选择，通过C统计量评估模型性能。
- 在不同亚组（基于eGFR、年龄和糖尿病状态）中验证模型性能。

#### Step (3) Validation of the machine - learning models using data from a cohort study
- 使用独立的3年队列研究数据（26,906名患者）验证模型性能。
- 评估模型的C统计量和风险预测能力，比较高风险组和低风险组的生存概率。

#### Step (4) Development of a Web - based risk - prediction system and its specifications
- 开发了一个基于Web的风险预测系统，用户可通过输入患者数据获得预测结果。
- 系统支持不同国家的变量单位选择，并基于eGFR计算风险概率。

### Supporting information
提供了补充材料，包括模型开发和验证的详细图表、变量定义、亚组分析结果等。

### Author Contributions
描述了作者在研究中的具体贡献，包括概念设计、数据分析、软件开发、写作和编辑等。

### References
列出了研究中引用的相关文献，涉及CKD流行病学、风险预测模型、机器学习应用等领域。


## 四、主要方法和实施计划
### 基于机器学习的慢性肾脏病（CKD）风险预测模型开发与实施方法总结

#### 研究设计与伦理（Study design and ethics）
- **数据来源**：开发阶段使用川崎医科大学医院2014 - 2017年的电子病历（EMR）数据，验证阶段采用2018 - 2020年的独立队列研究数据。
- **伦理批准**：获得川崎医科大学伦理委员会批准，因研究的观察性质免除知情同意。
- **研究目标**：开发并验证基于机器学习的模型以预测CKD患者的疾病进展（ESKD）和死亡风险，并应用于临床实践的Web系统。

#### 模型开发（Development of machine - learning models）
- **研究人群（Study population）**
  - **数据预处理**：从医院服务器提取EMR数据，排除年龄小于20岁、正在接受透析治疗、患有恶性肿瘤或缺失基线数据的患者。
  - **数据集划分**：最终纳入3,714名患者的重复测量数据（66,981条记录），分为基线数据集（n = 3,714）和时间序列数据集（n = 66,981）。
  - **数据分配**：开发数据集（80%患者）用于模型训练和测试，选择数据集（20%患者）用于模型选择。
- **变量选择（Variables）**
  - **变量定义**：选取22个与CKD相关的常用变量，涵盖人口统计学信息、合并症、实验室指标（如eGFR、血清白蛋白、钾、磷等）和药物使用情况。
  - **数据处理**：缺失数据用最后一次观察值向前填补；尿白蛋白/肌酐比（UACR）未测量时，通过公式从尿蛋白/肌酐比（UPCR）估算。
  - **结局定义**：主要结局为ESKD或死亡，ESKD指开始肾脏替代治疗，死亡数据从EMR获取。

#### 模型开发与选择（Model development and selection）
- **机器学习模型（Machine - learning models）**
  - **算法选择**：采用随机森林（RF）、梯度提升决策树（GB）和极端梯度提升（XG）算法开发模型。
  - **模型训练**：借助Python的scikit - learn库，通过网格搜索算法优化模型参数（如树的数量、最大深度、样本分割数等）。
  - **变量选择**：基于排列重要性选关键变量，RF_time_all模型含22个变量，RF_time_v8模型含8个变量。
- **模型选择（Model selection）**
  - **数据集**：使用747名患者的基线数据进行模型选择。
  - **性能评估**：通过1000次自助法（bootstrap）计算C统计量及其95%置信区间，评估模型对1年、2年和3年主要结局的预测能力。
  - **亚组分析**：在不同亚组（高/低eGFR、糖尿病状态、年龄<65岁/≥65岁）中评估模型性能。
  - **风险预测**：使用Cox比例风险模型评估预测概率与实际风险的关系，确定高风险患者的截断值。

#### 模型验证（Validation of the machine - learning models）
- **验证数据集（Validation dataset）**
  - **数据来源**：使用2018 - 2020年的独立队列研究数据，纳入26,906名患者。
  - **数据处理**：对缺失数据采用多重插补法处理。
- **性能评估（Performance evaluation）**
  - **C统计量**：用1000次自助法评估模型对主要结局（ESKD或死亡）的预测准确性。
  - **生存分析**：借助Cox比例风险模型和Fine - Gray竞争风险模型评估预测概率与实际风险的关系。
  - **风险分层**：依模型预测概率将患者分高、低风险组，比较两组生存概率和风险比（HR）。

#### Web系统开发（Development of a Web - based risk - prediction system）
- **系统设计（System design）**
  - **功能**：用户能通过Web浏览器输入患者数据，系统基于机器学习模型实时计算风险预测概率并提供临床建议。
  - **技术实现**：系统基于Python开发，后端部署RF模型，支持多种设备（PC和智能手机）访问。
  - **用户体验**：系统支持不同国家的变量单位选择，通过eGFR计算风险概率以减少因肌酐测量方法不同导致的误差。
- **系统部署（System deployment）**
  - **网址**：开发的Web系统可通过http://160.16.88.112:8000/ 访问。
  - **目标**：为临床医生和研究人员提供快速、便捷的风险预测工具，辅助识别高风险患者并制定治疗策略。

#### 总结（Summary）
- **方法总结**：通过开发和验证基于机器学习的模型，构建了高准确性的CKD风险预测系统。该系统基于少量常用变量，便于临床实践应用。
- **实施计划**：未来研究方向包括扩大数据来源提升模型普适性、纳入更多临床变量（如生活方式和新药物数据），并通过干预性研究验证系统临床效果。

通过上述方法和实施计划，研究团队成功开发并验证了基于机器学习的CKD风险预测系统，为临床实践提供新工具。 

## 五、重要变量和数据(英文展示)
### 文献主要变量数据统计信息

#### 连续变量（Continuous Variables）
| Variable | Mean ± SD | Median (IQR) |
| ---- | ---- | ---- |
| Age (years) | 61.2 ± 16.3 | 61.2 (16.3) |
| eGFR (mL/min/1.73m²) | 73.3 ± 31.0 | 73.3 (31.0) |
| Albumin (g/dL) | 4.2 ± 0.5 | 4.2 (0.5) |
| Sodium (mmol/L) | 140.3 ± 2.6 | 140.3 (2.6) |
| Potassium (mmol/L) | 4.2 ± 0.4 | 4.2 (0.4) |
| Calcium (mg/dL) | 9.2 ± 0.5 | 9.2 (0.5) |
| Phosphorus (mg/dL) | 3.4 ± 0.7 | 3.4 (0.7) |
| LDL (mg/dL) | 111.0 ± 32.1 | 111.0 (32.1) |
| Uric acid (mg/dL) | 5.2 ± 1.5 | 5.2 (1.5) |
| WBC (10³/μL) | 6.3 ± 3.0 | 6.3 (3.0) |
| Hemoglobin (g/dL) | 13.6 ± 1.8 | 13.6 (1.8) |
| UPCR (g/gCre) | -0.27 [0.11, 1.00] | - |

#### 分类变量（Categorical Variables）
| Variable | Frequency (n) | Percentage (%) |
| ---- | ---- | ---- |
| Male | 13,000 | 51.1 |
| Diabetes Mellitus (DM) | 5,305 | 20.9 |
| Hypertension | 5,234 | 20.6 |
| Cardiovascular Disease (CVD) | 159 | 0.6 |
| RAASI Use (%) | 4,407 | 17.3 |
| Phosphorus Absorbent Use (%) | 209 | 0.8 |
| Vitamin D Use (%) | 653 | 2.6 |
| Statin Use (%) | 4,162 | 16.4 |
| Uric-acid-lowering Medicines Use (%) | 2,272 | 8.9 |
| ESA Use (%) | 665 | 2.6 |

#### 说明
- 连续变量：提供了均值（Mean）、标准差（SD）、中位数（Median）和四分位数范围（IQR）。
- 分类变量：提供了频数（Frequency）和百分比（Percentage）。
- UPCR：由于原文中仅提供了中位数和四分位数范围，未提供均值和标准差。
- RAASI：Renin-Angiotensin-Aldosterone System Inhibitor（肾素 - 血管紧张素 - 醛固酮系统抑制剂）。
- ESA：Erythropoiesis-Stimulating Agent（促红细胞生成素刺激剂）。


## 五、重要变量和数据(中文展示)
### 文献主要变量数据统计信息

#### 连续变量（连续型变量）
| 变量 | 均值 ± 标准差 | 中位数（四分位距） |
| ---- | ---- | ---- |
| 年龄（岁） | 61.2 ± 16.3 | 61.2 (16.3) |
| 估算肾小球滤过率[毫升/分钟/1.73平方米] | 73.3 ± 31.0 | 73.3 (31.0) |
| 白蛋白（克/分升） | 4.2 ± 0.5 | 4.2 (0.5) |
| 钠（毫摩尔/升） | 140.3 ± 2.6 | 140.3 (2.6) |
| 钾（毫摩尔/升） | 4.2 ± 0.4 | 4.2 (0.4) |
| 钙（毫克/分升） | 9.2 ± 0.5 | 9.2 (0.5) |
| 磷（毫克/分升） | 3.4 ± 0.7 | 3.4 (0.7) |
| 低密度脂蛋白（毫克/分升） | 111.0 ± 32.1 | 111.0 (32.1) |
| 尿酸（毫克/分升） | 5.2 ± 1.5 | 5.2 (1.5) |
| 白细胞（10³/微升） | 6.3 ± 3.0 | 6.3 (3.0) |
| 血红蛋白（克/分升） | 13.6 ± 1.8 | 13.6 (1.8) |
| 尿蛋白肌酐比（克/克肌酐） | -0.27 [0.11, 1.00] | - |

#### 分类变量（分类型变量）
| 变量 | 频数（n） | 百分比（%） |
| ---- | ---- | ---- |
| 男性 | 13000 | 51.1 |
| 糖尿病（DM） | 5305 | 20.9 |
| 高血压 | 5234 | 20.6 |
| 心血管疾病（CVD） | 159 | 0.6 |
| 肾素 - 血管紧张素 - 醛固酮系统抑制剂使用情况（%） | 4407 | 17.3 |
| 磷结合剂使用情况（%） | 209 | 0.8 |
| 维生素D使用情况（%） | 653 | 2.6 |
| 他汀类药物使用情况（%） | 4162 | 16.4 |
| 降尿酸药物使用情况（%） | 2272 | 8.9 |
| 促红细胞生成素刺激剂使用情况（%） | 665 | 2.6 |

#### 说明
- 连续变量：提供了均值（Mean）、标准差（SD）、中位数（Median）和四分位距（IQR）。
- 分类变量：提供了频数（Frequency）和百分比（Percentage）。
- 尿蛋白肌酐比：由于原文仅提供了中位数和四分位距，未提供均值和标准差。
- RAASI：肾素 - 血管紧张素 - 醛固酮系统抑制剂（Renin - Angiotensin - Aldosterone System Inhibitor）
- ESA：促红细胞生成素刺激剂（Erythropoiesis - Stimulating Agent）



## 六、模拟数据
### Python代码：模拟慢性肾病数据并保存为CSV格式

#### 代码文件信息
- 代码名称：simulate_ckd_data.py

```python
import pandas as pd
import numpy as np
import os

# 设置保存路径
save_path = r"04文献阅读\05肾病\09基于机器学习的慢性肾脏疾病进展和死亡率预测网络系统\01data"
os.makedirs(save_path, exist_ok=True)  # 如果路径不存在，则创建

# 模拟数据的函数
def simulate_continuous_data(mean, std, median, iqr, size):
    """模拟连续变量数据"""
    data = np.random.normal(mean, std, size)
    q1, q3 = median - (iqr / 2), median + (iqr / 2)
    data = np.clip(data, q1, q3)  # 限制数据在IQR范围内
    return data

def simulate_categorical_data(freq, total_size):
    """模拟分类变量数据"""
    data = np.random.binomial(1, freq / 100, total_size)
    return data

# 模拟数据的大小
total_size = 1000  # 模拟1000条数据

# 连续变量
continuous_data = {
    "Age": simulate_continuous_data(mean=61.2, std=16.3, median=61.2, iqr=16.3, size=total_size),
    "eGFR": simulate_continuous_data(mean=73.3, std=31.0, median=73.3, iqr=31.0, size=total_size),
    "Albumin": simulate_continuous_data(mean=4.2, std=0.5, median=4.2, iqr=0.5, size=total_size),
    "Sodium": simulate_continuous_data(mean=140.3, std=2.6, median=140.3, iqr=2.6, size=total_size),
    "Potassium": simulate_continuous_data(mean=4.2, std=0.4, median=4.2, iqr=0.4, size=total_size),
    "Calcium": simulate_continuous_data(mean=9.2, std=0.5, median=9.2, iqr=0.5, size=total_size),
    "Phosphorus": simulate_continuous_data(mean=3.4, std=0.7, median=3.4, iqr=0.7, size=total_size),
    "LDL": simulate_continuous_data(mean=111.0, std=32.1, median=111.0, iqr=32.1, size=total_size),
    "Uric_acid": simulate_continuous_data(mean=5.2, std=1.5, median=5.2, iqr=1.5, size=total_size),
    "WBC": simulate_continuous_data(mean=6.3, std=3.0, median=6.3, iqr=3.0, size=total_size),
    "Hemoglobin": simulate_continuous_data(mean=13.6, std=1.8, median=13.6, iqr=1.8, size=total_size),
    "UPCR": np.random.uniform(low=0.11, high=1.00, size=total_size)  # 假设均匀分布
}

# 分类变量
categorical_data = {
    "Male": simulate_categorical_data(freq=51.1, total_size=total_size),
    "Diabetes_Mellitus": simulate_categorical_data(freq=20.9, total_size=total_size),
    "Hypertension": simulate_categorical_data(freq=20.6, total_size=total_size),
    "Cardiovascular_Disease": simulate_categorical_data(freq=0.6, total_size=total_size),
    "RAASI_Use": simulate_categorical_data(freq=17.3, total_size=total_size),
    "Phosphorus_Absorbent_Use": simulate_categorical_data(freq=0.8, total_size=total_size),
    "Vitamin_D_Use": simulate_categorical_data(freq=2.6, total_size=total_size),
    "Statin_Use": simulate_categorical_data(freq=16.4, total_size=total_size),
    "Uric_acid_lowering_Medicines_Use": simulate_categorical_data(freq=8.9, total_size=total_size),
    "ESA_Use": simulate_categorical_data(freq=2.6, total_size=total_size)
}

# 合并数据
data = {**continuous_data, **categorical_data}
df = pd.DataFrame(data)

# 保存为CSV文件
file_path = os.path.join(save_path, "simulated_ckd_data.csv")
df.to_csv(file_path, index=False, encoding="utf-8")

print(f"模拟数据已保存到路径：{file_path}")
```

### 代码说明
#### 数据模拟
- 连续变量使用正态分布模拟，并通过 `np.clip` 限制在四分位数范围内。
- 分类变量使用二项分布模拟，根据文献中的频率生成0和1。
- 特殊变量（如UPCR）假设为均匀分布。

#### 保存路径
- 数据将保存到指定路径 `04文献阅读\05肾病\09基于机器学习的慢性肾脏疾病进展和死亡率预测网络系统\01data`。
- 如果路径不存在，代码会自动创建。

#### 保存格式
- 数据保存为CSV格式，文件名为 `simulated_ckd_data.csv`。

### 运行代码
将上述代码保存为 `simulate_ckd_data.py` 文件，并在Python环境中运行。运行后，模拟数据将保存到指定路径。


## 七、复现代码
### Python代码：复现慢性肾病机器学习模型开发和验证过程

#### 代码文件信息
- 代码名称：reproduce_ckd_model.py

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, coxsnell_c_statistic
from sklearn.preprocessing import StandardScaler
import os
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

# 设置数据路径
data_path = r"04文献阅读\05肾病\09基于机器学习的慢性肾脏疾病进展和死亡率预测网络系统\01data\simulated_ckd_data.csv"

# 加载数据
df = pd.read_csv(data_path)

# 定义特征和目标变量
features = [
    "Age", "eGFR", "Albumin", "Sodium", "Potassium", "Calcium", "Phosphorus", "LDL",
    "Uric_acid", "WBC", "Hemoglobin", "UPCR", "Male", "Diabetes_Mellitus",
    "Hypertension", "Cardiovascular_Disease", "RAASI_Use", "Phosphorus_Absorbent_Use",
    "Vitamin_D_Use", "Statin_Use", "Uric_acid_lowering_Medicines_Use", "ESA_Use"
]

# 假设目标变量为随机生成的二分类标签（0: 无事件，1: 事件发生）
np.random.seed(42)
df["Outcome"] = np.random.binomial(1, 0.1, size=len(df))  # 假设事件发生率为10%

# 数据划分
X = df[features]
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 随机森林模型
rf_model = RandomForestClassifier(random_state=42)

# 超参数优化
param_grid = {
    "n_estimators": [100, 500, 1000],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 最佳模型
best_rf_model = grid_search.best_estimator_

# 模型评估
y_pred_proba = best_rf_model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"模型的ROC AUC: {roc_auc:.4f}")

# 保存模型结果
results = pd.DataFrame({
    "True_Outcome": y_test,
    "Predicted_Probability": y_pred_proba
})

results_path = os.path.join(os.path.dirname(data_path), "model_results.csv")
results.to_csv(results_path, index=False, encoding="utf-8")
print(f"模型结果已保存到路径：{results_path}")
```

### 代码说明
#### 数据加载
- 从指定路径加载之前生成的模拟数据。
- 定义特征变量和目标变量（假设目标变量为随机生成的二分类标签，表示事件是否发生）。

#### 数据预处理
- 使用 `train_test_split` 将数据划分为训练集和测试集。
- 对特征变量进行标准化处理。

#### 模型开发
- 使用随机森林分类器（`RandomForestClassifier`）。
- 通过网格搜索（`GridSearchCV`）优化超参数。

#### 模型评估
- 使用ROC AUC评估模型性能。
- 保存模型预测结果到CSV文件中。

#### 结果保存
- 将模型的预测结果保存到指定路径。

### 运行代码
- 确保之前生成的模拟数据文件 `simulated_ckd_data.csv` 存在于指定路径。
- 将上述代码保存为 `reproduce_ckd_model.py` 文件。
- 在Python环境中运行代码，模型结果将保存到指定路径。

### 注意事项
- 文献中使用了多种机器学习算法（如GB和XGBoost），这里仅以随机森林为例。如果需要复现其他算法，可以类似地实现。
- 文献中提到的C统计量（C - statistic）与ROC AUC类似，因此这里使用ROC AUC作为评估指标。
- 如果需要更复杂的验证（如亚组分析或生存分析），可以在代码中进一步扩展。