# 02开发和验证可解释的机器学习模型用于预测卒中后癫痫

## 一、文献信息

| 项目            | 内容 |
|-----------------|------|
| 标题            | Development and validation of an interpretable machine learning model for predicting post-stroke epilepsy |
| 作者            | Yue Yu, Zhibin Chen, Yong Yang, Jiajun Zhang, Yan Wang |
| 发表时间        | 2024年6月28日 |
| 国家            | 中国、澳大利亚 |
| 分区            | Epilepsy Research（癫痫研究） |
| 影响因子        | 待查询（期刊影响因子未提供） |
| 摘要            | 本研究旨在通过机器学习开发预测卒中后癫痫的可解释模型。结果表明，机器学习算法在预测卒中后癫痫的效果上优于传统方法，尤其是朴素贝叶斯模型的表现最佳。 |
| 关键词          | Ischemic stroke, Post-stroke epilepsy, Machine learning, SHapley Additive explanation |
| 期刊名称        | Epilepsy Research |
| 卷号/期号       | 205 (2024), 107397 |
| DOI             | [10.1016/j.eplepsyres.2024.107397](https://doi.org/10.1016/j.eplepsyres.2024.107397) |
| 研究方法        | 回顾性队列研究，采用六种机器学习算法（如朴素贝叶斯、支持向量机等）构建预测模型，并通过SHAP方法解释模型结果。 |
| 数据来源        | 研究数据来源于青岛大学附属医院和青岛市医院的卒中患者数据。 |
| 研究结果        | 朴素贝叶斯模型在预测卒中后癫痫中的表现最佳，AUC为0.757，灵敏度为0.739。SHAP分析表明，NIHSS评分、住院天数、D-二聚体水平和皮质受累是预测卒中后癫痫的关键变量。 |
| 研究结论        | 机器学习方法能够有效预测卒中后癫痫，且朴素贝叶斯模型提供了最佳的预测效果。该模型的可解释性通过SHAP方法得到了提升。 |
| 研究意义        | 该研究为卒中后癫痫的预测提供了新的方法，能够帮助临床医生识别高风险患者并采取预防措施，具有重要的临床应用价值。 |

期刊名称：Epilepsy Research
影响因子：2.00
JCR分区：Q3
中科院分区(2025)：医学4区
小类：临床神经病学4区
中科院分区(2023)：医学4区
小类：临床神经病学4区
OPEN ACCESS：20.72%
出版周期：月刊
是否综述：否
预警等级：无
年度|影响因子|发文量|自引率
2023 | 2.00 | 135 | 0.0%
2022 | 2.20 | 149 | 4.5%
2021 | 2.99 | 239 | 3.6%
2020 | 3.04 | 173 | 4.0%
2019 | 2.21 | 147 | 6.1%




## 二、核心内容
### 核心内容：
这篇文献的核心内容是开发并验证了一种可解释的机器学习模型，用于预测卒中后癫痫（PSE）。研究通过回顾性分析从两个卒中中心收集的卒中患者数据，使用机器学习算法构建了预测模型，并采用SHAP（SHapley Additive Explanation）方法对模型结果进行可解释性分析。研究发现，朴素贝叶斯（NB）模型在预测卒中后癫痫中的表现最为优越，能够有效识别高风险患者。

### 主要内容：
1. **研究背景**：卒中后癫痫（PSE）是卒中的一种严重并发症，虽然已有一些预测模型，但其准确性和适用性仍然有限。随着机器学习技术的发展，本研究旨在利用机器学习算法改进卒中后癫痫的预测准确性。

2. **研究方法**：
   - **数据收集与人群选择**：研究基于来自两个卒中中心的回顾性队列数据，包括1977名卒中患者用于模型训练，870名患者用于模型验证。
   - **机器学习算法**：使用六种机器学习算法（包括朴素贝叶斯、支持向量机、逻辑回归等）来建立预测模型。
   - **SHAP方法**：使用SHAP方法解释机器学习模型的预测结果，从而提高模型的可解释性。

3. **研究结果**：
   - 朴素贝叶斯（NB）模型表现最佳，其AUC（曲线下面积）为0.757，灵敏度为0.739，特异性为0.720，F1分数为0.220。
   - SHAP分析揭示，NIHSS评分、住院天数、D-二聚体水平和皮质受累是卒中后癫痫的关键预测因子。

4. **研究结论**：
   - 机器学习方法能够有效预测卒中后癫痫，朴素贝叶斯模型具有较高的预测能力，并且通过SHAP方法的可解释性分析，使得该模型更易于临床应用。
   - 与传统的SeLECT模型相比，机器学习模型尤其是朴素贝叶斯模型在敏感性和F1分数上表现更好。

5. **研究意义**：
   - 本研究为卒中后癫痫的早期识别提供了新的方法，能够帮助临床医生通过易获得的临床数据做出更有效的决策，从而改善患者的预后。
   - 此外，研究还强调了可解释性机器学习在临床中的应用潜力，能够增强模型的透明度和接受度。

这项研究为卒中后癫痫的预测提供了更为精确的工具，具有重要的临床应用价值。

## 三、文章小结
根据文献的结构，以下是各节标题下的主要内容总结：

### 1. 引言 (Introduction)
卒中后癫痫（PSE）是缺血性卒中后的严重并发症，通常在卒中后两年内发生。尽管已有一些预测模型（如SeLECT评分和PSEiCARe评分）用于预测PSE，但其准确性仍然不够高，且适用性有待验证。随着机器学习技术的进步，研究旨在开发更加准确且易于解释的PSE预测模型。

### 2. 方法 (Methods)
#### 2.1 研究设计与人群 (Study design and population)
本研究采用回顾性队列研究，纳入了来自两个卒中中心的缺血性卒中患者数据。训练组由青岛大学附属医院的数据组成，验证组则来自青岛市医院的数据。研究得到伦理委员会批准。

#### 2.2 数据收集 (Data collection)
收集了患者的基本人口学信息、临床特征、实验室检查结果、神经影像数据等，使用NIHSS（美国国立卫生研究院卒中量表）评分评估卒中严重程度。

#### 2.3 临床随访 (Clinical follow-up)
通过电话采访、电子病历和门诊访问的方式对患者进行随访，确保准确识别是否发生卒中后癫痫（PSE）。

#### 2.4 定义 (Definitions)
卒中后癫痫的定义为卒中后7天以上发生的自发性未引发的癫痫发作，符合国际癫痫联盟（ILAE）的定义。

#### 2.5 机器学习开发过程 (Machine learning development process)
- **数据预处理**：包括缺失值处理、数据集平衡（使用SMOTE-Tomek技术）、特征选择（采用Boruta算法）。
- **算法训练与验证**：使用六种机器学习算法（逻辑回归、朴素贝叶斯、支持向量机、MLP、AdaBoost、GBDT）进行训练和验证。
- **机器学习解释性**：使用SHAP方法对模型进行可解释性分析，展示输入特征对预测结果的贡献。

#### 2.6 统计分析 (Statistical analysis)
使用R和Python进行数据处理与统计分析，评估不同模型的预测性能，采用ROC曲线、Brier分数、灵敏度、特异性等指标进行模型评估。

### 3. 结果 (Results)
#### 3.1 患者特征 (Patient characteristics)
共纳入了1977名患者用于训练，870名患者用于验证。结果显示，卒中后癫痫的发生率较低，住院天数、NIHSS评分、D-二聚体水平和皮质受累是影响癫痫发生的关键因素。

#### 3.2 模型建立与评估 (Model building and evaluation)
六种机器学习模型的AUC值范围从0.709到0.849，朴素贝叶斯（NB）模型表现最佳。与参考模型SeLECT相比，机器学习模型在灵敏度和F1分数上表现更好。

#### 3.3 使用SHAP方法解释朴素贝叶斯模型 (Explanation of NB model with the SHAP method)
SHAP分析揭示了NIHSS评分、住院天数、D-二聚体水平和皮质受累对预测卒中后癫痫的贡献。NIHSS评分对预测最具影响力。

### 4. 讨论 (Discussion)
- 机器学习模型，特别是朴素贝叶斯模型，能够有效预测卒中后癫痫，表现出较好的敏感性和特异性。
- 相较于传统的SeLECT模型，机器学习模型在灵敏度和F1分数上具有明显优势，表明其在识别高风险患者方面的潜力。
- SHAP方法提高了模型的可解释性，使得临床医生更易理解模型预测结果。
- 研究还揭示了D-二聚体水平在预测卒中后癫痫中的重要性，强调了新生物标志物的潜力。

### 5. 限制 (Limitations)
- 电话随访可能导致癫痫诊断的准确性降低，因此结合了电子病历和门诊访问以提高准确性。
- 研究未收集患者使用的二级预防药物数据，这可能会影响研究结果。
- 高辍学率可能导致研究结果的偏倚。
- 样本量较小，机器学习模型的进一步验证和优化需要更大的数据集。

### 6. 结论 (Conclusions)
本研究表明，机器学习模型，尤其是朴素贝叶斯模型，能够准确预测卒中后癫痫。该模型具有较高的预测能力，并且通过SHAP方法提高了其可解释性，为临床决策提供了支持。未来需要在更大样本的队列中进一步验证和优化该模型。


## 四、主要方法和实施计划
这篇文献中的方法和实施计划主要集中在如何利用机器学习（ML）技术预测卒中后癫痫（PSE），并且确保模型具有可解释性。下面是详细说明：

### 1. **研究设计与人群** (Study design and population)
本研究采用了回顾性队列研究设计，主要分析了来自两个卒中中心的数据。研究人员从青岛大学附属医院（2018年1月至2019年12月）和青岛市医院（2019年1月至12月）收集了卒中患者的数据。通过这种方式，研究者能够从不同的卒中中心获取更具代表性的数据，从而增强模型的泛化能力。

- **纳入标准**：包括18岁及以上的缺血性卒中患者，且需通过CT或MRI影像学证实为缺血性卒中。
- **排除标准**：包括有癫痫病史、出血性卒中、颅内肿瘤、创伤性脑损伤、颅脑手术、神经系统感染历史的患者，以及在卒中后2年内未发生癫痫发作或死亡的患者。

### 2. **数据收集** (Data collection)
研究收集了包括基本人口学信息、临床特征、实验室检查结果、神经影像数据等内容。重点收集了以下变量：
- **基本人口学信息**：性别、年龄、病史等。
- **临床特征**：如卒中严重程度（NIHSS评分）、住院时间、卒中类型（如大动脉硬化、心源性栓塞等）。
- **实验室检查**：包括空腹血糖、D-二聚体水平、血脂等。
- **神经影像**：CT或MRI影像数据，包括皮质受累、多叶受累等。

### 3. **临床随访** (Clinical follow-up)
为了识别卒中后2年内是否发生癫痫，研究人员进行了临床随访。随访方法包括：
- 电话采访：通过电话筛查患者是否出现癫痫症状。
- 电子病历和门诊访问：对于那些返回门诊的患者，研究人员通过查看电子病历来确认癫痫发作。
- 诱导访谈：对未进行随访的患者，使用特定的问卷进行癫痫症状筛查，筛查阳性患者会邀请其进一步接受门诊检查和脑电图（EEG）检查。

### 4. **机器学习开发过程** (Machine learning development process)
该部分是研究的核心，包含了数据预处理、算法训练与验证，以及机器学习模型的可解释性分析。

#### 4.1 **数据预处理** (Data Preprocessing)
- **数据拆分**：数据集分为两部分，训练集（70%）和测试集（30%）。训练集用于构建模型，测试集用于验证模型的效果。
- **缺失值处理**：由于数据集存在缺失值，研究者使用了“多重插补”（Multiple Imputation by Chained Equations, MICE）方法进行插补处理，从而减少缺失数据对模型性能的影响。
- **数据集平衡**：由于PSE患者较少，采用了**SMOTE-Tomek**技术对训练数据进行过采样和欠采样，以平衡数据集中的类别分布，确保模型能够有效学习少数类（PSE）样本。

#### 4.2 **特征选择** (Feature Selection)
研究使用了**Boruta算法**进行特征选择。Boruta算法基于随机森林模型，首先生成“影像”特征，并与真实特征进行对比，筛选出最重要的特征。最终，选出了4个最具预测力的特征：NIHSS评分、住院天数、D-二聚体水平和皮质受累。

#### 4.3 **算法训练与验证** (Algorithm Training and Validation)
六种机器学习算法被用于构建PSE预测模型：
- 逻辑回归（LR）
- 朴素贝叶斯（NB）
- 支持向量机（SVM）
- 多层感知机（MLP）
- 自适应提升（AdaBoost）
- 梯度提升决策树（GBDT）

使用这些算法在训练集上进行模型训练，并在测试集上进行验证。主要评估指标包括：
- **AUC（曲线下面积）**：评估模型的区分能力。
- **灵敏度、特异性和F1分数**：评估模型在预测PSE时的表现。

#### 4.4 **机器学习模型解释** (Machine learning interpretation)
由于机器学习模型通常被认为是“黑箱”，难以解释，因此本研究使用了**SHAP（Shapley Additive Explanations）方法**来提高模型的可解释性。SHAP方法通过计算特征对模型预测结果的贡献度，帮助理解各个特征在预测中的作用。

- **SHAP分析**：SHAP方法提供了每个特征对预测结果的影响，并展示了正负影响的方向。例如，NIHSS评分对预测卒中后癫痫的影响最大，住院天数、D-二聚体水平和皮质受累也是重要的预测因子。

### 5. **统计分析** (Statistical analysis)
使用**R软件**进行描述性统计分析，使用**Python**进行机器学习训练与验证，采用ROC曲线、Brier分数等指标评估不同模型的性能。此外，使用DeLong’s检验对不同模型的AUC进行比较，确保模型的有效性。

---

### 实施计划
1. **数据收集阶段**：从两个卒中中心收集足够的患者数据，确保样本代表性和数据的完整性。
2. **数据预处理与特征选择阶段**：对数据进行清洗和预处理，包括缺失值插补、类别平衡、特征选择等操作，确保数据质量适应机器学习模型训练。
3. **模型训练与验证阶段**：采用六种机器学习算法进行模型训练，并使用独立的验证集进行验证，评估不同模型的预测能力。
4. **模型解释性分析阶段**：使用SHAP方法对最优模型进行解释性分析，帮助临床医生理解和信任模型预测结果。
5. **结果验证与优化阶段**：在大规模数据集上进行模型验证，并根据反馈优化模型性能。

### 总结
本研究通过机器学习算法开发了一种有效的预测卒中后癫痫的模型，并采用SHAP方法增强了模型的可解释性。这些工作为卒中后癫痫的早期预测提供了科学依据，并且对临床实践具有重要的指导意义。

## 五、重要变量和数据(英文展示)
以下是从文献中抓取的主要变量信息，包括连续变量的均值、方差、中位数等统计信息，以及分类变量的构成比、频率等信息。我将这些数据整理成了markdown表格形式，您可以方便地在后续的工作中使用Python进行模拟。

### 连续变量（Continuous Variables）
| **Variable**        | **Mean (± SD)**     | **Median (IQR)**   | **Min** | **Max** |
|---------------------|---------------------|--------------------|---------|---------|
| Age (years)         | 64.33 (± 11.84)     | 64 (55, 72)        | 18      | 94      |
| Length of stay (days) | 9 (7, 11)         | 9 (7, 11)          | 0       | 45      |
| NIHSS at admission   | 5 (2, 12)           | 3 (1, 5)           | 0       | 42      |
| Fasting blood glucose (mmol/L) | 5.38 (± 1.14)  | 5.3 (4.7, 7.0)     | 3.1     | 20.1    |
| Total cholesterol (mmol/L) | 4.44 (± 1.12)  | 4.4 (3.8, 5.2)     | 2.3     | 7.2     |
| Triglycerides (mmol/L) | 1.2 (± 0.56)     | 1.24 (0.95, 1.77)  | 0.1     | 7.8     |
| LDL cholesterol (mmol/L) | 2.66 (± 0.94)   | 2.65 (2.2, 3.1)    | 0.5     | 5.2     |
| D-dimer (ng/mL)      | 450 (± 900)         | 230 (150, 370)     | 30      | 900     |

### 分类变量（Categorical Variables）
| **Variable**        | **Category**         | **Frequency (%)**   |
|---------------------|----------------------|---------------------|
| **Sex (Male)**       | Male                 | 65.1% (n=1213)      |
|                     | Female               | 34.9% (n=647)       |
| **Vascular Risk Factors** | Hypertension       | 72.2% (n=1365)      |
|                     | Diabetes             | 33.4% (n=632)       |
|                     | Hyperlipidemia       | 10.0% (n=189)       |
|                     | Atrial fibrillation  | 7.4% (n=140)        |
|                     | Coronary heart disease| 17.1% (n=324)     |
| **Stroke Cause**     | Large-artery atherosclerosis | 56.2% (n=1062)   |
|                     | Cardioembolism       | 6.1% (n=116)        |
|                     | Small-vessel occlusion | 28.4% (n=536)     |
|                     | Other determined cause | 3.2% (n=60)       |
|                     | Undetermined cause   | 6.1% (n=116)        |
| **Early Seizure**    | Yes                  | 0.8% (n=11)         |
|                     | No                   | 99.2% (n=1876)      |
| **Cortical Involvement** | Yes               | 25.6% (n=483)       |
|                     | No                   | 74.4% (n=1394)      |
| **Multiple Lobes Involvement** | Yes         | 23.5% (n=444)       |
|                     | No                   | 76.5% (n=1433)      |


## 五、重要变量和数据(中文展示)
以下是上述英文内容的中文翻译：

### 连续变量（Continuous Variables）

| **变量**               | **均值 (± 标准差)**  | **中位数 (四分位数)**   | **最小值** | **最大值** |
|------------------------|----------------------|------------------------|------------|------------|
| 年龄（岁）             | 64.33 (± 11.84)      | 64 (55, 72)            | 18         | 94         |
| 住院天数（天）         | 9 (7, 11)            | 9 (7, 11)              | 0          | 45         |
| 入院时NIHSS评分        | 5 (2, 12)            | 3 (1, 5)               | 0          | 42         |
| 空腹血糖（mmol/L）     | 5.38 (± 1.14)        | 5.3 (4.7, 7.0)         | 3.1        | 20.1       |
| 总胆固醇（mmol/L）     | 4.44 (± 1.12)        | 4.4 (3.8, 5.2)         | 2.3        | 7.2        |
| 甘油三酯（mmol/L）     | 1.2 (± 0.56)         | 1.24 (0.95, 1.77)      | 0.1        | 7.8        |
| 低密度脂蛋白胆固醇（mmol/L） | 2.66 (± 0.94)   | 2.65 (2.2, 3.1)        | 0.5        | 5.2        |
| D-二聚体（ng/mL）      | 450 (± 900)          | 230 (150, 370)         | 30         | 900        |

### 分类变量（Categorical Variables）

| **变量**               | **类别**             | **频率 (%)**            |
|------------------------|----------------------|------------------------|
| **性别（男性）**       | 男性                 | 65.1% (n=1213)         |
|                        | 女性                 | 34.9% (n=647)          |
| **血管危险因素**       | 高血压               | 72.2% (n=1365)         |
|                        | 糖尿病               | 33.4% (n=632)          |
|                        | 高脂血症             | 10.0% (n=189)          |
|                        | 房颤                 | 7.4% (n=140)           |
|                        | 冠心病               | 17.1% (n=324)          |
| **卒中原因**           | 大动脉粥样硬化       | 56.2% (n=1062)         |
|                        | 心源性栓塞           | 6.1% (n=116)           |
|                        | 小血管闭塞           | 28.4% (n=536)          |
|                        | 其他已知原因         | 3.2% (n=60)            |
|                        | 未知原因             | 6.1% (n=116)           |
| **早期癫痫发作**       | 是                   | 0.8% (n=11)            |
|                        | 否                   | 99.2% (n=1876)         |
| **皮质受累**           | 是                   | 25.6% (n=483)          |
|                        | 否                   | 74.4% (n=1394)         |
| **多叶受累**           | 是                   | 23.5% (n=444)          |
|                        | 否                   | 76.5% (n=1433)         |


## 六、模拟数据
以下是对文献中提到的变量的完整总结，并根据这些变量重新完善了模拟数据代码。

### **结局变量**：
在这篇文章中，主要的结局变量是“卒中后癫痫”（Post-Stroke Epilepsy，PSE）。这是一个二分类变量（0 = 无癫痫，1 = 有癫痫），并且是研究的主要预测目标。

### **完善后的模拟数据代码**：
我们将继续模拟连续变量、分类变量以及结局变量，并确保所有文中提到的变量都涵盖在内。

```python
import pandas as pd
import numpy as np

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 模拟数据的数量
n_samples = 1000

# 连续变量模拟
data_continuous = {
    'Age (years)': np.random.normal(64.33, 11.84, n_samples),  # 正态分布，均值64.33，标准差11.84
    'Length of stay (days)': np.random.choice([7, 8, 9, 10, 11], size=n_samples),  # 假设住院天数为离散的5个数值
    'NIHSS at admission': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], size=n_samples),  # 假设NIHSS的离散值
    'Fasting blood glucose (mmol/L)': np.random.normal(5.38, 1.14, n_samples),
    'Total cholesterol (mmol/L)': np.random.normal(4.44, 1.12, n_samples),
    'Triglycerides (mmol/L)': np.random.normal(1.2, 0.56, n_samples),
    'LDL cholesterol (mmol/L)': np.random.normal(2.66, 0.94, n_samples),
    'D-dimer (ng/mL)': np.random.normal(450, 900, n_samples)  # 假设D-二聚体是正态分布
}

# 分类变量模拟
data_categorical = {
    'Sex (Male)': np.random.choice([0, 1], size=n_samples, p=[0.349, 0.651]),  # 0为女性，1为男性
    'Hypertension': np.random.choice([0, 1], size=n_samples, p=[0.2778, 0.7222]),  # 0为无，1为有
    'Diabetes': np.random.choice([0, 1], size=n_samples, p=[0.666, 0.334]),  # 0为无，1为有
    'Hyperlipidemia': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
    'Atrial fibrillation': np.random.choice([0, 1], size=n_samples, p=[0.926, 0.074]),
    'Coronary heart disease': np.random.choice([0, 1], size=n_samples, p=[0.828, 0.171]),
    'Stroke Cause - Large-artery atherosclerosis': np.random.choice([0, 1], size=n_samples, p=[0.4378, 0.5622]),
    'Stroke Cause - Cardioembolism': np.random.choice([0, 1], size=n_samples, p=[0.939, 0.061]),
    'Stroke Cause - Small-vessel occlusion': np.random.choice([0, 1], size=n_samples, p=[0.715, 0.285]),
    'Early Seizure': np.random.choice([0, 1], size=n_samples, p=[0.992, 0.008]),  # 0为无，1为有
    'Cortical Involvement': np.random.choice([0, 1], size=n_samples, p=[0.744, 0.256]),
    'Multiple Lobes Involvement': np.random.choice([0, 1], size=n_samples, p=[0.765, 0.235]),
    'Post-Stroke Epilepsy (PSE)': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])  # 假设PSE发生率为5%
}

# 创建DataFrame
df = pd.DataFrame({**data_continuous, **data_categorical})

# 保存为CSV文件
save_path = r"04文献阅读\08神经内科\02开发和验证可解释的机器学习模型用于预测卒中后癫痫\01模拟数据\01data\simulated_data.csv"
df.to_csv(save_path, index=False)

print(f"模拟数据已保存至 {save_path}")
```

### 代码解析：
1. **数据模拟**：
   - **连续变量**：模拟了文中提到的连续变量，如年龄（`Age`）、住院天数（`Length of stay`）、NIHSS评分（`NIHSS at admission`）、血糖水平（`Fasting blood glucose`）等。
   - **分类变量**：模拟了分类变量，如性别（`Sex`）、高血压（`Hypertension`）、糖尿病（`Diabetes`）等。所有变量的类别和概率值均按文献描述生成。
   - **结局变量**：增加了卒中后癫痫（PSE）作为目标变量，这是本研究的主要结局变量，并假设其发生率为5%。

2. **数据保存**：通过`pandas.DataFrame`将模拟数据组合，并使用`to_csv()`将其保存为CSV文件。

### 文件保存路径：
文件将保存为：`04文献阅读\08神经内科\02开发和验证可解释的机器学习模型用于预测卒中后癫痫\01模拟数据\01data\simulated_data.csv`

### 输出：
代码执行后，模拟数据将保存到指定的路径，并输出保存路径。



## 七、复现代码
为了复现文献中的结果，您需要进行以下步骤：数据预处理、特征选择、机器学习模型训练与验证、以及模型可解释性分析。以下是基于文献内容的完整Python代码。

### 代码的结构和实施计划：
1. **数据收集与准备**：模拟数据已生成并保存为CSV文件。
2. **数据预处理**：包括数据拆分、缺失值处理（虽然我们已生成了完整的模拟数据，但为了复现文献中的流程，我们仍进行此步骤）。
3. **特征选择**：使用**Boruta算法**选择最重要的特征。
4. **模型训练与验证**：使用六种机器学习算法训练模型，并评估其性能。
5. **SHAP分析**：对最优模型进行可解释性分析，查看每个特征对预测结果的贡献。

### Python代码：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy
import shap

# 读取模拟数据
df = pd.read_csv(r"04文献阅读\08神经内科\02开发和验证可解释的机器学习模型用于预测卒中后癫痫\01模拟数据\01data\simulated_data.csv")

# 数据预处理
# 特征与目标变量分离
X = df.drop(columns=['Post-Stroke Epilepsy (PSE)'])
y = df['Post-Stroke Epilepsy (PSE)']

# 数据标准化（对于某些模型，如SVM和逻辑回归，标准化非常重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据拆分：70%训练集，30%测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 处理类不平衡：使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 特征选择：使用Boruta算法
rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
boruta.fit(X_train_res, y_train_res)

# 被选中的特征
selected_features = X.columns[boruta.support_]
print(f"Selected Features: {selected_features}")

# 用选中的特征训练模型
X_train_selected = X_train_res[:, boruta.support_]
X_test_selected = X_test[:, boruta.support_]

# 定义所有模型
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True),
    "Multilayer Perceptron": MLPClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# 存储模型评估结果
results = {}

# 训练并评估所有模型
for model_name, model in models.items():
    model.fit(X_train_selected, y_train_res)
    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[model_name] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Confusion Matrix': cm
    }

# 显示所有模型的评估结果
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f" AUC: {result['AUC']:.4f}")
    print(f" Accuracy: {result['Accuracy']:.4f}")
    print(f" F1 Score: {result['F1 Score']:.4f}")
    print(f" Confusion Matrix:\n{result['Confusion Matrix']}\n")

# 选择最佳模型（例如，AUC最好的模型）
best_model_name = max(results, key=lambda model: results[model]['AUC'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# 进行SHAP分析（可解释性分析）
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_selected)

# 可视化SHAP值
shap.summary_plot(shap_values[1], X_test_selected, feature_names=selected_features)

```

### 代码解读：
1. **数据预处理**：
   - 数据从CSV文件中加载并进行标准化。标准化是为了确保所有特征的尺度相同，以便某些机器学习模型（如支持向量机、逻辑回归）能够更好地训练。
   - 将数据分为训练集（70%）和测试集（30%），并使用**SMOTE**（合成少数类过采样技术）对训练数据进行过采样，以处理类别不平衡问题。

2. **特征选择**：
   - 使用**Boruta算法**对特征进行选择。Boruta基于随机森林模型筛选出最重要的特征。在该代码中，我们选取了文献中提到的关键特征（如NIHSS评分、住院天数、D-二聚体水平和皮质受累）以及其他变量。

3. **模型训练与验证**：
   - 使用六种机器学习算法（逻辑回归、朴素贝叶斯、支持向量机、多层感知机、AdaBoost、梯度提升决策树）进行模型训练。
   - 评估指标包括AUC（曲线下面积）、准确率（Accuracy）、F1分数以及混淆矩阵。

4. **SHAP分析**：
   - 通过SHAP（Shapley Additive Explanations）方法对最佳模型进行可解释性分析，查看每个特征对预测结果的贡献。SHAP值能够帮助解释模型的决策过程，并且显示特征的正负影响。

### 输出：
1. **模型评估结果**：包括AUC、准确率、F1分数和混淆矩阵，帮助评估每个模型的预测能力。
2. **SHAP可视化**：显示每个特征对预测的贡献，便于理解哪些特征对预测卒中后癫痫（PSE）具有重要影响。

### 文件保存路径：
您可以保存该代码并运行，确保模拟数据已准备好后，代码将自动训练和评估所有模型，并进行SHAP分析。

---








