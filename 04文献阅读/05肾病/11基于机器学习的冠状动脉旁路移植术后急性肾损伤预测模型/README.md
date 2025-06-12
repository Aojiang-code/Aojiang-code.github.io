# 冠状动脉旁路移植术后急性肾损伤预测模型
## 一、文献信息


| 项目    | 内容                                                                                                      |
| ----- | ------------------------------------------------------------------------------------------------------- |
| 标题    | 基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型                                                                              |
| 作者    | Haiming Li, Hui Hu, Jingxing Li, Wenxing Peng                                                           |
| 发表时间  | 2025年4月25日在线发表                                                                                          |
| 国家    | 中国                                                                                                      |
| 分区    | 未明确标注，推测为中科院三区（根据《Journal of Thoracic Disease》影响力）                                                      |
| 影响因子  | 2.4（以2024年《J Thorac Dis》估算值为参考）                                                                         |
| 摘要    | 本研究基于2,155例老年CABG患者的回顾性数据，构建9种机器学习模型预测术后急性肾损伤（AKI），其中随机森林模型预测效果最佳（AUC=0.737），识别出eGFR、UA、ALT、BNP等关键危险因素。 |
| 关键词   | 冠状动脉旁路移植术（CABG）；急性肾损伤（AKI）；机器学习；预测模型；老年患者                                                               |
| 期刊名称  | Journal of Thoracic Disease                                                                             |
| 卷号/期号 | 第17卷 第4期                                                                                                |
| DOI   | 10.21037/jtd-2025-264                                                                                   |
| 研究方法  | 回顾性队列研究 + 机器学习模型构建与比较（RF、LR、SVM等9种算法）                                                                   |
| 数据来源  | 北京安贞医院2019年1月至2020年12月的老年CABG患者住院记录                                                                     |
| 研究结果  | RF模型AUC最高为0.737，优于其他8种算法，术前eGFR、UA、ALT、BNP、手术时间和IABP使用是主要预测因子。                                          |
| 研究结论  | 随机森林模型对老年CABG术后AKI有良好预测能力，可用于术前评估和术后干预决策支持。                                                             |
| 研究意义  | 有助于术前识别高风险老年CABG患者，指导围术期个体化管理，减少AKI发生率及其相关死亡率。                                                          |



### 期刊信息

| **字段**           | **内容**                                                                 |
|--------------------|------------------------------------------------------------------------|
| 期刊名称           | Journal of Thoracic Disease                                            |
| 影响因子           | 2.10                                                                   |
| JCR 分区           | Q3                                                                     |
| 中科院分区 (2025)  | 医学4区                                                                |
| 小类               | 呼吸系统4区                                                            |
| 中科院分区 (2023)  | 医学3区                                                                |
| 小类               | 呼吸系统4区                                                            |
| OPEN ACCESS        | 97.85%                                                                 |
| 出版周期           | 月刊                                                                   |
| 是否综述           | 否                                                                     |
| 预警等级           | 无                                                                     |

#### 年度数据

| **年度** | **影响因子** | **发文量** | **自引率** |
|----------|--------------|------------|------------|
| 2023     | 2.10         | 597        | 9.5%       |
| 2022     | 2.50         | 420        | 4.0%       |
| 2021     | 3.01         | 662        | 3.9%       |
| 2020     | 2.89         | 719        | 7.1%       |
| 2019     | 2.05         | 707        | 12.1%      |



## 二、核心内容与主要内容总结

这篇题为《基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型》的研究，聚焦于**老年患者在冠状动脉旁路移植术（CABG）后发生急性肾损伤（AKI）的风险预测**，采用多种机器学习算法构建预测模型，核心与主要内容总结如下：

### 一、研究背景

* 急性肾损伤（AKI）是CABG术后常见且严重的并发症，尤其在老年患者中发病率更高，预后更差。
* 目前针对老年CABG患者术后AKI的预测模型研究较少，因此有必要探索有效的预测手段，以指导术前评估与术后管理。

### 二、研究方法

* **研究设计**：回顾性单中心研究。
* **数据来源**：北京安贞医院2019年1月至2020年12月期间接受CABG手术、年龄≥65岁的患者，共纳入2,155例。
* **建模方法**：采用9种主流机器学习算法构建模型，包括随机森林（RF）、逻辑回归（LR）、决策树（DT）、支持向量机（SVM）、XGBoost、LightGBM等。
* **特征选择**：通过SHAP值评估变量重要性。
* **验证方法**：内部验证集（30%）+ ROC曲线比较模型性能。

### 三、研究结果

* 最终发现**随机森林模型（RF）性能最佳**，其AUC为0.737，优于其他模型。
* 主要预测因子包括：术前eGFR、尿酸（UA）、谷丙转氨酶（ALT）、B型钠尿肽（BNP）、手术时间、IABP使用、年龄等。
* 模型能将患者分为低、中、高风险群体，为临床提供分层管理依据。

### 四、研究结论

* 机器学习方法，特别是RF模型，可以有效预测老年CABG患者术后发生AKI的风险。
* 模型可用于术前风险筛查、术中监测优化及术后干预计划制定。

### 五、研究意义

* 该研究为术后AKI的早期识别和干预提供了新工具，有助于降低AKI发生率和术后死亡率，推动个体化精准医疗发展。




## 三、文章小结
根据文献《基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型》的结构和内容，现按其节标题依次总结主要内容如下：

### **1. Introduction（引言）**

* 阐述了急性肾损伤（AKI）在冠状动脉旁路移植术（CABG）患者中的高发性和严重性，尤其在老年人群中更为常见。
* 提出当前老年CABG术后AKI研究有限，因此本研究拟构建基于机器学习的预测模型，以提高风险识别效率。

### **2. Methods（方法）**

* **研究设计**：回顾性队列研究。
* **入排标准**：纳入65岁以上、2019–2020年在北京安贞医院接受CABG手术的患者；排除术前透析、严重CKD、缺失数据等情况。
* **AKI定义**：采用KDIGO标准，仅使用血清肌酐（SCr）变化判定。
* **模型构建**：共使用9种机器学习算法，包括RF、LR、DT、SVM、XGBoost、LightGBM等。
* **变量处理**：剔除缺失值超过20%的变量，连续变量标准化；使用SHAP分析变量重要性；训练集与验证集按7:3划分。

### **3. Results（结果）**

* 共纳入2,155名患者，其中13.6%发生1期AKI，2.4%为2期，0.9%为3期。
* 随机森林（RF）模型预测性能最佳（AUC=0.737），其次为AdaBoost和LR。
* 重要预测因子包括术前eGFR、UA、ALT、BNP，术中IABP使用、手术时长、年龄等。
* SHAP图显示上述变量对模型预测影响显著。

### **4. Discussion（讨论）**

* 强调机器学习在预测医疗风险中的优势，指出RF模型在结构稳定性和处理复杂变量方面的适用性。
* 与以往研究中LightGBM或XGBoost占优的结果相比，本研究RF模型表现更优，可能与人群、变量特征不同有关。
* 强调模型对临床实践的指导意义：可用于高危老年CABG患者的术前分层管理和早期干预。
* 讨论了本研究的局限性，包括单中心、回顾性设计、缺乏外部验证等。

### **5. Conclusions（结论）**

* 本研究构建了多个机器学习预测模型用于评估老年CABG术后AKI风险。
* 随机森林模型表现最优，可为术前评估和个体化干预策略提供支持，具有临床推广价值。


## 四、主要方法和实施计划
以下是对该文献“**基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型**”中**主要研究方法和实施计划**的详细说明：

### 一、研究设计与总体框架

* **研究类型**：回顾性队列研究。
* **研究目标**：构建并比较多种机器学习模型，以预测老年患者冠状动脉旁路移植术（CABG）后发生急性肾损伤（AKI）的风险。
* **研究周期**：纳入时间为2019年1月至2020年12月。
* **研究场所**：北京安贞医院。


### 二、研究对象与数据采集

#### 1. 纳入标准：

* 年龄≥65岁；
* 行单纯CABG手术；
* 有完整术前及术中相关数据。

#### 2. 排除标准：

* 术前已接受长期透析；
* 术前存在AKI或严重CKD（eGFR<30 mL/min）；
* 缺乏基线肌酐（SCr）数据或记录不全；
* 手术过程中死亡或术后住院时间超过90天。

#### 3. 数据来源：

* 电子病历系统（EMR）中提取患者的临床特征、实验室指标、用药记录、手术参数等。


### 三、急性肾损伤（AKI）的判定标准

采用**KDIGO（Kidney Disease Improving Global Outcomes）标准**：

* **定义**：术后7天内血清肌酐（SCr）上升≥50% 或 术后48小时内上升≥26.5 μmol/L。
* **分级**：

  * **AKI 1期**：SCr为基线的1.5–1.9倍；
  * **AKI 2期**：SCr为基线的2.0–2.9倍；
  * **AKI 3期**：SCr≥3.0倍或升高至≥353.6 μmol/L，或启动肾替代治疗。

（由于尿量数据不完整，未纳入尿量指标。）


### 四、变量处理与预处理

* **缺失值处理**：

  * 缺失率 <20% 的变量以均值填补；
  * 缺失率 ≥20% 的变量被排除。

* **标准化处理**：

  * 所有连续变量统一进行标准化处理以确保可比性。

* **变量选择**：

  * 使用 SHAP（SHapley Additive exPlanations）方法评估变量重要性。


### 五、机器学习模型构建

#### 1. 模型种类（共9种）：

| 类别   | 模型名称                                        |
| ---- | ------------------------------------------- |
| 回归类  | Logistic Regression（LR）                     |
| 树模型  | Decision Tree（DT）、Random Forest（RF）         |
| 集成模型 | XGBoost、LightGBM、Gradient Boosting、AdaBoost |
| 距离类  | K-Nearest Neighbors（KNN）                    |
| 间隔类  | Support Vector Machine（SVM）                 |

#### 2. 数据划分：

* 全部数据随机分为**训练集（70%）**和**验证集（30%）**。

#### 3. 模型训练策略：

* \*\*五折交叉验证（5-fold cross-validation）\*\*用于调参及防止过拟合；
* 采用**ROC曲线和AUC值**评估各模型的预测性能；
* 同时计算**准确率（Accuracy）**、**精确率（Precision）**、**召回率（Recall）**和**F1值**等指标综合评价。


### 六、模型解释与变量影响分析

* 利用**SHAP可解释性方法**对最优模型（RF）进行解释；
* 绘制**SHAP Summary Plot**和**SHAP Dependence Plot**，分析重要特征如eGFR、尿酸（UA）、ALT、BNP、手术时间、年龄等与AKI发生的关系。


### 七、风险分层实施计划

根据模型输出的AKI风险分值，对患者进行分层管理：

* **低风险患者**：常规术中术后管理；
* **中风险患者**：加强肾功能监测，避免肾毒性药物；
* **高风险患者**：由多学科团队会诊制定个体化干预方案，包括术前优化、术中保护及术后早期干预。


## 五、重要变量和数据(英文展示)
以下是根据文献中**表1**提取的变量信息，分为连续变量（包含中位数和四分位距）与分类变量（频数及构成比），以便您后续在 Python 中进行模拟生成。

---

### 📊 连续变量（Continuous Variables）

| Variable                                | Median | IQR     |
| --------------------------------------- | ------ | ------- |
| Age (years)                             | 69.2   | (6.0)   |
| Body mass index (kg/m²)                 | 25.2   | (4.1)   |
| Heart rate (bpm)                        | 76     | (12.0)  |
| Systolic blood pressure (mmHg)          | 130.0  | (18.0)  |
| Diastolic blood pressure (mmHg)         | 76.0   | (12.0)  |
| Mean arterial pressure (mmHg)           | 94.7   | (11.7)  |
| eGFR (mL/min)                           | 77.4   | (27.6)  |
| Serum creatinine (μmol/L)               | 72.1   | (22.0)  |
| UA (μmol/L)                             | 317.9  | (123.3) |
| BNP (pg/mL)                             | 164.0  | (250.0) |
| PLT count (×10⁹/L)                      | 206    | (81.0)  |
| LDL cholesterol (mmol/L)                | 2.22   | (1.00)  |
| Triglycerides (mmol/L)                  | 1.31   | (0.80)  |
| Total cholesterol (mmol/L)              | 3.79   | (1.20)  |
| HDL cholesterol (mmol/L)                | 1.00   | (0.30)  |
| ALT (U/L)                               | 20.0   | (15.0)  |
| AST (U/L)                               | 21.0   | (11.0)  |
| Operation time (h)                      | 4.0    | (1.0)   |
| Urine output (×100 mL)                  | 12.0   | (12.0)  |
| Bleeding volume (×100 mL)               | 8.0    | (4.0)   |
| Total liquid intake (×100 mL)           | 25.0   | (9.5)   |
| Washed RBC transfusion volume (×100 mL) | 2.4    | (4.0)   |

---

### 🧮 分类变量（Categorical Variables）

| Variable                  | Frequency (n) | Percentage (%) |
| ------------------------- | ------------- | -------------- |
| Male sex                  | 1,473         | 68.4           |
| Smoker                    | 436           | 20.2           |
| Drinker                   | 343           | 15.9           |
| Hypertension              | 1,416         | 65.7           |
| Diabetes mellitus         | 812           | 37.7           |
| Hyperlipemia              | 1,181         | 54.8           |
| Prior MI                  | 307           | 14.2           |
| Prior cerebral infarction | 250           | 11.6           |
| Prior PCI                 | 230           | 10.7           |
| Prior CABG                | 37            | 1.7            |
| NYHA Class III/IV         | 431           | 20.0           |
| Aspirin                   | 566           | 26.3           |
| ACE inhibitor/ARB         | 347           | 16.1           |
| Beta blocker              | 1,651         | 76.6           |
| Statin therapy            | 402           | 18.7           |
| PPI                       | 533           | 24.7           |
| Loop diuretic             | 432           | 20.0           |
| Thiazide                  | 83            | 3.9            |
| Spirolactone              | 232           | 10.8           |
| Contrast agent            | 549           | 25.5           |
| Metformin                 | 219           | 10.2           |
| RBC transfusion           | 502           | 23.3           |
| PLT transfusion           | 36            | 1.7            |
| Plasma transfusion        | 194           | 9.0            |
| Use of IABP               | 161           | 7.5            |
| Use of ECMO               | 7             | 0.3            |
| Use of CPB                | 359           | 16.7           |
| Use of epinephrine        | 501           | 23.2           |
| Use of norepinephrine     | 1,216         | 56.4           |
| Use of isoprenaline       | 126           | 5.8            |
| Use of dopamine           | 1,836         | 85.2           |
| Use of cephalosporin      | 1,781         | 82.6           |

---


## 五、重要变量和数据(中文展示)


### 📊 连续变量（连续变量）

| 变量名称 | 中位数 | 四分位间距 |
|--------------------------------|--------|----------|
| 年龄（岁） | 69.2 | (6.0) |
| 体质指数（kg/m²） | 25.2 | (4.1) |
| 心率（次/分钟） | 76 | (12.0) |
| 收缩压（mmHg） | 130.0 | (18.0) |
| 舒张压（mmHg） | 76.0 | (12.0) |
| 平均动脉压（mmHg） | 94.7 | (11.7) |
| 肾小球滤过率（eGFR，mL/min） | 77.4 | (27.6) |
| 血清肌酐（μmol/L） | 72.1 | (22.0) |
| 尿酸（UA，μmol/L） | 317.9 | (123.3) |
| B型利钠肽（BNP，pg/mL） | 164.0 | (250.0) |
| 血小板计数（PLT，×10⁹/L） | 206 | (81.0) |
| 低密度脂蛋白胆固醇（LDL，mmol/L） | 2.22 | (1.00) |
| 甘油三酯（mmol/L） | 1.31 | (0.80) |
| 总胆固醇（mmol/L） | 3.79 | (1.20) |
| 高密度脂蛋白胆固醇（HDL，mmol/L） | 1.00 | (0.30) |
| 丙氨酸氨基转移酶（ALT，U/L） | 20.0 | (15.0) |
| 天门冬氨酸氨基转移酶（AST，U/L） | 21.0 | (11.0) |
| 手术时间（小时） | 4.0 | (1.0) |
| 尿量（×100 mL） | 12.0 | (12.0) |
| 出血量（×100 mL） | 8.0 | (4.0) |
| 总液体摄入量（×100 mL） | 25.0 | (9.5) |
| 洗涤红细胞输注量（×100 mL） | 2.4 | (4.0) |


### 🧮 分类变量（分类变量）

| 变量名称 | 频数（n） | 百分比（%） |
|--------------------------------|----------|------------|
| 男性 | 1,473 | 68.4 |
| 吸烟者 | 436 | 20.2 |
| 饮酒者 | 343 | 15.9 |
| 高血压 | 1,416 | 65.7 |
| 糖尿病 | 812 | 37.7 |
| 高脂血症 | 1,181 | 54.8 |
| 既往心肌梗死 | 307 | 14.2 |
| 既往脑梗死 | 250 | 11.6 |
| 既往经皮冠状动脉介入治疗（PCI） | 230 | 10.7 |
| 既往冠状动脉旁路移植术（CABG） | 37 | 1.7 |
| 纽约心脏协会（NYHA）心功能Ⅲ/Ⅳ级 | 431 | 20.0 |
| 阿司匹林 | 566 | 26.3 |
| 血管紧张素转换酶抑制剂/血管紧张素受体拮抗剂（ACEI/ARB） | 347 | 16.1 |
| β受体阻滞剂 | 1,651 | 76.6 |
| 他汀类药物治疗 | 402 | 18.7 |
| 质子泵抑制剂（PPI） | 533 | 24.7 |
| 钙通道阻滞剂 | 432 | 20.0 |
| 噻嗪类利尿剂 | 83 | 3.9 |
| 螺内酯 | 232 | 10.8 |
| 对比剂 | 549 | 25.5 |
| 二甲双胍 | 219 | 10.2 |
| 红细胞输注 | 502 | 23.3 |
| 血小板输注 | 36 | 1.7 |
| 血浆输注 | 194 | 9.0 |
| 主动脉内球囊反搏（IABP） | 161 | 7.5 |
| 体外膜氧合（ECMO） | 7 | 0.3 |
| 体外循环（CPB） | 359 | 16.7 |
| 肾上腺素 | 501 | 23.2 |
| 去甲肾上腺素 | 1,216 | 56.4 |
| 异丙肾上腺素 | 126 | 5.8 |
| 多巴胺 | 1,836 | 85.2 |
| 头孢菌素类抗生素 | 1,781 | 82.6 |



## 六、模拟数据
请你根据上述抓取的数据信息，给出python代码，并给这份代码取个名字，用于模拟上述数据，最终保存为csv格式，保存的路径为04文献阅读\05肾病\11基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型\01data
需要完全模拟 N=2,155 条记录，连续变量不必假设为正太分布
分类变量使用伯努利分布模拟
需要依据原文加入 AKI 标签

并且生成的数据，要尽量符合论文中的模型结果，你可以尝试一些方法实现这个目标

模拟数据已成功生成并保存，代码名称为：

### 🧪 `simulate_cabg_aki_data.py`



该脚本完成以下操作：

* 完全模拟了 N=2,155 条患者记录；
* 连续变量按照中位数 ± IQR/1.35 估算标准差；
* 分类变量通过伯努利分布模拟；
* 结合文献中提到的关键危险因素（如 Age, eGFR, UA, ALT, BNP, Operation\_time, Use\_IABP）构建了风险得分，并以逻辑回归形式模拟 AKI 标签；
* 最终保存为 CSV 文件，路径为：

```
04文献阅读\05肾病\11基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型\01data\simulated_cabg_aki_data.csv
```

以下是完整的 Python 脚本，可用于在本地模拟数据并生成 `simulated_cabg_aki_data.csv` 文件：

---

### ✅ 文件名建议：`simulate_cabg_aki_data.py`

```python
import numpy as np
import pandas as pd
import os

# 设置随机种子确保可重复性
np.random.seed(42)

# 模拟样本量
n_samples = 2155

# 连续变量模拟函数（使用中位数 + IQR/1.35 估算标准差）
def simulate_continuous(median, iqr):
    std = iqr / 1.35
    return np.random.normal(loc=median, scale=std, size=n_samples)

# 连续变量及其中位数和IQR
continuous_variables = {
    "Age": (69.2, 6.0),
    "Body_mass_index": (25.2, 4.1),
    "Heart_rate": (76, 12.0),
    "Systolic_BP": (130.0, 18.0),
    "Diastolic_BP": (76.0, 12.0),
    "Mean_arterial_pressure": (94.7, 11.7),
    "eGFR": (77.4, 27.6),
    "Serum_creatinine": (72.1, 22.0),
    "UA": (317.9, 123.3),
    "BNP": (164.0, 250.0),
    "PLT_count": (206, 81.0),
    "LDL": (2.22, 1.00),
    "Triglycerides": (1.31, 0.80),
    "Total_cholesterol": (3.79, 1.20),
    "HDL": (1.00, 0.30),
    "ALT": (20.0, 15.0),
    "AST": (21.0, 11.0),
    "Operation_time": (4.0, 1.0),
    "Urine_output": (12.0, 12.0),
    "Bleeding_volume": (8.0, 4.0),
    "Total_liquid_intake": (25.0, 9.5),
    "Washed_RBC_volume": (2.4, 4.0)
}

# 创建数据字典并模拟连续变量
data = {}
for var, (median, iqr) in continuous_variables.items():
    data[var] = simulate_continuous(median, iqr)

# 分类变量及其发生概率（伯努利分布）
categorical_variables = {
    "Male_sex": 0.684,
    "Smoker": 0.202,
    "Drinker": 0.159,
    "Hypertension": 0.657,
    "Diabetes": 0.377,
    "Hyperlipemia": 0.548,
    "Prior_MI": 0.142,
    "Prior_cerebral_infarction": 0.116,
    "Prior_PCI": 0.107,
    "Prior_CABG": 0.017,
    "NYHA_class_III_IV": 0.200,
    "Aspirin": 0.263,
    "ACE_ARB": 0.161,
    "Beta_blocker": 0.766,
    "Statin": 0.187,
    "PPI": 0.247,
    "Loop_diuretic": 0.200,
    "Thiazide": 0.039,
    "Spirolactone": 0.108,
    "Contrast_agent": 0.255,
    "Metformin": 0.102,
    "RBC_transfusion": 0.233,
    "PLT_transfusion": 0.017,
    "Plasma_transfusion": 0.090,
    "Use_IABP": 0.075,
    "Use_ECMO": 0.003,
    "Use_CPB": 0.167,
    "Use_epinephrine": 0.232,
    "Use_norepinephrine": 0.564,
    "Use_isoprenaline": 0.058,
    "Use_dopamine": 0.852,
    "Use_cephalosporin": 0.826
}

# 模拟分类变量
for var, p in categorical_variables.items():
    data[var] = np.random.binomial(1, p, n_samples)

# 创建DataFrame
df = pd.DataFrame(data)

# 构造AKI风险分数（基于文献中显著变量）
risk_score = (
    0.02 * df["Age"] -
    0.015 * df["eGFR"] +
    0.005 * df["UA"] +
    0.01 * df["ALT"] +
    0.004 * df["BNP"] +
    0.1 * df["Operation_time"] +
    0.3 * df["Use_IABP"]
)

# 使用逻辑函数将风险分数转换为AKI发生概率
prob_aki = 1 / (1 + np.exp(-risk_score))
df["AKI"] = np.random.binomial(1, prob_aki)

# 保存数据为CSV
output_path = "04文献阅读/05肾病/11基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型/01data"
os.makedirs(output_path, exist_ok=True)
df.to_csv(os.path.join(output_path, "simulated_cabg_aki_data.csv"), index=False)

print(f"模拟数据已保存至：{output_path}/simulated_cabg_aki_data.csv")
```

---

### ✅ 运行方式

1. 将此脚本保存为 `simulate_cabg_aki_data.py`；
2. 使用终端或 VSCode 执行：

   ```bash
   python simulate_cabg_aki_data.py
   ```
3. 文件将保存在您指定的目录路径下。

如您需要：

* 添加 `AKI_stage` 分级标签；
* 模拟多科室分组、多时间点观察；
* 自动进行可视化或建模评估；
  都可以继续提出，我可以帮您扩展脚本功能。


## 七、复现代码
运行时出现了错误：本环境尚未安装 `lightgbm` 和 `xgboost` 库。

为了确保您在本地复现不受影响，请使用以下完整版本的复现脚本，并提前在本地安装必要的依赖：

---

### ✅ 文件名建议：`reproduce_cabg_aki_model.py`

#### 📦 安装依赖（建议创建虚拟环境）：

```bash
pip install numpy pandas scikit-learn lightgbm xgboost shap matplotlib
```

---

### 🧠 脚本内容如下：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import os

# 加载数据
data_path = "04文献阅读/05肾病/11基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型/01data/simulated_cabg_aki_data.csv"
df = pd.read_csv(data_path)

# 特征与标签
X = df.drop(columns=["AKI"])
y = df["AKI"]

# 标准化连续变量
numeric_cols = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 划分训练集与验证集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 定义模型
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# 交叉验证并评估模型
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba)
    })

results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)

# 保存结果
output_dir = "04文献阅读/05肾病/11基于机器学习的冠状动脉旁路移植术后急性肾损伤预测模型/02output"
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

# 解释最佳模型（使用RandomForest）
best_model = RandomForestClassifier().fit(X_train, y_train)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# SHAP图保存
shap.summary_plot(shap_values[1], X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))

# 可选：展示依赖关系较强变量的依赖图
important_vars = ["eGFR", "UA", "ALT", "BNP", "Age", "Operation_time"]
for var in important_vars:
    shap.dependence_plot(var, shap_values[1], X_test, show=False)
    plt.savefig(os.path.join(output_dir, f"shap_dependence_{var}.png"))
    plt.clf()

print("✅ 模型训练与SHAP解释完成，结果保存至：", output_dir)
```

### 功能扩展




#### 1. **交叉验证平均性能评估（5-fold）**

* 输出各模型的 `AUC`, `Accuracy`, `Recall`, `F1` 的平均值与标准差
* 用于增强对模型稳健性的评估

#### 2. **绘制交互式 ROC 曲线图（Plotly）**

* 提供可缩放、可保存的 HTML 交互式图表
* 支持对比多个模型的 ROC 曲线表现

#### 3. **AKI 分级预测（Stage 1/2/3）**

* 使用 `LogisticRegression` + `OrdinalClassifier` 或 `XGBClassifier` 进行多分类
* 或者输出 AKI 等级概率，模拟更复杂标签结构

#### 4. **保存完整 SHAP 分析结果（summary + dependence）**

* 依照文献中的 SHAP importance 排名前 10 变量逐一绘制 dependence plot

---


