# 预测糖尿病酮症酸中毒相关急性肾损伤的危险因素：使用 XGBoost 的机器学习方法

## 一、文献信息


| 项目    | 内容                                                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| 标题    | Predicting the risk factors of diabetic ketoacidosis-associated acute kidney injury: A machine learning approach using XGBoost |
| 作者    | Tingting Fan, Jiaxin Wang, Luyao Li, Jing Kang, Wenrui Wang, Chuan Zhang                                                       |
| 发表时间  | 2023年4月6日                                                                                                                      |
| 国家    | 中国（吉林大学第二医院）                                                                                                                   |
| 分区    | JCR Q2（基于《Frontiers in Public Health》）                                                                                         |
| 影响因子  | 5.2（2023年）                                                                                                                     |
| 摘要    | 本研究基于MIMIC-IV数据库，利用XGBoost机器学习算法开发预测糖尿病酮症酸中毒（DKA）相关急性肾损伤（AKI）的模型，以早期识别高风险患者，辅助临床决策。                                            |
| 关键词   | diabetic ketosis, acute kidney injury, machine learning, XGBoost, outcome                                                      |
| 期刊名称  | Frontiers in Public Health                                                                                                     |
| 卷号/期号 | Vol. 11, Article 1087297                                                                                                       |
| DOI   | [10.3389/fpubh.2023.1087297](https://doi.org/10.3389/fpubh.2023.1087297)                                                       |
| 研究方法  | 回顾性研究；机器学习模型（XGBoost）；LASSO特征选择；交叉验证；多模型比较                                                                                     |
| 数据来源  | MIMIC-IV数据库（2008–2019，Beth Israel Deaconess Medical Center）                                                                    |
| 研究结果  | 1,322例DKA患者中有497例（37.6%）在ICU住院一周内发生AKI，XGBoost模型在训练集和验证集中的AUC分别为0.835和0.800，表现优于其他7种模型。主要预测因素包括BUN、尿量、体重、年龄和血小板计数。             |
| 研究结论  | 构建的ML模型可有效预测DKA相关AKI，有助于早期识别高危患者，辅助临床干预并改善预后。                                                                                  |
| 研究意义  | 提供了一种基于真实世界大数据与先进算法构建的临床预测工具，填补了针对DKA-AKI机器学习预测模型的研究空白，具有潜在的推广价值。                                                              |




## 二、核心内容

本研究基于MIMIC-IV数据库，采用XGBoost机器学习算法构建并验证了糖尿病酮症酸中毒（DKA）相关急性肾损伤（AKI）的预测模型，筛选出关键风险因素，实现对高风险患者的早期识别与临床干预支持。

### 1. **研究背景与目的**

* DKA 是糖尿病常见的急性并发症，常并发 AKI，显著增加死亡率和住院时间。
* 目前临床上尚缺乏有效工具预测 DKA 发生 AKI 的风险。
* 本研究旨在构建一个基于机器学习的个体化风险预测模型，提高 ICU 患者早期风险识别能力。

### 2. **数据来源与研究设计**

* 数据来自 MIMIC-IV 数据库（2008–2019），包含1,322例符合ICD-9/10诊断的DKA患者。
* 研究为回顾性队列设计，将患者按8:2比例分为训练集与验证集。

### 3. **方法与建模过程**

* 采集人口学特征、生命体征、实验室指标和干预治疗等变量。
* 使用LASSO方法进行特征选择，构建8种机器学习模型进行比较。
* 最终选择表现最优的 XGBoost 作为预测模型。

### 4. **主要结果**

* 共有497例（37.6%）DKA患者在ICU住院一周内发生AKI。
* XGBoost模型在训练集与验证集中的AUC分别为0.835和0.800，表现优于其他模型。
* 模型最重要的预测因素依次为：血尿素氮（BUN）、尿量、体重、年龄和血小板计数（PLT）。

### 5. **研究结论**

* XGBoost模型表现优异，可用于早期识别DKA-AKI高危患者。
* 可辅助临床制定干预策略，有望改善患者预后。

### 6. **研究意义**

* 本研究为DKA相关AKI风险预测提供了精准化、数据驱动的临床决策工具。
* 填补了该领域机器学习方法应用的空白，具备推广与多中心验证的潜力。

### 7. **局限性**

* 单中心数据、回顾性设计可能影响外部适用性。
* 数据中存在缺失，部分变量因缺失比例高被剔除。
* 无法排除入组前已发生AKI的患者。




## 三、文章小结


### 1. **Introduction 引言**

* 介绍DKA（糖尿病酮症酸中毒）是一种严重急性并发症，常引发AKI（急性肾损伤），增加住院时间、死亡率和慢性肾病风险。
* 强调传统AKI诊断依赖肌酐和尿量变化，存在延迟识别问题。
* 提出基于机器学习的预测模型可实现早期干预，但针对DKA-AKI的研究尚少，因此本研究拟开发ML模型以预测DKA患者发生AKI的风险。

---

### 2. **Methods 方法**

#### 2.1 Database 数据库

* 数据来源为 **MIMIC-IV**，包含2008–2019年美国BIDMC重症监护病人电子病历。

#### 2.2 Study Population 研究对象

* 纳入ICD-9/10诊断为DKA的ICU患者，排除CKD 5期、重复住院记录及缺失数据>20%的样本，共计1,322人。

#### 2.3 Data Extraction and Pre-processing 数据提取与预处理

* 提取人口统计、生命体征、实验室指标、治疗措施等；对缺失值采用最近邻填补法，所有特征为入ICU后24小时内数据。

#### 2.4 Outcome 结局

* 主要结局为入ICU后一周内发生AKI（基于KDIGO标准）。

#### 2.5 Model Development and Validation 模型构建与验证

* 使用LASSO交叉验证筛选特征变量，构建8种监督学习模型（如XGBoost、Logistic回归、SVM等）。
* 采用AUC、灵敏度、特异度等指标评估模型性能，最终选择XGBoost为最佳模型。

#### 2.6 Statistical Analysis 统计分析

* 使用Python和R进行数据分析；连续变量用t检验或Mann–Whitney U检验，分类变量用χ²检验或Fisher确切检验。

---

### 3. **Results 结果**

#### 3.1 Patients’ Characteristics 患者特征

* 最终纳入1,322名患者，其中497例（37.6%）在一周内发生AKI。
* AKI组患者年龄更大、BUN更高、尿量更低、体重更大，且住院时间与死亡率显著增加。

#### 3.2 Predictive Model Performance 预测模型表现

* XGBoost在训练集与验证集中的AUC分别为0.835和0.800，准确率和灵敏度优于其他模型。
* 模型经DCA（决策曲线分析）与校准曲线验证，表现稳健。

#### 3.3 Relative Importance of Variables 特征重要性

* 最重要的五个变量依次为：BUN、尿量、体重、年龄、血小板计数（PLT）。

---

### 4. **Discussion 讨论**

* 本研究首次构建了针对DKA-AKI的ML预测模型，在MIMIC-IV真实世界大数据上验证。
* 提出了体重和PLT等新颖预测指标，强调其临床相关性。
* 讨论了肥胖、高龄、尿量减少等机制与AKI发生的关系。
* 指出模型在识别高危人群、指导早期干预方面具有实际应用价值。
* 提出局限性：单中心数据、回顾性设计、变量缺失、潜在混杂因素未完全控制。

---

### 5. **Conclusion 结论**

* 成功建立并验证了一个基于XGBoost的DKA-AKI预测模型。
* 可用于早期识别高风险患者，辅助个体化临床管理，改善患者预后。


## 四、主要方法和实施计划

本研究为一项**基于MIMIC-IV数据库的回顾性队列研究**，整体流程包括数据提取、变量筛选、模型构建、性能评估和解释五个步骤。

---

### 1. **数据来源（2.1 Database）**

* 使用公开的**MIMIC-IV数据库**（Medical Information Mart for Intensive Care IV）。
* 数据涵盖2008–2019年在美国Beth Israel Deaconess Medical Center住院ICU患者的临床资料，包括：

  * 人口统计学信息
  * 生命体征
  * 实验室检查
  * 治疗与药物使用
  * 生存与预后状态
* 使用此数据库前已完成认证（certificate number: 9168028）。

---

### 2. **研究对象（2.2 Study Population）**

* **纳入标准**：

  * 依据ICD-9/10编码确诊为糖尿病酮症酸中毒（DKA）；
* **排除标准**：

  1. 慢性肾病（CKD）5期患者；
  2. 同一住院期间ICU多次入科者，仅保留首次入ICU记录；
  3. 缺失数据比例超过20%的病例。

> 最终纳入 1,322 名 DKA 患者用于建模。

---

### 3. **数据提取与预处理（2.3 Data Extraction and Pre-processing）**

* **提取变量包括**：

  * 人口学资料（年龄、性别、体重等）
  * 生命体征（HR、RR、血压等）
  * 实验室指标（BUN、肌酐、血糖、血小板计数等）
  * 合并症（高血压、AMI、感染等）
  * 评分系统（SOFA、SAPS-II、OASIS）
  * 处理干预（是否机械通气、是否使用CRRT等）
* **预处理方式**：

  * 对缺失值比例＞20%的变量予以删除；
  * 其余缺失值使用**K最近邻（KNN）算法插补**；
  * 所有特征均在入ICU后**24小时内采集**；
  * 若同一变量在24小时内有多次记录，仅保留**首次检测值**。

---

### 4. **研究结局定义（2.4 Outcome）**

* **主要结局**为：入ICU后一周内发生急性肾损伤（AKI）。
* 依据**KDIGO标准**进行AKI诊断：

  * 血清肌酐水平变化；
  * 尿量减少；
  * 临床医嘱记录等。

---

### 5. **模型构建与验证（2.5 Model Development and Validation）**

#### 5.1 特征选择

* 使用\*\*LASSO交叉验证（Least Absolute Shrinkage and Selection Operator CV）\*\*进行特征筛选，以降低模型复杂度、防止过拟合，并提高建模效率。

#### 5.2 建模算法

* 共比较**8种机器学习模型**：

  1. XGBoost（eXtreme Gradient Boosting）
  2. Logistic Regression
  3. LightGBM
  4. AdaBoost
  5. Gaussian Naïve Bayes (GNB)
  6. Complement Naïve Bayes (CNB)
  7. Multi-Layer Perceptron (MLP)
  8. Support Vector Machine (SVM)

#### 5.3 数据划分与交叉验证

* 将数据集**随机划分为训练集（85%）和验证集（15%）**。
* 使用**10折交叉验证**调优参数，提升泛化性能。

#### 5.4 性能评估指标

* 使用以下指标综合评估模型表现：

  * AUC（曲线下面积）
  * Accuracy（准确率）
  * Sensitivity（敏感性/召回率）
  * Specificity（特异性）
  * PPV/NPV（阳性/阴性预测值）
  * F1分数（平衡准确率与召回率）

#### 5.5 模型选择与解释

* 最终选择**XGBoost**为最佳模型，因其在训练集和验证集上均表现最优（AUC最高）。
* 使用\*\*特征重要性分析（Feature Importance）\*\*解释模型贡献变量，排名前五为：

  1. 血尿素氮（BUN）
  2. 尿量
  3. 体重
  4. 年龄
  5. 血小板计数（PLT）

---

### 6. **统计分析（2.6 Statistical Analysis）**

* 软件：R 3.6.3 和 Python 3.7。
* 正态分布变量：均值±标准差，使用t检验；
* 非正态分布变量：使用中位数+IQR，采用Mann–Whitney U检验；
* 类别变量：χ²检验或Fisher确切检验；
* 代码由“Extreme Smart Analysis平台”提供并上传至附录。

---

### 总结（实施计划要点）

| 步骤   | 内容                      |
| ---- | ----------------------- |
| 数据源  | MIMIC-IV（ICU数据）         |
| 研究类型 | 回顾性队列研究                 |
| 建模方法 | 监督学习 + LASSO特征选择        |
| 算法比较 | 8种机器学习模型（含XGBoost）      |
| 验证方式 | 训练/验证集 + 10折交叉验证        |
| 评价指标 | AUC、敏感性、特异性、准确率等        |
| 输出结果 | 构建DKA-AKI预测模型 + 特征重要性解释 |




## 五、重要变量和数据(英文展示)


### 🧮 Continuous Variables Summary

| Variable     | Median (Total) | IQR (Total)     | Median (AKI) | IQR (AKI)       | Median (Non-AKI) | IQR (Non-AKI)   | p-value |
| ------------ | -------------- | --------------- | ------------ | --------------- | ---------------- | --------------- | ------- |
| Age          | 50             | \[35, 62]       | 58           | \[46, 68]       | 43               | \[30, 57]       | <0.001  |
| Weight       | 73.2           | \[62.0, 87.1]   | 78.5         | \[66.0, 94.0]   | 70.0             | \[60.3, 83.6]   | <0.001  |
| HR           | 100            | \[88, 111]      | 98           | \[85, 109]      | 101              | \[89, 113]      | <0.001  |
| RR           | 19             | \[16, 23]       | 20           | \[17, 24]       | 19               | \[16, 23]       | 0.002   |
| DBP          | 71             | \[60, 83]       | 69           | \[56, 82]       | 72               | \[62, 83]       | 0.002   |
| BUN          | 16             | \[9, 31]        | 28           | \[14, 46]       | 13               | \[8, 20]        | <0.001  |
| Scr          | 0.9            | \[0.7, 1.5]     | 1.4          | \[0.9, 2.5]     | 0.8              | \[0.6, 1.1]     | <0.001  |
| Urine output | 1950           | \[1085, 3000]   | 1300         | \[600, 2250]    | 2300             | \[1470, 3440]   | <0.001  |
| eGFR         | 0.992          | \[0.854, 1.093] | 0.930        | \[0.829, 1.067] | 1.000            | \[0.882, 1.114] | <0.001  |


### 🧾 Categorical Variables Summary

| Variable                     | Total n (%) | AKI n (%)   | Non-AKI n (%) | p-value |
| ---------------------------- | ----------- | ----------- | ------------- | ------- |
| Gender (Female)              | 689 (52.1%) | 264 (53.1%) | 425 (51.5%)   | 0.572   |
| Preexisting CKD              | 316 (23.9%) | 195 (39.2%) | 121 (14.7%)   | <0.001  |
| UTI (Yes)                    | 147 (11.1%) | 76 (15.3%)  | 71 (8.6%)     | <0.001  |
| Mechanical ventilation (Yes) | 152 (11.5%) | 121 (24.3%) | 31 (3.8%)     | <0.001  |





## 五、重要变量和数据(中文展示)


### 连续变量统计信息（中位数、四分位数、p值）

| 变量名              | 总体中位数 | 总体IQR           | AKI组中位数 | AKI组IQR         | 非AKI组中位数 | 非AKI组IQR        | p值     |
| ---------------- | ----- | --------------- | ------- | --------------- | -------- | --------------- | ------ |
| 年龄（Age）          | 50    | \[35, 62]       | 58      | \[46, 68]       | 43       | \[30, 57]       | <0.001 |
| 体重（Weight）       | 73.2  | \[62.0, 87.1]   | 78.5    | \[66.0, 94.0]   | 70.0     | \[60.3, 83.6]   | <0.001 |
| 心率（HR）           | 100   | \[88, 111]      | 98      | \[85, 109]      | 101      | \[89, 113]      | <0.001 |
| 呼吸频率（RR）         | 19    | \[16, 23]       | 20      | \[17, 24]       | 19       | \[16, 23]       | 0.002  |
| 舒张压（DBP）         | 71    | \[60, 83]       | 69      | \[56, 82]       | 72       | \[62, 83]       | 0.002  |
| 血尿素氮（BUN）        | 16    | \[9, 31]        | 28      | \[14, 46]       | 13       | \[8, 20]        | <0.001 |
| 肌酐（Scr）          | 0.9   | \[0.7, 1.5]     | 1.4     | \[0.9, 2.5]     | 0.8      | \[0.6, 1.1]     | <0.001 |
| 尿量（Urine output） | 1950  | \[1085, 3000]   | 1300    | \[600, 2250]    | 2300     | \[1470, 3440]   | <0.001 |
| eGFR（估算肾小球滤过率）   | 0.992 | \[0.854, 1.093] | 0.930   | \[0.829, 1.067] | 1.000    | \[0.882, 1.114] | <0.001 |

---

### 分类变量统计信息（频数、构成比、p值）

| 变量名                               | 总人数 | 总体比例（%） | AKI人数 | AKI比例（%） | 非AKI人数 | 非AKI比例（%） | p值     |
| --------------------------------- | --- | ------- | ----- | -------- | ------ | --------- | ------ |
| 女性（Gender, Female）                | 689 | 52.1    | 264   | 53.1     | 425    | 51.5      | 0.572  |
| 既往慢性肾病（Preexisting CKD）           | 316 | 23.9    | 195   | 39.2     | 121    | 14.7      | <0.001 |
| 泌尿感染（UTI, Yes）                    | 147 | 11.1    | 76    | 15.3     | 71     | 8.6       | <0.001 |
| 机械通气（Mechanical ventilation, Yes） | 152 | 11.5    | 121   | 24.3     | 31     | 3.8       | <0.001 |




## 六、模拟数据
以下是加入“是否发生 AKI”标签变量后的完整 Jupyter Notebook 脚本（`simulate_DKA_AKI_data.ipynb`）。标签变量的生成依据文献中发现的关键特征（如 BUN 升高、尿量减少、年龄增加等）构建逻辑规则和概率模型。

---

```python
# simulate_DKA_AKI_data.ipynb
# -----------------------------------------------
# 模拟文献《Predicting the risk factors of DKA-associated AKI》中的基础变量数据
# 并生成标签变量 AKI（是否发生急性肾损伤）
# 用于构建机器学习预测模型或分析训练集特征
# -----------------------------------------------

import numpy as np
import pandas as pd
import os

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 模拟样本数量
n_samples = 1000

# -----------------------------------------------
# 连续变量模拟（近似正态分布 + 截断）
# -----------------------------------------------
age = np.random.normal(loc=50, scale=12, size=n_samples).clip(18, 90)
weight = np.random.normal(loc=73.2, scale=15, size=n_samples).clip(40, 140)
hr = np.random.normal(loc=100, scale=10, size=n_samples).clip(50, 150)
rr = np.random.normal(loc=19, scale=3, size=n_samples).clip(10, 40)
dbp = np.random.normal(loc=71, scale=10, size=n_samples).clip(40, 120)
bun = np.random.normal(loc=16, scale=10, size=n_samples).clip(3, 80)
scr = np.random.normal(loc=0.9, scale=0.5, size=n_samples).clip(0.2, 5)
urine_output = np.random.normal(loc=1950, scale=1000, size=n_samples).clip(100, 6000)
egfr = np.random.normal(loc=1.0, scale=0.2, size=n_samples).clip(0.2, 2)

# -----------------------------------------------
# 分类变量模拟（使用二项分布）
# -----------------------------------------------
gender_female = np.random.binomial(1, 0.521, size=n_samples)
preexisting_ckd = np.random.binomial(1, 0.239, size=n_samples)
uti_yes = np.random.binomial(1, 0.111, size=n_samples)
mech_vent_yes = np.random.binomial(1, 0.115, size=n_samples)

# -----------------------------------------------
# 构建 DataFrame（未包含标签）
# -----------------------------------------------
df = pd.DataFrame({
    "Age": age,
    "Weight": weight,
    "HR": hr,
    "RR": rr,
    "DBP": dbp,
    "BUN": bun,
    "Scr": scr,
    "Urine_output": urine_output,
    "eGFR": egfr,
    "Gender_Female": gender_female,
    "Preexisting_CKD": preexisting_ckd,
    "UTI_Yes": uti_yes,
    "Mechanical_ventilation_Yes": mech_vent_yes
})

# -----------------------------------------------
# 生成 AKI 标签变量（根据关键变量设定概率规则）
# 影响因子：BUN↑、Urine_output↓、Age↑、PLT↓（此处未模拟 PLT，用前三者）
# -----------------------------------------------
# 归一化得分
bun_score = (df["BUN"] - 10) / 40     # 正向
urine_score = (3000 - df["Urine_output"]) / 3000  # 逆向
age_score = (df["Age"] - 30) / 60      # 正向
ckd_score = df["Preexisting_CKD"] * 0.3
vent_score = df["Mechanical_ventilation_Yes"] * 0.3

# 综合得分：线性组合，归一化到概率空间（0~1）
risk_score = (
    0.3 * bun_score +
    0.3 * urine_score +
    0.2 * age_score +
    ckd_score +
    vent_score
).clip(0, 1)

# 根据风险概率随机生成 AKI 标签（1=发生，0=未发生）
aki = np.random.binomial(1, risk_score)

# 添加标签列
df["AKI"] = aki

# 打印模拟总体 AKI 发生率
aki_rate = df["AKI"].mean()
print(f"🔎 模拟总体 AKI 发生率为：{aki_rate:.2%}")

# -----------------------------------------------
# 保存数据为 CSV 文件
# -----------------------------------------------
save_path = "04文献阅读/05肾病/13预测糖尿病酮症酸中毒相关急性肾损伤的危险因素/01data"
os.makedirs(save_path, exist_ok=True)
file_path = os.path.join(save_path, "simulated_DKA_AKI_data.csv")
df.to_csv(file_path, index=False)

print(f"✅ 模拟数据已保存至：{file_path}")
```

---

### 📝 特点说明：

* 模拟 AKI 的逻辑基于文献中公认的关键预测因子（如 BUN、尿量、年龄、CKD、是否机械通气）。
* 每个样本通过逻辑加权得出 AKI 风险概率，再用二项分布采样生成标签。
* 你可以通过 `df["AKI"].mean()` 检查发生率（应接近真实的 37.6%）。
* 可直接用于模型开发、特征重要性分析与机器学习实训。

如需扩展为分类分层数据（AKI分期1/2/3），或加入 PLT、GLU 等变量模拟，请继续告知。



## 七、复现代码
好的，以下是完整的 Jupyter Notebook 代码，文件名建议为：

---

### 📘 文件名：`reproduce_DKA_AKI_prediction_pipeline.ipynb`

---

```python
# reproduce_DKA_AKI_prediction_pipeline.ipynb
# -------------------------------------------
# 用于复现文献《Predicting the risk factors of DKA-associated AKI》
# 的模型构建与验证流程，基于模拟数据实现机器学习建模、评估与特征重要性分析
# -------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------
# 1. 加载模拟数据
# -------------------------------------------
data_path = "04文献阅读/05肾病/13预测糖尿病酮症酸中毒相关急性肾损伤的危险因素/01data/simulated_DKA_AKI_data.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["AKI"])
y = df["AKI"]

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------
# 2. 特征选择：LASSO
# -------------------------------------------
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)
selected = lasso.coef_ != 0
X_selected = X.loc[:, selected]

print(f"✅ 选中特征数量: {X_selected.shape[1]}")
print(f"特征名: {list(X_selected.columns)}")

# -------------------------------------------
# 3. 数据集划分
# -------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.15, stratify=y, random_state=42
)

# -------------------------------------------
# 4. 模型列表（8种）
# -------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GaussianNB": GaussianNB(),
    "ComplementNB": ComplementNB(),
    "MLP": MLPClassifier(max_iter=500),
    "SVM": SVC(probability=True)
}

# -------------------------------------------
# 5. 训练与评估
# -------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "AUC": roc_auc_score(y_val, y_prob),
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1 Score": f1_score(y_val, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("📊 模型评估结果：")
print(results_df)

# -------------------------------------------
# 6. 特征重要性分析（XGBoost）
# -------------------------------------------
best_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
best_model.fit(X_train, y_train)

importances = best_model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X_selected.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# 可视化
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_df)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()
```

---

如您需要后续继续加入：

* 校准曲线（Calibration plot）
* 决策曲线分析（DCA）
* 混淆矩阵可视化

也可以继续在此基础上扩展。欢迎随时继续提问！


## 八、参考作者提供的核心代码

明白，您希望**完全复现论文中使用的全部模型比较过程**。根据论文与您提供的函数库代码，以下是最终生成的完整复现脚本结构说明与源码：

---

## 📁 文件名称

**`reproduce_DKA_AKI_full_pipeline.py`**

---

## ✅ 功能说明

该脚本将完成以下工作：

1. 读取已准备好的模拟数据；
2. 使用 LASSO 进行特征选择；
3. 使用如下 8 种模型进行建模、交叉验证、性能评估、绘图（ROC、DCA、校准等）：

   * LogisticRegression
   * XGBClassifier
   * LightGBMClassifier
   * AdaBoostClassifier
   * GaussianNB
   * ComplementNB
   * MLPClassifier
   * SVC
4. 使用 SHAP 进行模型解释；
5. 自动保存模型及图像结果。

---

## 🧾 运行环境依赖（推荐提前安装）

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn
```

---

## 📜 脚本源码：`reproduce_DKA_AKI_full_pipeline.py`

```python
# reproduce_DKA_AKI_full_pipeline.py

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from Machine_Learning_Code import ML_Classfication

# ---------------------- #
# 📌 路径配置
# ---------------------- #
data_path = "04文献阅读/05肾病/13预测糖尿病酮症酸中毒相关急性肾损伤的危险因素/01data/simulated_DKA_AKI_data.csv"
save_path = "04文献阅读/05肾病/13预测糖尿病酮症酸中毒相关急性肾损伤的危险因素/02results/"

# ---------------------- #
# 📌 数据加载与准备
# ---------------------- #
df = pd.read_csv(data_path)

# 因变量（标签）
target_column = "AKI"

# 特征列：去除标签列
features = [col for col in df.columns if col != target_column]

# ---------------------- #
# ✂️ 特征选择（LASSO）
# ---------------------- #
X = df[features]
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=0).fit(X_scaled, y)
selected_features = [features[i] for i, coef in enumerate(lasso.coef_) if coef != 0]

print(f"🎯 LASSO selected features ({len(selected_features)}):\n{selected_features}")

# ---------------------- #
# 🤖 模型构建与评估
# ---------------------- #
models_to_run = [
    'LogisticRegression',
    'XGBClassifier',
    'LGBMClassifier',
    'AdaBoostClassifier',
    'GaussianNB',
    'ComplementNB',
    'MLPClassifier',
    'SVC'
]

# 循环运行各模型
for model_name in models_to_run:
    print(f"\n🧪 Running model: {model_name} ...")
    result = ML_Classfication(
        df=df,
        group=target_column,
        features=selected_features,
        method=model_name,
        scoring='roc_auc',
        n_splits=10,
        validation_ratio=0.15,
        savePath=save_path,
        explain=True,
        shapSet=2,
        explain_sample=4,
        explain_numvar=5,
        dpi=300,
        picFormat='png',
        decimal_num=3,
        resultType=0,  # use SD, not CI
        modelSave=True,
        randomState=42
    )
    print(result["str_result"]["分析结果描述"])
```

---

## ✅ 如何使用：

1. 将该脚本与 `Machine_Learning_Code.txt` 中的函数整理为模块（建议保存为 `Machine_Learning_Code.py`）；
2. 保证模拟数据 `simulated_DKA_AKI_data.csv` 已生成；
3. 在命令行或 Jupyter 中运行：

```bash
python reproduce_DKA_AKI_full_pipeline.py
```

---

如果您还需要：

* 生成交互式 SHAP force plot；
* 将结果汇总为 Excel 或 Markdown 报告；
* 加入 DeLong 检验与多模型对比图（如森林图）；

请告诉我，我将继续扩展脚本。是否需要我帮您封装为 `.ipynb` notebook 版本？
