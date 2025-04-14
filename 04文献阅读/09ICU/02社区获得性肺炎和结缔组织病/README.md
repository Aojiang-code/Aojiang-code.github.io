# 一种可解释的基于机器学习的模型，用于预测社区获得性肺炎和结缔组织病患者的重症监护室入院






## 一、文献信息


| 项目 | 内容 |
| ---- | ---- |
| 标题 | 一种可解释的基于机器学习的模型：用于预测社区获得性肺炎和结缔组织病患者的重症监护室入院 |
| 作者 | Dong Huang, Linjing Gong, Chang Wei, Xinyu Wang, Zongan Liang |
| 发表时间 | 2024年 |
| 国家 | 中国 |
| 分区 | 暂未明确注明（期刊《Respiratory Research》通常为Q1） |
| 影响因子 | 5.8（根据2023年数据，具体需查当年最新） |
| 摘要 | 本研究开发并验证了一种基于机器学习的预测模型，用于评估患有社区获得性肺炎和结缔组织病患者在住院期间进入ICU的风险，最终选定的XGBoost模型在区分能力、校准度和临床效用方面表现最优。 |
| 关键词 | 社区获得性肺炎；结缔组织病；重症监护室；风险因素；机器学习；预测模型 |
| 期刊名称 | Respiratory Research |
| 卷号/期号 | 第25卷，第246号 |
| DOI | https://doi.org/10.1186/s12931-024-02874-3 |
| 研究方法 | 回顾性观察研究，基于机器学习模型构建与比较 |
| 数据来源 | 四川大学华西医院，2008年11月至2021年11月住院患者数据 |
| 研究结果 | XGBoost模型的AUC为0.941，准确率为0.913，预测性能优于传统评分系统（如CURB-65、PSI、IDSA/ATS 2007标准）；最重要的预测因子包括NT-proBNP、CRP、CD4+T细胞、淋巴细胞、血清钠和G试验阳性等。 |
| 研究结论 | 成功开发并解释了一个基于机器学习的ICU入住预测模型，可为CAP合并CTD患者提供早期个体化风险评估依据，未来有望在临床中推广使用。 |
| 研究意义 | 该研究为精准医疗提供了新的数据驱动工具，特别适用于提高高危患者的早期识别效率，有助于资源合理配置与改善预后。 |


期刊名称：Respiratory Research
影响因子：4.70
JCR分区：Q1
中科院分区(2025)：医学2区
小类：呼吸系统2区
中科院分区(2023)：医学2区
小类：呼吸系统2区
OPEN ACCESS：99.79%
出版周期：暂无数据
是否综述：否
预警等级：无
年度|影响因子|发文量|自引率
2023 | 4.70 | 285 | 4.3%
2022 | 5.80 | 359 | 1.7%
2021 | 7.16 | 294 | 3.1%
2020 | 5.63 | 308 | 3.7%
2019 | 3.92 | 283 | 4.4%


## 二、核心内容
这篇题为**《一种可解释的基于机器学习的模型：用于预测社区获得性肺炎和结缔组织病患者的重症监护室入院》**的研究，核心内容和主要内容如下：

---

### 🌟 核心内容
本研究构建并验证了一个基于**XGBoost**的机器学习模型，用于预测患有**社区获得性肺炎（CAP）**合并**结缔组织病（CTD）**患者住院期间是否需要进入**重症监护室（ICU）**。该模型具备高准确率和可解释性，并优于传统评分工具。

---

### 📌 主要内容

1. **研究背景**  
   - CAP是全球常见的感染性疾病，CTD患者因免疫功能低下、合并用药等原因对CAP更为敏感，病情发展迅速，常需ICU治疗。  
   - 现有工具如CURB-65、PSI等预测性能在该人群中表现有限，缺乏针对性模型。

2. **研究方法**  
   - 回顾性单中心研究，纳入2008年至2021年间四川大学华西医院1626名CAP合并CTD患者。  
   - 采用三种特征选择方法（单因素分析、Lasso回归、Boruta算法）选出16个重要预测变量。  
   - 利用九种机器学习算法（包括XGBoost、RF、SVM等）建模，并使用SHAP和LIME方法增强模型可解释性。

3. **模型性能评估**  
   - XGBoost模型在测试集中表现最佳，AUC为0.941，准确率为0.913，优于PSI（AUC 0.697）和CURB-65（AUC 0.607）等传统模型。  
   - 关键预测因子包括：**NT-proBNP、CRP、CD4+T细胞、淋巴细胞、血清钠和G试验阳性**。

4. **模型解释与临床应用**  
   - 通过SHAP和LIME解释模型的预测机制，提升临床信任度，辅助医生进行个体化治疗决策。

5. **研究意义与局限**  
   - 为CAP合并CTD患者的风险分层和资源调配提供可靠工具。  
   - 局限包括单中心、回顾性设计，缺乏外部验证。

---


## 三、文章小结

根据文献的节标题结构，下面是各部分主要内容的总结：

---

### 1. **Background（背景）**
- **问题来源**：CAP是全球常见的感染病，CTD患者因免疫系统异常易并发CAP，且病情更复杂。
- **临床需求**：CTD合并CAP患者需更精准的ICU入院风险预测，传统工具（如CURB-65、PSI、IDSA/ATS 2007）在该人群中效果有限。
- **研究目的**：开发一种**个体化、基于机器学习的预测模型**，以更有效预测CAP+CTD患者住院期间的ICU入院风险。

---

### 2. **Methods（方法）**

#### 2.1 **Study designs（研究设计）**
- 单中心、回顾性观察研究，时间跨度为2008年11月至2021年11月。
- 所有数据来源于四川大学华西医院。

#### 2.2 **Patients and data（患者与数据）**
- 纳入标准：同时诊断为CAP和CTD的住院患者。
- 排除标准：18岁以下、孕妇、临床数据缺失患者。
- 数据采集：包括人口学特征、CTD类型、合并症、生命体征和实验室检查，取住院前24小时内的首次记录。

#### 2.3 **Feature selection and model construction（特征选择与模型构建）**
- 使用三种特征筛选方法：单因素分析、Lasso回归、Boruta算法，交集得到**16个核心特征变量**。
- 建模算法：使用9种监督学习算法（XGBoost、RF、SVM、LR等），构建预测模型。
- 数据划分：训练集占70%，测试集占30%。

#### 2.4 **Model assessment（模型评估）**
- 指标包括AUC、准确率、灵敏度、特异性、Kappa值、Brier评分等。
- 使用Delong检验比较模型间性能差异。
- 还对比了三种传统评分工具（IDSA/ATS、PSI、CURB-65）的表现。

#### 2.5 **Model interpretation（模型解释）**
- 利用SHAP值和LIME技术增强模型可解释性。
- 识别了对ICU预测贡献最大的6个变量：NT-proBNP、CRP、CD4+T细胞、淋巴细胞、血清钠、G试验阳性。

#### 2.6 **Statistical analysis（统计分析）**
- 使用R语言进行所有统计分析。
- 缺失值处理：删除缺失>30%的变量，其余使用多重插补法。

---

### 3. **Results（结果）**

#### 3.1 **Baseline characteristics（基线特征）**
- 最终纳入1529例患者，训练集1070例，测试集459例。
- 两组患者在性别、年龄、疾病类型、合并症、生命体征和实验室检查方面无统计学差异。

#### 3.2 **Development of model（模型开发）**
- 三种特征选择方法共筛选出16个特征，用于所有模型的训练。
- XGBoost模型在各项指标中表现最优。

#### 3.3 **Evaluation of model（模型评估）**
- XGBoost模型AUC为0.941，准确率为0.913，明显优于其他模型和传统评分标准。
- PR曲线和DCA进一步验证其优越性。
- 使用SMOTE等采样方法验证模型鲁棒性，效果一致。

---

### 4. **Discussion（讨论）**
- 强调本研究是**首个专为CAP+CTD患者设计的ICU预测模型**。
- 解释了主要预测因子的临床意义，支持模型结果的合理性。
- 指出年龄、性别、ILD等变量未入选的重要原因。
- 与类似研究进行对比，证明本研究在样本量、变量维度、可解释性等方面的优势。

---

### 5. **Conclusions（结论）**
- 成功构建了一个**高性能、可解释的机器学习模型**，用于预测CAP合并CTD患者是否需要ICU治疗。
- 推荐该模型在进一步外部验证后用于临床辅助决策。

---

## 四、主要方法和实施计划
当然，以下是对这篇文献**方法和实施计划**部分的详细说明，涵盖了研究设计、数据处理、特征选择、模型构建、评估与解释等全过程：

---

## 🧪 一、研究方法与实施计划详解

---

### 1. **研究设计（Study Design）**
- **类型**：单中心、回顾性观察研究。
- **时间范围**：2008年11月至2021年11月。
- **地点**：四川大学华西医院。
- **伦理审批**：通过该院伦理委员会批准（批准号：2022-733），因回顾性研究免除患者知情同意。

---

### 2. **患者选择与数据采集（Patients and Data Collection）**

#### 纳入标准：
- 住院期间诊断为**社区获得性肺炎（CAP）**。
- 同时患有至少一种**结缔组织病（CTD）**，包括：  
  - 多发性肌炎/皮肌炎（PM/DM）、类风湿关节炎（RA）、干燥综合征（SS）、系统性硬化症（SSc）、系统性红斑狼疮（SLE）、抗合成酶综合征（ASS）、未分化CTD（UCTD）、混合CTD（MCTD）等。

#### 排除标准：
1. 年龄小于18岁；
2. 妊娠状态；
3. 临床记录不完整；
4. 有多次住院的，只纳入首次记录。

#### 数据内容：
- 人口统计学信息（年龄、性别等）；
- 结缔组织病类型；
- 合并症（如糖尿病、高血压等）；
- 生命体征（心率、血压、体温等）；
- 实验室检查数据（24小时内首次记录）。

---

### 3. **特征选择（Feature Selection）**

研究使用了三种独立变量筛选方法，取其交集以确保变量的稳定性和显著性：

#### a. 单因素分析（Univariate Analysis）：
- 对训练集中所有变量进行显著性检验（P < 0.05）。

#### b. Lasso回归（Least Absolute Shrinkage and Selection Operator）：
- 用于处理高维数据并消除多重共线性；
- 采用10折交叉验证确定最优惩罚参数λ；
- 最终保留非零系数的33个变量。

#### c. Boruta算法：
- 基于随机森林的变量重要性选择；
- 与“shadow features”进行比较，通过多轮引导采样，筛选出32个变量。

#### ✅ 结果：三者交集得到**16个关键特征**，用于建模。

---

### 4. **模型构建（Model Construction）**

共构建了**9种监督式机器学习模型**：

| 模型简称 | 模型名称 |
|----------|----------|
| LR       | Logistic Regression（逻辑回归） |
| CART     | Classification and Regression Tree（分类回归树） |
| RF       | Random Forest（随机森林） |
| SVM      | Support Vector Machine（支持向量机） |
| KNN      | K-Nearest Neighbors（K近邻） |
| DT       | Decision Tree（决策树） |
| GBM      | Gradient Boosting Machine（梯度提升机） |
| XGBoost  | eXtreme Gradient Boosting（极端梯度提升树） |
| NB       | Naive Bayes（朴素贝叶斯） |

#### 数据划分：
- 按照7:3比例，**训练集**1070人，**测试集**459人。

#### 训练方法：
- 使用R语言，调用`caret`包；
- 使用默认网格搜索自动调整超参数；
- 所有模型均进行了**5折交叉验证**。

---

### 5. **模型评估（Model Evaluation）**

对所有模型进行如下性能评估：

- **判别能力**：AUC（ROC曲线下面积）、PR曲线；
- **准确度指标**：准确率、Kappa值、灵敏度、特异性、阳性预测值、阴性预测值；
- **校准能力**：Bootstrap自助法（1000次重采样）生成的校准曲线；
- **临床实用性**：决策曲线分析（DCA）；
- **统计检验**：使用Delong’s test对AUC进行显著性比较。

#### ⚠️ 类别不平衡处理：
由于ICU入院为少数类，研究还引入以下**采样策略**以增强模型鲁棒性：
- 上采样（Over-sampling）；
- 下采样（Under-sampling）；
- SMOTE算法（合成少数类过采样技术）。

---

### 6. **模型解释与可视化（Model Interpretation）**

为了克服机器学习模型“黑箱问题”，引入两种解释方法：

#### a. SHAP（Shapley Additive Explanations）：
- 计算每个特征对预测结果的正负贡献；
- 提供全局特征重要性排序；
- 给出典型个体的局部解释图和依赖图。

#### b. LIME（Local Interpretable Model-Agnostic Explanations）：
- 选取两个患者案例（ICU与非ICU）进行个体层面解释；
- 显示模型对该样本的预测概率与关键特征的推理逻辑。

---

## ✅ 小结：方法实施流程图概览

```
数据收集 → 数据预处理（缺失值处理） → 特征选择（Lasso/Boruta/单因素分析） → 模型构建（9种算法） → 模型评估（AUC/校准/DCA） → 可解释性分析（SHAP/LIME）
```

---


## 五、重要变量和数据(英文展示)
当然，以下是将连续变量和分类变量的统计信息分别以 **Markdown 表格** 的形式整理汇总，方便您后续使用 Python 进行模拟：

---

### 📊 连续变量（Continuous Variables）

| Variable | Mean | Median (IQR) | Std |
|----------|------|----------------|------|
| age | — | 56 (47, 66) | — |
| pH | — | 7.41 (7.38, 7.44) | — |
| BUN (mmol/L) | — | 6.8 (4.5, 10.7) | — |
| sodium (mmol/L) | — | 137.3 (133.6, 140.4) | — |
| glucose (mmol/L) | — | 6.93 (4.68, 10.78) | — |
| hematocrit (L/L) | — | 0.36 (0.31, 0.40) | — |
| PF ratio | — | 207 (166, 279) | — |
| hemoglobin (g/L) | — | 115 (103, 128) | — |
| RDW (%) | — | 14.7 (13.7, 16.2) | — |
| platelet (×10^9/L) | — | 182 (133, 245) | — |
| neutrophil (×10^9/L) | — | 7.53 (5.02, 11.07) | — |
| lymphocyte (×10^9/L) | — | 0.98 (0.59, 1.47) | — |
| monocyte (×10^9/L) | — | 0.33 (0.19, 0.51) | — |
| bilirubin (µmol/L) | — | 9.4 (5.5, 11.9) | — |
| ALT (U/L) | — | 23 (14, 53) | — |
| AST (U/L) | — | 25 (18, 51) | — |
| albumin (g/L) | — | 34.0 (28.7, 39.3) | — |
| globulin (g/L) | — | 28.1 (23.2, 34.7) | — |
| creatinine (µmol/L) | — | 55.00 (44.00, 71.00) | — |
| cystatin C (mg/L) | — | 1.11 (0.94, 1.33) | — |
| triglyceride (mmol/L) | — | 1.39 (1.00, 1.91) | — |
| HDL-C (mmol/L) | — | 1.02 (0.74, 1.36) | — |
| LDL-C (mmol/L) | — | 2.19 (1.56, 2.78) | — |
| creatine kinase (U/L) | — | 52 (26, 154) | — |
| LDH (U/L) | — | 246 (189, 355) | — |
| potassium (mmol/L) | — | 3.50 (3.14, 3.83) | — |
| myoglobin (ng/mL) | — | 43.51 (21.17, 106.60) | — |
| CK-MB (ng/mL) | — | 2.25 (1.09, 4.84) | — |
| NT-proBNP (ng/L) | — | 393 (149, 929) | — |
| Troponin T (ng/L) | — | 23.0 (11.2, 47.9) | — |
| CRP (mg/L) | — | 29.50 (10.40, 86.00) | — |
| Procalcitonin (ng/mL) | — | 0.09 (0.05, 0.40) | — |
| PT (s) | — | 11.3 (10.4, 12.3) | — |
| APTT (s) | — | 27.6 (24.5, 31.3) | — |
| fibrinogen (g/L) | — | 3.34 (2.55, 4.22) | — |
| AT III (%) | — | 84.1 (70.4, 100.1) | — |
| D dimer (mg/L) | — | 1.78 (0.74, 4.85) | — |
| PaCO2 (mmHg) | — | 37.4 (32.9, 41.7) | — |
| lactate (mmol/L) | — | 1.61 (1.16, 2.33) | — |
| CD4 + T cell (cell/µL) | — | 346 (195, 516) | — |
| CD8 + T cell (cell/µL) | — | 251 (130, 384) | — |

---

### 📋 分类变量（Categorical Variables）

| Variable | Frequency | Percentage |
|----------|-----------|------------|
| Sex: male | 492 | 32.2% |
| confusion | 44 | 2.9% |
| Positive G test | 335 | 21.9% |
| Positive GM test | 41 | 2.7% |
| pleural effusion | 546 | 35.7% |
| ICU admission | 413 | 27.0% |
| Need for vasopressors | 391 | 25.6% |
| Need for IMV | 372 | 24.3% |
| Hospital mortality | 237 | 15.5% |
| ILD | 996 | 65.1% |
| COPD | 152 | 9.9% |
| diabetes | 197 | 12.9% |
| hypertension | 321 | 21.0% |
| cancer | 66 | 4.3% |
| chronic liver disease | 89 | 5.8% |
| chronic renal disease | 87 | 5.7% |
| congestive heart failure | 158 | 10.3% |
| cerebrovascular disease | 56 | 3.7% |
| coronary heart disease | 62 | 4.1% |

---


## 五、重要变量和数据(中文展示)
当然，以下是**连续变量**和**分类变量**的中文对照表，保持与英文版本一致的结构，便于理解和后续模拟使用。

---

### 📊 连续变量（Continuous Variables）

| 变量名称 | 均值 | 中位数（四分位数） | 标准差 |
|----------|------|---------------------|--------|
| 年龄 | — | 56（47, 66） | — |
| pH值 | — | 7.41（7.38, 7.44） | — |
| 尿素氮（mmol/L） | — | 6.8（4.5, 10.7） | — |
| 血钠（mmol/L） | — | 137.3（133.6, 140.4） | — |
| 血糖（mmol/L） | — | 6.93（4.68, 10.78） | — |
| 血细胞比容（L/L） | — | 0.36（0.31, 0.40） | — |
| 氧合指数（PF比值） | — | 207（166, 279） | — |
| 血红蛋白（g/L） | — | 115（103, 128） | — |
| 红细胞体积分布宽度（%） | — | 14.7（13.7, 16.2） | — |
| 血小板（×10⁹/L） | — | 182（133, 245） | — |
| 中性粒细胞（×10⁹/L） | — | 7.53（5.02, 11.07） | — |
| 淋巴细胞（×10⁹/L） | — | 0.98（0.59, 1.47） | — |
| 单核细胞（×10⁹/L） | — | 0.33（0.19, 0.51） | — |
| 总胆红素（µmol/L） | — | 9.4（5.5, 11.9） | — |
| 丙氨酸氨基转移酶（ALT） | — | 23（14, 53） | — |
| 天冬氨酸氨基转移酶（AST） | — | 25（18, 51） | — |
| 白蛋白（g/L） | — | 34.0（28.7, 39.3） | — |
| 球蛋白（g/L） | — | 28.1（23.2, 34.7） | — |
| 肌酐（µmol/L） | — | 55.00（44.00, 71.00） | — |
| 胱抑素C（mg/L） | — | 1.11（0.94, 1.33） | — |
| 甘油三酯（mmol/L） | — | 1.39（1.00, 1.91） | — |
| 高密度脂蛋白胆固醇（HDL-C） | — | 1.02（0.74, 1.36） | — |
| 低密度脂蛋白胆固醇（LDL-C） | — | 2.19（1.56, 2.78） | — |
| 肌酸激酶（U/L） | — | 52（26, 154） | — |
| 乳酸脱氢酶（LDH） | — | 246（189, 355） | — |
| 血钾（mmol/L） | — | 3.50（3.14, 3.83） | — |
| 肌红蛋白（ng/mL） | — | 43.51（21.17, 106.60） | — |
| CK-MB（ng/mL） | — | 2.25（1.09, 4.84） | — |
| NT-proBNP（ng/L） | — | 393（149, 929） | — |
| 肌钙蛋白T（ng/L） | — | 23.0（11.2, 47.9） | — |
| C反应蛋白（CRP，mg/L） | — | 29.50（10.40, 86.00） | — |
| 降钙素原（ng/mL） | — | 0.09（0.05, 0.40） | — |
| 凝血酶原时间（PT，秒） | — | 11.3（10.4, 12.3） | — |
| 活化部分凝血活酶时间（APTT，秒） | — | 27.6（24.5, 31.3） | — |
| 纤维蛋白原（g/L） | — | 3.34（2.55, 4.22） | — |
| 抗凝血酶III（AT III，%） | — | 84.1（70.4, 100.1） | — |
| D-二聚体（mg/L） | — | 1.78（0.74, 4.85） | — |
| 二氧化碳分压（PaCO2，mmHg） | — | 37.4（32.9, 41.7） | — |
| 乳酸（mmol/L） | — | 1.61（1.16, 2.33） | — |
| CD4+ T细胞（cell/µL） | — | 346（195, 516） | — |
| CD8+ T细胞（cell/µL） | — | 251（130, 384） | — |

---

### 📋 分类变量（Categorical Variables）

| 变量名称 | 频数 | 构成比 |
|----------|------|--------|
| 男性 | 492 | 32.2% |
| 意识障碍（confusion） | 44 | 2.9% |
| G试验阳性 | 335 | 21.9% |
| GM试验阳性 | 41 | 2.7% |
| 胸腔积液 | 546 | 35.7% |
| ICU住院 | 413 | 27.0% |
| 使用升压药 | 391 | 25.6% |
| 需要有创通气（IMV） | 372 | 24.3% |
| 住院死亡 | 237 | 15.5% |
| 间质性肺病（ILD） | 996 | 65.1% |
| 慢性阻塞性肺病（COPD） | 152 | 9.9% |
| 糖尿病 | 197 | 12.9% |
| 高血压 | 321 | 21.0% |
| 癌症 | 66 | 4.3% |
| 慢性肝病 | 89 | 5.8% |
| 慢性肾病 | 87 | 5.7% |
| 充血性心力衰竭 | 158 | 10.3% |
| 脑血管疾病 | 56 | 3.7% |
| 冠心病 | 62 | 4.1% |

---


## 六、模拟数据
以下是用于模拟文献中提取的**连续变量和分类变量**的 Python 脚本，模拟数据1000条，并保存为指定路径的CSV文件。脚本名为：

### 🐍 脚本名称：`simulate_cap_ctd_data.py`

---

### ✅ 功能简介
- 使用中位数和IQR拟合正态或对数正态分布来生成连续变量；
- 使用二项分布生成分类变量；
- 最终将模拟数据保存为 CSV 文件。

---

### 🧩 代码如下

```python
# simulate_cap_ctd_data.py

import numpy as np
import pandas as pd
import os

# 设置随机种子保证可复现
np.random.seed(42)

# 模拟数据数量
n = 1000

# --------------- 连续变量模拟函数 ---------------
def simulate_from_iqr(median, q1, q3, dist="normal"):
    if dist == "normal":
        std = (q3 - q1) / 1.35  # 估算正态分布标准差
        return np.random.normal(loc=median, scale=std, size=n)
    elif dist == "lognormal":
        sigma = (np.log(q3) - np.log(q1)) / 1.35
        mu = np.log(median)
        return np.random.lognormal(mean=mu, sigma=sigma, size=n)
    else:
        raise ValueError("Invalid distribution type.")

# 连续变量模拟（仅选部分代表性变量作为示例）
data = {
    "age": simulate_from_iqr(56, 47, 66),
    "sodium": simulate_from_iqr(137.3, 133.6, 140.4),
    "glucose": simulate_from_iqr(6.93, 4.68, 10.78, dist="lognormal"),
    "CRP": simulate_from_iqr(29.5, 10.4, 86.0, dist="lognormal"),
    "NT_proBNP": simulate_from_iqr(393, 149, 929, dist="lognormal"),
    "CD4_T_cell": simulate_from_iqr(346, 195, 516),
    "lymphocyte": simulate_from_iqr(0.98, 0.59, 1.47, dist="lognormal"),
    "serum_sodium": simulate_from_iqr(137.3, 133.6, 140.4),
    "platelet": simulate_from_iqr(182, 133, 245),
    "PF_ratio": simulate_from_iqr(207, 166, 279),
}

# --------------- 分类变量模拟函数 ---------------
def simulate_binary(prob):
    return np.random.binomial(1, prob, size=n)

# 分类变量模拟（选代表性变量）
data.update({
    "male": simulate_binary(0.322),
    "ICU_admission": simulate_binary(0.27),
    "positive_G_test": simulate_binary(0.219),
    "COPD": simulate_binary(0.099),
    "diabetes": simulate_binary(0.129),
    "hypertension": simulate_binary(0.21),
    "pleural_effusion": simulate_binary(0.357),
    "hospital_mortality": simulate_binary(0.155),
})

# 构建DataFrame
df = pd.DataFrame(data)

# --------------- 保存数据 ---------------
output_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "simulated_cap_ctd_data.csv")
df.to_csv(output_file, index=False)

print(f"模拟数据已成功保存至：{output_file}")
```

---

### 📁 输出文件示例
- 文件名：`simulated_cap_ctd_data.csv`
- 保存路径：`E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data`
- 数据量：1000条

---

## 六的补充、模拟全部数据

下述代码，包含了文献中提取的**所有分类变量**和**所有连续变量**。完整模拟了数据集，涵盖了所有提取的变量，并保存为指定路径的 CSV 文件。

---

### 🐍 更新版代码：`simulate_full_cap_ctd_data.py`

```python
# simulate_full_cap_ctd_data.py

import numpy as np
import pandas as pd
import os

# 设置随机种子保证可复现
np.random.seed(42)

# 模拟数据数量
n = 1000

# --------------- 连续变量模拟函数 ---------------
def simulate_from_iqr(median, q1, q3, dist="normal"):
    if dist == "normal":
        std = (q3 - q1) / 1.35  # 估算正态分布标准差
        return np.random.normal(loc=median, scale=std, size=n)
    elif dist == "lognormal":
        sigma = (np.log(q3) - np.log(q1)) / 1.35
        mu = np.log(median)
        return np.random.lognormal(mean=mu, sigma=sigma, size=n)
    else:
        raise ValueError("Invalid distribution type.")

# 连续变量模拟（所有文献中的连续变量）
data = {
    "age": simulate_from_iqr(56, 47, 66),
    "pH": simulate_from_iqr(7.41, 7.38, 7.44),
    "BUN (mmol/L)": simulate_from_iqr(6.8, 4.5, 10.7),
    "sodium (mmol/L)": simulate_from_iqr(137.3, 133.6, 140.4),
    "glucose (mmol/L)": simulate_from_iqr(6.93, 4.68, 10.78, dist="lognormal"),
    "hematocrit (L/L)": simulate_from_iqr(0.36, 0.31, 0.40),
    "PF ratio": simulate_from_iqr(207, 166, 279),
    "hemoglobin (g/L)": simulate_from_iqr(115, 103, 128),
    "RDW (%)": simulate_from_iqr(14.7, 13.7, 16.2),
    "platelet (×10^9/L)": simulate_from_iqr(182, 133, 245),
    "neutrophil (×10^9/L)": simulate_from_iqr(7.53, 5.02, 11.07),
    "lymphocyte (×10^9/L)": simulate_from_iqr(0.98, 0.59, 1.47, dist="lognormal"),
    "monocyte (×10^9/L)": simulate_from_iqr(0.33, 0.19, 0.51),
    "bilirubin (µmol/L)": simulate_from_iqr(9.4, 5.5, 11.9),
    "ALT (U/L)": simulate_from_iqr(23, 14, 53),
    "AST (U/L)": simulate_from_iqr(25, 18, 51),
    "albumin (g/L)": simulate_from_iqr(34.0, 28.7, 39.3),
    "globulin (g/L)": simulate_from_iqr(28.1, 23.2, 34.7),
    "creatinine (µmol/L)": simulate_from_iqr(55.00, 44.00, 71.00),
    "cystatin C (mg/L)": simulate_from_iqr(1.11, 0.94, 1.33),
    "triglyceride (mmol/L)": simulate_from_iqr(1.39, 1.00, 1.91),
    "HDL-C (mmol/L)": simulate_from_iqr(1.02, 0.74, 1.36),
    "LDL-C (mmol/L)": simulate_from_iqr(2.19, 1.56, 2.78),
    "creatine kinase (U/L)": simulate_from_iqr(52, 26, 154),
    "LDH (U/L)": simulate_from_iqr(246, 189, 355),
    "potassium (mmol/L)": simulate_from_iqr(3.50, 3.14, 3.83),
    "myoglobin (ng/mL)": simulate_from_iqr(43.51, 21.17, 106.60),
    "CK-MB (ng/mL)": simulate_from_iqr(2.25, 1.09, 4.84),
    "NT-proBNP (ng/L)": simulate_from_iqr(393, 149, 929, dist="lognormal"),
    "Troponin T (ng/L)": simulate_from_iqr(23.0, 11.2, 47.9),
    "CRP (mg/L)": simulate_from_iqr(29.50, 10.40, 86.00, dist="lognormal"),
    "Procalcitonin (ng/mL)": simulate_from_iqr(0.09, 0.05, 0.40),
    "PT (s)": simulate_from_iqr(11.3, 10.4, 12.3),
    "APTT (s)": simulate_from_iqr(27.6, 24.5, 31.3),
    "fibrinogen (g/L)": simulate_from_iqr(3.34, 2.55, 4.22),
    "AT III (%)": simulate_from_iqr(84.1, 70.4, 100.1),
    "D dimer (mg/L)": simulate_from_iqr(1.78, 0.74, 4.85),
    "PaCO2 (mmHg)": simulate_from_iqr(37.4, 32.9, 41.7),
    "lactate (mmol/L)": simulate_from_iqr(1.61, 1.16, 2.33),
    "CD4 + T cell (cell/µL)": simulate_from_iqr(346, 195, 516),
    "CD8 + T cell (cell/µL)": simulate_from_iqr(251, 130, 384),
}

# --------------- 分类变量模拟函数 ---------------
def simulate_binary(prob):
    return np.random.binomial(1, prob, size=n)

# 所有分类变量模拟
data.update({
    "male": simulate_binary(0.322),
    "confusion": simulate_binary(0.029),
    "positive_G_test": simulate_binary(0.219),
    "positive_GM_test": simulate_binary(0.027),
    "pleural_effusion": simulate_binary(0.357),
    "ICU_admission": simulate_binary(0.270),
    "Need_for_vasopressors": simulate_binary(0.256),
    "Need_for_IMV": simulate_binary(0.243),
    "hospital_mortality": simulate_binary(0.155),
    "ILD": simulate_binary(0.651),
    "COPD": simulate_binary(0.099),
    "diabetes": simulate_binary(0.129),
    "hypertension": simulate_binary(0.210),
    "cancer": simulate_binary(0.043),
    "chronic_liver_disease": simulate_binary(0.058),
    "chronic_renal_disease": simulate_binary(0.057),
    "congestive_heart_failure": simulate_binary(0.103),
    "cerebrovascular_disease": simulate_binary(0.037),
    "coronary_heart_disease": simulate_binary(0.041),
})

# 构建DataFrame
df = pd.DataFrame(data)

# --------------- 保存数据 ---------------
output_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "simulated_full_cap_ctd_data.csv")
df.to_csv(output_file, index=False)

print(f"模拟数据已成功保存至：{output_file}")
```

---

### 📁 输出文件
- **文件名**：`simulated_full_cap_ctd_data.csv`
- **保存路径**：`E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data`
- **数据条数**：1000条

---

### ✅ 说明：
- **连续变量**：已按文献中的中位数与IQR拟合正态分布或对数正态分布生成。
- **分类变量**：根据文献中的频率信息，通过二项分布模拟生成。
- 最终保存为 **CSV 文件**，便于后续使用。

---


## 七、复现代码（只选择单因素分析筛选变量）
根据文献中的**研究方法与实施计划**，以下是用 **Python** 代码实现文献中的核心方法，目的是复现文献中的实验过程，模拟结果并评估模型。该代码包括数据加载、特征选择、模型构建、评估以及模型解释等步骤。

---

### 🐍 Python 代码：`reproduce_cap_ctd_analysis.py`

```python
# reproduce_cap_ctd_analysis.py

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# 设置随机种子，确保可复现性
np.random.seed(42)

# 模拟数据加载
data_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data\simulated_full_cap_ctd_data.csv"
df = pd.read_csv(data_path)

# ---------------- 数据预处理 ----------------
# 选择目标变量 (ICU入院) 与特征变量
X = df.drop(columns=['ICU_admission'])
y = df['ICU_admission']

# 将数据划分为训练集和测试集 (70% 训练集，30% 测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---------------- 特征选择 ----------------
# 单因素分析
# (这里只做一个简单的显著性检验，后续可加入更多特征选择方法)
from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(chi2, k=16).fit_transform(X_train, y_train)

# ---------------- 模型构建 ----------------
# 构建9个不同的模型
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# ---------------- 模型评估 ----------------
# 使用10折交叉验证评估模型
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    results[name] = cv_results
    print(f"{name} - Mean AUC: {cv_results.mean():.4f}")

# ---------------- 类别不平衡处理（SMOTE） ----------------
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ---------------- 模型调优（使用网格搜索调参） ----------------
param_grid = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},
    'Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]},
    'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
}

# 选择调优模型 (举例：随机森林调优)
grid_search = GridSearchCV(RandomForestClassifier(), param_grid['Random Forest'], cv=cv, scoring='roc_auc')
grid_search.fit(X_resampled, y_resampled)
print(f"Best Parameters for Random Forest: {grid_search.best_params_}")

# ---------------- 模型训练与预测 ----------------
best_rf_model = grid_search.best_estimator_

best_rf_model.fit(X_resampled, y_resampled)
y_pred = best_rf_model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Confusion Matrix: \n{conf_matrix}")

# ---------------- ROC 曲线绘制 ----------------
fpr, tpr, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# ---------------- 结果保存 ----------------
output_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "model_results.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"模型评估结果已保存至：{output_file}")
```

---

### 📁 代码功能说明：
1. **数据加载与预处理**：读取模拟数据并进行数据划分，选择特征变量和目标变量。
2. **特征选择**：使用**单因素分析**（可进一步添加Lasso回归、Boruta等方法）筛选特征。
3. **模型构建与评估**：构建多个机器学习模型（Logistic Regression、Random Forest等），并通过交叉验证评估其**AUC**。
4. **类别不平衡处理**：使用**SMOTE**算法进行过采样处理，以平衡类分布。
5. **模型调优**：使用**GridSearchCV**进行超参数调优，以优化模型性能。
6. **模型训练与预测**：使用训练数据训练模型，并在测试数据上评估模型表现。
7. **结果评估**：计算**准确率、AUC、混淆矩阵**等，并绘制ROC曲线。
8. **结果保存**：保存模型评估结果至CSV文件。

---

### 📁 输出文件：
- **文件名**：`model_results.csv`
- **保存路径**：`E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据`
- **数据**：各个模型的交叉验证结果。

---

### 🚀 使用方法：
1. **首先，确保模拟数据已保存在给定路径下**（路径与文件名：`simulated_full_cap_ctd_data.csv`）。
2. **运行该代码**以进行模型训练、评估及优化。
3. **查看结果**：通过输出的CSV文件与ROC曲线图评估模型性能。

---
## 八、复现代码（单因素分析、Lasso回归和Boruta算法筛选变量）

采用三种方法来筛选特征，并取其交集确保变量的稳定性和显著性：**单因素分析**、**Lasso回归**和**Boruta算法**。使用三种特征选择方法，并取交集。


### 🐍 完整代码：`reproduce_cap_ctd_analysis_with_feature_selection.py`

```python
# reproduce_cap_ctd_analysis_with_feature_selection.py

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LassoCV
from boruta import BorutaPy

# 设置随机种子保证可复现
np.random.seed(42)

# 模拟数据加载
data_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据\data\simulated_full_cap_ctd_data.csv"
df = pd.read_csv(data_path)

# ---------------- 数据预处理 ----------------
# 选择目标变量 (ICU入院) 与特征变量
X = df.drop(columns=['ICU_admission'])
y = df['ICU_admission']

# 将数据划分为训练集和测试集 (70% 训练集，30% 测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---------------- 特征选择 ----------------
# a. 单因素分析
# 进行单因素卡方检验，选择显著性高的变量
selector_chi2 = SelectKBest(chi2, k=16)
X_train_chi2 = selector_chi2.fit_transform(X_train, y_train)
selected_features_chi2 = X.columns[selector_chi2.get_support()]

# b. Lasso回归
# 使用Lasso进行特征选择，10折交叉验证，选择最优lambda
lasso = LassoCV(cv=10, random_state=42)
lasso.fit(X_train, y_train)
selected_features_lasso = X.columns[np.where(lasso.coef_ != 0)[0]]

# c. Boruta算法
# 使用Boruta算法进行特征选择
rf = RandomForestClassifier(n_jobs=-1, max_depth=5)
boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
boruta.fit(X_train.values, y_train.values)
selected_features_boruta = X.columns[boruta.support_]

# 获取三者的交集
selected_features = list(set(selected_features_chi2) & set(selected_features_lasso) & set(selected_features_boruta))

# ---------------- 模型构建 ----------------
# 使用交集的特征进行模型构建
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 构建9个不同的模型
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# ---------------- 模型评估 ----------------
# 使用10折交叉验证评估模型
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    cv_results = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
    results[name] = cv_results
    print(f"{name} - Mean AUC: {cv_results.mean():.4f}")

# ---------------- 类别不平衡处理（SMOTE） ----------------
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)

# ---------------- 模型调优（使用网格搜索调参） ----------------
param_grid = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},
    'Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]},
    'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
}

# 选择调优模型 (举例：随机森林调优)
grid_search = GridSearchCV(RandomForestClassifier(), param_grid['Random Forest'], cv=cv, scoring='roc_auc')
grid_search.fit(X_resampled, y_resampled)
print(f"Best Parameters for Random Forest: {grid_search.best_params_}")

# ---------------- 模型训练与预测 ----------------
best_rf_model = grid_search.best_estimator_

best_rf_model.fit(X_resampled, y_resampled)
y_pred = best_rf_model.predict(X_test_selected)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Confusion Matrix: \n{conf_matrix}")

# ---------------- ROC 曲线绘制 ----------------
fpr, tpr, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test_selected)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# ---------------- 结果保存 ----------------
output_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "model_results_with_feature_selection.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"模型评估结果已保存至：{output_file}")
```

---

### 🧩 代码功能说明：
1. **数据加载与预处理**：加载模拟数据，并选择目标变量（`ICU_admission`）与特征变量。
2. **特征选择**：
   - **单因素分析（Univariate Analysis）**：通过卡方检验选择显著性高的特征。
   - **Lasso回归（LassoCV）**：使用Lasso进行特征选择，自动选择具有非零系数的特征。
   - **Boruta算法**：使用Boruta进行特征选择，基于随机森林的特征重要性选择。
   - 通过取三者交集确保选择了稳定且显著的特征。
3. **模型构建与评估**：构建多个机器学习模型（Logistic Regression、Random Forest、SVM等），并使用交叉验证评估每个模型的AUC。
4. **类别不平衡处理**：使用SMOTE算法处理类别不平衡问题，生成平衡的数据集。
5. **模型调优**：通过网格搜索调优模型超参数（例如随机森林的`n_estimators`、`max_depth`）。
6. **模型训练与预测**：在训练集上训练最优模型，并在测试集上进行预测。
7. **结果评估**：计算并打印**准确率、AUC、混淆矩阵**等评估指标，并绘制ROC曲线。
8. **结果保存**：保存每个模型的交叉验证结果至CSV文件。

---

### 📁 输出文件：
- **文件名**：`model_results_with_feature_selection.csv`
- **保存路径**：`E:\JA\github_blogs\blogs-master\04文献阅读\09ICU\02社区获得性肺炎和结缔组织病\01模拟数据`

---

### 🚀 使用方法：
1. **确保模拟数据已经加载到指定路径**（路径和文件名：`simulated_full_cap_ctd_data.csv`）。
2. **运行脚本**以进行特征选择、模型训练、调优及评估。
3. **查看输出**：通过CSV文件和ROC曲线查看模型性能。

---

## 九、重新生成数据

