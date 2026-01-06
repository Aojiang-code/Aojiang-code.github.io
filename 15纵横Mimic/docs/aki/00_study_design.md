# AKI ICU人群的描述性分析（MIMIC-IV）

## 1. 研究背景与目的

急性肾损伤（Acute Kidney Injury, AKI）在重症监护室（ICU）人群中发病率较高，
与短期和长期死亡率、住院时间延长和医疗费用增加密切相关。
然而，对于真实世界ICU人群中AKI患者的人口学特征、实验室指标分布及短期结局，
仍需要基于大规模数据库进行系统性的描述。

**本项目的第一阶段目标：**

- 构建MIMIC-IV数据库中ICU AKI患者的研究队列；
- 描述该人群的人口学特征、主要实验室指标与短期结局（ICU/住院死亡）；
- 为后续统计建模与机器学习分析提供清晰、可重复的数据基础。

## 2. 数据来源

- 数据库：MIMIC-IV（version X.X，按你实际填写）
- 数据时间范围：2008–2019年（MIMIC-IV整体时间范围）
- 数据表：
  - `mimiciv_icu.icustays`：ICU入住信息
  - `mimiciv_hosp.admissions`：住院信息
  - `mimiciv_hosp.patients`：人口学信息（年龄、性别等）
  - `mimiciv_hosp.diagnoses_icd`：诊断信息（ICD-9/ICD-10）
  - `mimiciv_hosp.labevents`：实验室检查结果
  - （后续如需：`mimiciv_icu.inputevents`、`mimiciv_icu.chartevents`等）

## 3. 研究类型与总体设计

- 研究类型：回顾性队列研究（retrospective cohort）
- 研究人群：MIMIC-IV中入住ICU的成人患者
- 暴露定义：有急性肾损伤相关诊断（ICD-9/ICD-10）者
- 主要结局：
  - ICU期间死亡（ICU mortality）
  - 住院期间死亡（in-hospital mortality）
  - ICU停留时间（ICU length of stay）
  - 住院总天数（hospital length of stay）

## 4. 分析思路（第一阶段）

第一阶段仅进行描述性分析，不进行复杂建模，内容包括：

- 描述AKI ICU人群的人口学特征分布（年龄、性别、种族、保险类型等）；
- 描述部分关键实验室指标首日水平的分布（如血肌酐、尿素氮、电解质等）；
- 描述ICU/住院死亡率、ICU时间和住院时间的分布；
- 将上述结果以表格和可视化图形的形式展示在前端网页中。

后续阶段将在此基础上增加：

- 传统统计模型（如logistic回归、Cox回归）；
- 机器学习模型（如树模型、XGBoost等）。
