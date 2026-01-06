
# AKI 模块总览（MIMIC-IV, ICU）

本目录记录了 AKI（Acute Kidney Injury）模块的完整方法学与实现细节。

---

## 1. 文档结构

- `00_study_design.md`  
  概述研究背景、目的、数据来源及整体分析设计（相当于论文的方法部分框架）。

- `01_cohort_definition.md`  
  详细说明 AKI ICU 队列表的 **纳入/排除标准、AKI 定义版本、索引时间点** 等。

- `02_variable_dictionary.md`  
  列出本模块使用的所有变量及其在 MIMIC-IV 中的来源（表名、字段、时间窗等）。

---

## 2. 队列表与变量实现（SQL & Python）

### 2.1 SQL 视图

- `aki_cohort`  
  - 位置：PostgreSQL 数据库  
  - 来源脚本：`backend/sql/aki_cohort.sql`  
  - 内容：  
    - 成人首次 ICU 入住患者（age ≥ 18）  
    - 住院期间存在 AKI 相关 ICD-9/10 诊断（584\*/N17\*）  
    - ICU 住院时间 ≥ 6 小时  
    - 含人口学、ICU 停留时间、住院时间、ICU/住院死亡等信息。

- `aki_labs_firstday`  
  - 位置：PostgreSQL 数据库  
  - 来源脚本：`backend/sql/aki_labs_firstday.sql`  
  - 内容：  
    - 对 `aki_cohort` 中患者提取 ICU 入科后 0–24 小时内的实验室指标  
    - 包括首日 Scr、BUN、电解质、WBC、Hb、血小板等  
    - 每个指标按中位数聚合为一行（每个 stay_id 一行）。

### 2.2 Python 分析流程

- 数据加载：`backend/aki/data_loader.py`  
  - 从数据库读取 `aki_cohort` 与 `aki_labs_firstday`  
  - 按 `subject_id, hadm_id, stay_id` 合并成一个分析数据集。

- 描述性分析：`backend/aki/descriptive.py`  
  - 计算样本量、年龄分布、性别分布、ICU/住院死亡率等  
  - 生成适合前端可视化的统计结果结构。

- 结果导出：`backend/aki/export_results.py`  
  - 将分析结果输出为 JSON/CSV 到 `outputs/aki/`。

- 入口脚本：`backend/aki/run_aki_pipeline.py`  
  - 一键运行上述所有步骤，完成 AKI 描述性分析。

---

## 3. 当前 AKI 模块的功能范围（v0.1）

- ✅ 基于诊断码（ICD-9/10）的 AKI ICU 队列构建；
- ✅ 人口学特征与 ICU/住院结局的描述性统计；
- ✅ 基本可视化数据导出，用于前端页面展示：
  - 总样本量与整体概览；
  - 年龄分布；
  - 性别分布；
  - ICU 与住院死亡率。

暂未包含：

- KDIGO 标准的 AKI 定义；
- 长期结局、再入院等；
- 回归模型或机器学习模型。

---

## 4. 使用注意事项

- 本模块所有结果来源于 MIMIC-IV，且仅用于科研与教学；
- 所有公开展示内容（JSON/CSV/前端可视化）均为 **汇总数据**，不包含可识别的个体信息；
- 如需扩展新的变量或改变 AKI 定义，请：
  1. 先在 `02_variable_dictionary.md` 中补充/修改变量；
  2. 再更新 SQL 与 Python 代码；
  3. 在 `CHANGELOG.md` 中记录版本变更。