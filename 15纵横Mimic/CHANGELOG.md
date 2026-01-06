
# Changelog

所有重要变更都会记录在这里。

格式参考 Keep a Changelog，版本号采用 `major.minor.patch` 形式，
并附加简要说明。

---

## [0.1.0] - 2025-12-07

### Added

- 新增肾内科 AKI（Acute Kidney Injury）模块基础版本：
  - 编写 AKI 研究设计文档：`docs/aki/00_study_design.md`
  - 编写 AKI 队列定义文档：`docs/aki/01_cohort_definition.md`
  - 编写 AKI 变量字典：`docs/aki/02_variable_dictionary.md`
- 新增数据库视图：
  - `aki_cohort`：AKI ICU 队列（成人首次 ICU + AKI ICD 诊断）
  - `aki_labs_firstday`：AKI 患者 ICU 首日实验室指标
- 新增 Python 后端：
  - `backend/aki/data_loader.py`：读取并合并 AKI 数据
  - `backend/aki/descriptive.py`：基础描述性分析函数
  - `backend/aki/export_results.py`：结果导出为 JSON/CSV
  - `backend/aki/run_aki_pipeline.py`：AKI 分析一键运行入口
- 新增前端页面原型：
  - `frontend/aki.html`：展示 AKI 队列概览、年龄分布、性别分布和死亡率。

---

## [Unreleased]

### Planned

- 为 AKI 模块加入回归模型结果展示（如 ICU 死亡的 logistic 回归）；
- 将前端迁移至 React/Next.js，并支持多疾病模块切换；
- 扩展至其它常见 ICU 疾病（心衰、急性胰腺炎等）。