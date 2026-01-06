
# MIMIC-IV ICU Teaching Project

> 面向临床与科研新人的 MIMIC-IV 教学与可视化项目  
> 当前已完成：**肾内科 · AKI（Acute Kidney Injury）单病种模块 v0.1**

---

## 1. 项目简介

本项目基于公开的重症医学数据库 **MIMIC-IV**，  
按照「**科室 → 疾病 → 描述性分析 → 统计建模 → 机器学习**」的思路，
逐步构建一个面向教学的可视化平台。

项目特点：

- 面向临床与科研新人，强调 **方法学与可重复性**；
- 严格遵守 MIMIC 数据使用协议，仅展示 **汇总统计与可视化结果**；
- 后端使用 Python + PostgreSQL，前端使用 HTML/JavaScript + ECharts（后续可迁移至 React/Next.js）。

> ⚠️ **重要声明：**  
> 本项目仅用于科研与教学目的，不能作为临床诊疗或个体决策依据。

---

## 2. 当前进度

### ✅ 已完成：AKI ICU 描述性分析模块（v0.1）

- 队列定义文档：`docs/aki/01_cohort_definition.md`
- 研究设计与背景：`docs/aki/00_study_design.md`
- 变量字典：`docs/aki/02_variable_dictionary.md`
- 数据库视图：
  - `aki_cohort`：AKI ICU 队列表（人口学 + ICU 信息 + 结局）
  - `aki_labs_firstday`：AKI 患者 ICU 首日实验室指标汇总
- Python 后端：
  - 从数据库读取并合并 AKI 数据
  - 生成描述性统计与可视化所需 JSON/CSV 文件
- 前端页面（原型）：
  - `frontend/aki.html`：展示 AKI 的：
    - 样本量与概览卡片
    - 年龄分布
    - 性别分布
    - ICU/住院死亡率

---

## 3. 仓库结构（简要）

```text
.
├── backend/
│   ├── config.py           # 数据库配置（通过环境变量）
│   ├── paths.py            # 输出路径管理
│   └── aki/
│       ├── data_loader.py      # 读取 aki_cohort & aki_labs_firstday
│       ├── descriptive.py      # 描述性分析函数
│       ├── export_results.py   # 导出 JSON/CSV
│       └── run_aki_pipeline.py # 一键跑通 AKI 分析
├── docs/
│   └── aki/
│       ├── 00_study_design.md       # 研究设计
│       ├── 01_cohort_definition.md  # 队列定义
│       └── 02_variable_dictionary.md# 变量字典
├── frontend/
│   ├── aki.html              # AKI 可视化页面（静态原型）
│   └── data/
│       └── aki/
│           ├── aki_basic_stats.json
│           ├── aki_age_distribution.json
│           ├── aki_gender_distribution.json
│           └── aki_mortality.json
└── outputs/
    └── aki/                  # backend 生成的所有输出文件
```

---

## 4. 环境与运行方式

### 4.1 数据库准备

1. 已在 PostgreSQL 中加载 MIMIC-IV（hospital + icu schema）；
2. 已执行 SQL 脚本（示例路径）：

   * `backend/sql/aki_cohort.sql`
   * `backend/sql/aki_labs_firstday.sql`

### 4.2 Python 环境

```bash
# 创建环境（示例）
python -m venv venv
source venv/bin/activate  # Windows 使用 venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` 建议至少包括：

```text
pandas
sqlalchemy
psycopg2-binary
```

### 4.3 配置数据库连接

通过环境变量或 `.env`（示例）：

```bash
export MIMIC_DB_HOST=localhost
export MIMIC_DB_PORT=5432
export MIMIC_DB_NAME=mimiciv
export MIMIC_DB_USER=postgres
export MIMIC_DB_PASSWORD=your_password
```

或在 `backend/config.py` 中直接修改默认值。

### 4.4 运行 AKI 分析流程

```bash
# 在项目根目录
python -m backend.aki.run_aki_pipeline
```

运行成功后，将在 `outputs/aki/` 得到若干 JSON/CSV 文件。

---

## 5. 前端 AKI 页面预览

### 5.1 准备数据

将 `outputs/aki/` 中生成的 JSON 文件复制（或软链接）到：

```text
frontend/data/aki/
```

确保至少包含：

* `aki_basic_stats.json`
* `aki_age_distribution.json`
* `aki_gender_distribution.json`
* `aki_mortality.json`

### 5.2 启动简单静态服务器

```bash
cd frontend
python -m http.server 8000
```

然后在浏览器打开：

```text
http://localhost:8000/aki.html
```

即可看到 AKI 模块的初版可视化页面。

---

## 6. 版本与后续计划

当前版本：**v0.1 – AKI 描述性分析 MVP**

后续计划（示意）：

* v0.2：在 AKI 模块中加入基础统计模型（如 logistic 回归）结果展示；
* v0.3：扩展到其它典型 ICU 疾病（如心衰、急性胰腺炎等）；
* v0.4+：引入机器学习模型、共病网络等高级分析。

详细变更记录见：`CHANGELOG.md`。