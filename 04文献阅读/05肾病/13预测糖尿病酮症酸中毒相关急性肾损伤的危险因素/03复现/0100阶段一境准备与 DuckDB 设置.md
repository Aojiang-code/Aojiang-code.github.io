非常好，我们现在开始进入 DuckDB 版本的：

---

# 📁 阶段一：环境准备与 DuckDB 设置

🎯 目标：使用 DuckDB 高效读取 MIMIC-IV CSV 数据，避免内存溢出，并为后续分析注册视图。

---

## ✅ 第一步：环境依赖安装（仅需一次）

请确保您在 VSCode 的 Jupyter Notebook 环境中执行以下命令：

```bash
pip install duckdb pandas
```

---

## ✅ 第二步：初始化 DuckDB 引擎 & 建立连接

在您的 Notebook 开始处，执行如下代码：

```python
import duckdb
import pandas as pd

# 创建内存中的 DuckDB 实例，也可以保存为文件
con = duckdb.connect(database=':memory:')  # 或使用 'mimiciv.duckdb' 做持久化
```

---

## ✅ 第三步：读取并注册 CSV 表格为视图

此步骤将 MIMIC-IV 所需的 CSV 文件以流式方式注册为 DuckDB 可用的 SQL 视图（无需加载全表到内存）。

### 🎯 建议的文件结构（示意）

```
/your/project/folder/
├── mimiciv/
│   ├── hosp/
│   │   ├── patients.csv
│   │   ├── admissions.csv
│   │   ├── diagnoses_icd.csv
│   │   ├── procedures_icd.csv
│   │   └── prescriptions.csv
│   └── icu/
│       ├── icustays.csv
│       ├── labevents.csv
│       └── chartevents.csv
```

---

### ✅ 示例代码：批量注册 CSV 为 SQL 视图

```python
base_path = '/your/project/folder/mimiciv/'

# hosp 表
con.execute(f"""
CREATE VIEW patients AS
SELECT * FROM read_csv_auto('{base_path}hosp/patients.csv');
""")

con.execute(f"""
CREATE VIEW admissions AS
SELECT * FROM read_csv_auto('{base_path}hosp/admissions.csv');
""")

con.execute(f"""
CREATE VIEW diagnoses_icd AS
SELECT * FROM read_csv_auto('{base_path}hosp/diagnoses_icd.csv');
""")

con.execute(f"""
CREATE VIEW prescriptions AS
SELECT * FROM read_csv_auto('{base_path}hosp/prescriptions.csv');
""")

con.execute(f"""
CREATE VIEW procedures_icd AS
SELECT * FROM read_csv_auto('{base_path}hosp/procedures_icd.csv');
""")

# icu 表
con.execute(f"""
CREATE VIEW icustays AS
SELECT * FROM read_csv_auto('{base_path}icu/icustays.csv');
""")

con.execute(f"""
CREATE VIEW labevents AS
SELECT * FROM read_csv_auto('{base_path}icu/labevents.csv');
""")

con.execute(f"""
CREATE VIEW chartevents AS
SELECT * FROM read_csv_auto('{base_path}icu/chartevents.csv');
""")
```

📌 每一行语句都将一个 CSV 文件注册为 DuckDB 内部视图（您可以直接使用 SQL 查询它们，无需再读取 DataFrame）

---

## ✅ 第四步：验证数据读取是否成功

您可以运行如下 SQL 查询来快速预览表内容：

```python
# 显示前 5 行 ICU stays 数据
con.execute("SELECT * FROM icustays LIMIT 5").df()
```

或者：

```python
# 查看 patients 中字段
con.execute("DESCRIBE patients").df()
```

---

## 🧪 推荐附加检查：

```python
# 查看各表行数
for tbl in ['patients', 'admissions', 'diagnoses_icd', 'icustays', 'labevents', 'chartevents']:
    print(f"{tbl}: ", con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
```

---

## ✅ 阶段一总结成果：

您现在拥有：

* 🐥 初始化完成的 DuckDB 内存数据库
* 🧾 所有 MIMIC-IV CSV 表被注册为可用 SQL 视图
* 🧪 可以直接在 Python 中运行 SQL 语句

---

接下来是否继续进入：

# 📁 阶段二：筛选 DKA 患者 + 构建 cohort（年龄、性别、ICU stay）？

我将提供一段高效 SQL 查询，用 DuckDB 直接生成 DKA+ICU 成年患者 cohort。准备好了吗？
