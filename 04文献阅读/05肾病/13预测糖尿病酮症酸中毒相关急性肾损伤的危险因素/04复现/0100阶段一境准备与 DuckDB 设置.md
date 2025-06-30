非常好，我们现在重新开始：

---

# ✅ 阶段一：使用 DuckDB 构建 MIMIC-IV 数据环境（推荐使用持久化数据库）

🎯 本阶段目标：

* 使用 DuckDB 高效加载 MIMIC-IV 各表（CSV 文件）；
* 注册为 SQL 视图供后续查询；
* 最终将数据库导出为持久化 `.duckdb` 文件，供后续 Notebook 快速加载。

---

## 🧰 第一步：环境依赖安装（仅一次）

```bash
pip install duckdb pandas
```

---

## 📦 第二步：初始化 DuckDB 引擎（持久化方式）

在您的 Notebook 顶部写入：

```python
import duckdb
import pandas as pd

# 创建或连接持久化 DuckDB 文件（将被保存）
con = duckdb.connect('mimiciv.duckdb')
```

---

## 📁 第三步：注册 MIMIC-IV 各 CSV 表为 SQL 视图

假设您本地的文件结构如下：

```
/project_root/
├── mimiciv/
│   ├── hosp/
│   │   ├── patients.csv
│   │   ├── admissions.csv
│   │   ├── diagnoses_icd.csv
│   │   ├── prescriptions.csv
│   │   └── procedures_icd.csv
│   └── icu/
│       ├── icustays.csv
│       ├── labevents.csv
│       └── chartevents.csv
```

### 🧩 注册表格（建议全复制运行）

```python
base_path = '/your/absolute/path/to/mimiciv/'

# hosp 表
con.execute(f"""
CREATE OR REPLACE VIEW patients AS
SELECT * FROM read_csv_auto('{base_path}hosp/patients.csv');
""")

con.execute(f"""
CREATE OR REPLACE VIEW admissions AS
SELECT * FROM read_csv_auto('{base_path}hosp/admissions.csv');
""")

con.execute(f"""
CREATE OR REPLACE VIEW diagnoses_icd AS
SELECT * FROM read_csv_auto('{base_path}hosp/diagnoses_icd.csv');
""")

con.execute(f"""
CREATE OR REPLACE VIEW prescriptions AS
SELECT * FROM read_csv_auto('{base_path}hosp/prescriptions.csv');
""")

con.execute(f"""
CREATE OR REPLACE VIEW procedures_icd AS
SELECT * FROM read_csv_auto('{base_path}hosp/procedures_icd.csv');
""")

# icu 表
con.execute(f"""
CREATE OR REPLACE VIEW icustays AS
SELECT * FROM read_csv_auto('{base_path}icu/icustays.csv');
""")

con.execute(f"""
CREATE OR REPLACE VIEW labevents AS
SELECT * FROM read_csv_auto('{base_path}icu/labevents.csv');
""")

con.execute(f"""
CREATE OR REPLACE VIEW chartevents AS
SELECT * FROM read_csv_auto('{base_path}icu/chartevents.csv');
""")
```

> 📌 替换 `base_path` 为您的本地路径，如 `/public/home/aojiang/mimiciv/`

---

## ✅ 第四步：检查视图是否创建成功

### 检查视图可用性：

```python
con.execute("SHOW TABLES").df()
```

### 查看某个表的前几行：

```python
con.execute("SELECT * FROM patients LIMIT 5").df()
```

---

## 💾 第五步：将数据库保存为 `.duckdb` 文件（持久化推荐）

这样您后续的阶段二/三 Notebook 不再需要重复读取 CSV 文件：

```python
con.execute("EXPORT DATABASE 'mimiciv.duckdb' (FORMAT 'duckdb')")
```

---

## 📘 阶段一输出成果

* ✅ 一个名为 `mimiciv.duckdb` 的数据库文件
* ✅ 所有必要表（patients, admissions, icustays 等）注册为 SQL 视图
* ✅ 后续阶段只需一行代码即可重连：

```python
con = duckdb.connect('mimiciv.duckdb')
```

---

现在您可以进入阶段二（筛选 DKA ICU 成人 cohort），我已为其准备好 SQL 查询与解释。是否继续？
