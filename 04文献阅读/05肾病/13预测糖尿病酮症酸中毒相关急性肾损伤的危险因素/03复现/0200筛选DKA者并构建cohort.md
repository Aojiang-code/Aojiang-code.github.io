很好，我们现在进入：

---

# 📁 阶段二：筛选 DKA 患者 + 构建 cohort（年龄、性别、ICU stay）

🎯 目标：使用 DuckDB SQL 查询，筛选出患有糖尿病酮症酸中毒（DKA）的成年 ICU 患者，并构建包含年龄、性别、ICU 入出时间等关键信息的 cohort 表，为后续建模准备。

---

## ✅ 步骤 1：确定 DKA ICD 编码

根据原文与临床常用编码，DKA 主要对应以下 ICD 编码（支持 ICD-9 和 ICD-10）：

```text
-- ICD-9
250.10, 250.11, 250.12
-- ICD-10
E10.10, E11.10, E13.10
```

这些编码在 CSV 中通常以字符串形式存在，并可能带有 `.`。

---

## ✅ 步骤 2：构建 SQL 查询：DKA + ICU + 成人 Cohort

以下 SQL 脚本将在 DuckDB 中执行，并生成一个名为 `dka_cohort` 的临时视图（也可以导出为 Pandas DataFrame）。

```python
query = """
-- 创建 dka_cohort 视图
CREATE OR REPLACE VIEW dka_cohort AS
WITH dka_hadm AS (
    SELECT DISTINCT hadm_id
    FROM diagnoses_icd
    WHERE REPLACE(icd_code, '.', '') IN (
        '25010', '25011', '25012', 'E1010', 'E1110', 'E1310'
    )
),
dka_icu AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
           p.anchor_age AS age, p.gender
    FROM icustays i
    INNER JOIN dka_hadm d ON i.hadm_id = d.hadm_id
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
    WHERE p.anchor_age >= 18
)
SELECT * FROM dka_icu
"""
con.execute(query)
```

---

## ✅ 步骤 3：查看 cohort 样本结果

```python
# 查看前 5 个样本
con.execute("SELECT * FROM dka_cohort LIMIT 5").df()

# 查看总人数
con.execute("SELECT COUNT(*) FROM dka_cohort").fetchone()[0]
```

---

## ✅ cohort 表字段说明：

| 字段名          | 含义             |
| ------------ | -------------- |
| `subject_id` | 患者唯一编号         |
| `hadm_id`    | 住院编号           |
| `stay_id`    | ICU入组编号        |
| `intime`     | ICU入科时间        |
| `outtime`    | ICU出科时间        |
| `age`        | anchor 年龄（已成年） |
| `gender`     | 性别（M/F）        |

---

## 📘 下一阶段目标

我们将基于当前筛选好的 cohort 表，继续进行：

### 📁 阶段三：构建 AKI 标签（基于 Scr 与尿量，符合 KDIGO 标准）

届时我们将：

* 用 DuckDB 查询 `labevents` 提取 Scr 时间序列；
* 用 Python 逻辑计算是否满足 AKI 诊断标准；
* 添加新列 `aki_label` 到 `dka_cohort`。

---

📩 是否现在进入**阶段三：构建 AKI 标签**？我将为您输出精细的 Scr 分析 SQL + AKI 标记函数。
