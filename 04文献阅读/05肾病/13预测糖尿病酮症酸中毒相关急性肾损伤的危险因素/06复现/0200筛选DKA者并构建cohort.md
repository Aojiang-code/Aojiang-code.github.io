非常好，我们现在详细进入：

---

# 📁 阶段二：使用 DuckDB SQL 构建 `dka_cohort`

🎯 **目标**：从已注册的 MIMIC-IV 数据表中筛选出**ICU 中的 DKA 成人患者**，并构建包含基本人口信息与 ICU 入出时间的 cohort 视图。

---

## ✅ 一、回顾我们需要的信息

我们需要筛选出符合以下条件的患者：

| 条件         | 数据来源表           | 字段说明                      |
| ---------- | --------------- | ------------------------- |
| 有 DKA 诊断   | `diagnoses_icd` | ICD-9/10 编码匹配             |
| 有 ICU 入科记录 | `icustays`      | stay\_id, intime, outtime |
| 年龄 ≥ 18    | `patients`      | anchor\_age 字段            |
| 可获取性别      | `patients`      | gender 字段                 |

---

## ✅ 二、确认所需的 ICD 编码

根据临床经验和论文使用，我们确定 DKA 对应以下 ICD 编码：

```text
ICD-9:
- 250.10, 250.11, 250.12 → 转换为 25010, 25011, 25012

ICD-10:
- E10.10, E11.10, E13.10 → 转换为 E1010, E1110, E1310
```

在 `diagnoses_icd` 表中，ICD 编码有些含点（.），我们需要用 `REPLACE(icd_code, '.', '')` 来标准化。

---

## ✅ 三、创建 dka\_cohort 的 SQL 查询

以下 SQL 将在您已连接好的 `mimiciv.duckdb` 中运行：

```python
query = """
-- 创建 dka_cohort 视图，供后续步骤使用
CREATE OR REPLACE VIEW dka_cohort AS
WITH dka_hadm_ids AS (
    SELECT DISTINCT hadm_id
    FROM diagnoses_icd
    WHERE REPLACE(icd_code, '.', '') IN (
        '25010', '25011', '25012',  -- ICD-9
        'E1010', 'E1110', 'E1310'   -- ICD-10
    )
),
dka_icu AS (
    SELECT 
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        p.anchor_age AS age,
        p.gender
    FROM icustays i
    INNER JOIN dka_hadm_ids d ON i.hadm_id = d.hadm_id
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
    WHERE p.anchor_age >= 18
)
SELECT * FROM dka_icu
"""
con.execute(query)
```

---

## ✅ 四、查看构建结果

```python
# 查看前几行
con.execute("SELECT * FROM dka_cohort LIMIT 5").df()
```

```python
# 查看总共筛选出多少名 DKA + ICU 成年患者
con.execute("SELECT COUNT(*) FROM dka_cohort").fetchone()[0]
```

---

## ✅ 五、输出表结构说明

`dka_cohort` 表字段如下：

| 字段名          | 含义                |
| ------------ | ----------------- |
| `subject_id` | 患者 ID             |
| `hadm_id`    | 住院 ID             |
| `stay_id`    | ICU stay ID       |
| `intime`     | ICU 入科时间          |
| `outtime`    | ICU 出科时间          |
| `age`        | 患者年龄（anchor\_age） |
| `gender`     | 性别（M/F）           |

---

## 🧪 建议调试操作

检查值是否合理（例如年龄最小值、最大值）：

```python
con.execute("SELECT MIN(age), MAX(age) FROM dka_cohort").df()
```

统计性别分布：

```python
con.execute("SELECT gender, COUNT(*) FROM dka_cohort GROUP BY gender").df()
```

---

## ✅ 阶段二产出成果

您已成功创建了视图 `dka_cohort`，它是后续：

* 构建 AKI 标签（阶段三）、
* 提取 24h 内特征变量（阶段四）、
* 构建分析数据集（阶段五）

的核心基础。

---

是否继续进入：

# 📁 阶段三：构建 AKI 标签（基于 KDIGO 标准，结合 Scr 与尿量）？

我将为您提供：

* DuckDB SQL 提取 Scr 序列的查询
* Python 实现 AKI 判定的逻辑函数（完全遵循 KDIGO 标准）
