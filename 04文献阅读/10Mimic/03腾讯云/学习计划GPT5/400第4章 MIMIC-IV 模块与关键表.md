下面是**第4章：MIMIC-IV 模块与关键表（概览）**的详细学习笔记。
全章坚持你的技术约束：**DuckDB + CSV/Parquet + Python（pandas/Polars）+ Jupyter（VSCode）**，**不使用 SQL / Postgres**。示例均为**数据帧工作流**范式，可直接嵌入后续 Notebook。

---

# 第4章 MIMIC-IV 模块与关键表（概览）

## 4.1 模块：`hosp`、`icu`、（可选）`ed`、`note`、`derived`

> 推荐：将原始 CSV 放在 `data/raw/mimiciv_<module>/`，按需裁剪后渐进转换为 Parquet（第3章已说明）。

### 4.1.1 `mimiciv_hosp`（住院域，Admission-level）

**用途**：住院入出院、化验、医嘱、诊断/操作编码、转运路径、结局等。
**代表表**：

* `patients`（患者骨干信息）
* `admissions`（住院就诊记录）
* `labevents`（院内化验）
* `transfers`（科室/ICU/病区转运）
* `diagnoses_icd` & `procedures_icd`（ICD-10/9 诊断与操作）
* `omr`（门诊/随访：体重/BMI/血压等，覆盖面与粒度低于 ICU，若本地有）

### 4.1.2 `mimiciv_icu`（ICU 域，Stay-level）

**用途**：ICU 停留信息与高频临床观测。
**代表表**：

* `icustays`（ICU 停留记录，定义 ICU 住院边界）
* `chartevents`（ICU 生命体征/护理/床旁记录，**超大表**）
* `d_items`（ICU 项目字典，用于解释 `chartevents.itemid`）

### 4.1.3 可选模块

* `mimiciv_ed`（急诊）：到院即刻的测量/流程；可作 ICU/住院前置信息。
* `mimiciv_note`（临床文本）：出院小结/护理笔记等（合规处理与发布边界要特别注意）。
* `mimiciv_derived`（派生集合）：已计算的指标/评分（如 **KDIGO AKI**、SOFA、Charlson、尿量聚合、vitalsign 等）。
  我们主线以数据帧自行派生为主；若你有 **CSV/Parquet** 版本的派生表，也可直接读取使用。

---

## 4.2 关联键与时间字段：`subject_id`、`hadm_id`、`stay_id`、`intime/outtime`

三层 ID 是理解与建模的关键（**患者 → 住院 → ICU**），可无缝串联 `hosp` 与 `icu`。

### 4.2.1 三层主键（由粗到细）

* **`subject_id`（患者级）**：同一患者的所有住院/ICU 停留共享该 ID。
* **`hadm_id`（住院级）**：一次住院就诊（Admission）。同一 `subject_id` 可能有多个 `hadm_id`。
* **`stay_id`（ICU-stay 级）**：一次 ICU 停留（Stay）。同一 `hadm_id` 可能有 0\~多次 `stay_id`。

**连接示意**：

```
patients (subject_id)
  ←→ admissions (subject_id, hadm_id)
       ←→ icustays (subject_id, hadm_id, stay_id)

labevents  贴 hadm_id（住院粒度）
chartevents 贴 stay_id（ICU 粒度）
```

### 4.2.2 常见时间字段

* `admissions`：`admittime`, `dischtime`, `deathtime`
* `icustays`：`intime`, `outtime`（ICU 进入/离开）
* 事件表：`charttime`（发生/记录时间），`storetime`（入库时间，可能滞后）
* `omr`：`chartdate`（天级日期）

### 4.2.3 偏移时间与相对窗口

* MIMIC-IV 对**绝对日期**做了**患者级偏移**；**同一患者内部相对时间**是可靠的。
* 构建任务时（例如“入科后 **0–24h** 特征 → 预测 **24–48h** AKI”）使用**相对窗口**：以 `icustays.intime` 或 `admissions.admittime` 为锚点。
* **跨患者**绝对季节/节假日分析不适合；若必须做，使用锚点/分组字段进行**粗粒度近似**并在方法中声明局限（第1章已述）。

---

## 4.3 常用表速览与字段字典（无 SQL，数据帧范式）

> 下面列出与 AKI 预测密切相关的关键列与实操要点；其余字段可在 EDA 中按需保留。

### 4.3.1 `mimiciv_hosp.patients`（患者骨干）

**关键列**：`subject_id`, `gender`, `anchor_age`（或用于年龄计算的字段）, `anchor_year`, `anchor_year_group`
**常见用法**：性别/年龄特征、粗粒度年代控制（研究描述性统计）
**注意**：高龄（>89）合并；anchor 年代用于近似时间背景。
**数据帧读取（Polars 惰性 + 列裁剪）**：

```python
import polars as pl
PAT = "data/raw/mimiciv_hosp/patients.csv.gz"
pat = (
    pl.scan_csv(PAT)
      .select(["subject_id","gender","anchor_year","anchor_year_group"])
      .collect()
)
```

### 4.3.2 `mimiciv_hosp.admissions`（住院就诊）

**关键列**：`subject_id`, `hadm_id`, `admittime`, `dischtime`, `deathtime`,
`admission_type`, `admission_location`, `discharge_location`
**常见用法**：定义住院样本、住院时长、院内死亡等；与 ICU 停留对齐
**注意**：同一患者可能多次住院；建模需按住院/ICU 粒度去重与切分

```python
ADM = "data/raw/mimiciv_hosp/admissions.csv.gz"
adm = (
    pl.scan_csv(ADM)
      .select(["subject_id","hadm_id","admittime","dischtime","deathtime",
               "admission_type","admission_location","discharge_location"])
      .collect()
)
```

### 4.3.3 `mimiciv_hosp.labevents`（化验）

**关键列**：`subject_id`, `hadm_id`, `itemid`, `charttime`, `valuenum`, `valueuom`（`value` 文本仅作补充）
**字典表**：`mimiciv_hosp.d_labitems`（`itemid → label/loinc_code/category/fluid`）
**常见用法**：抽取 **SCr（肌酐）**、**BUN**、电解质、酸碱等 → **0–24h** 聚合（`mean/max/min/last/斜率`）
**注意**：

* 单位统一（`valueuom`）；少数项目有多单位或字符串值，需清洗；
* 同时刻可能多条记录，聚合前做去重/稳健汇总；
* **先筛 `itemid` 再读**，极大降低 I/O。

```python
D_LAB = "data/raw/mimiciv_hosp/d_labitems.csv.gz"
LAB   = "data/raw/mimiciv_hosp/labevents.csv.gz"
d_lab = pl.scan_csv(D_LAB).select(["itemid","label","fluid","category","loinc_code"]).collect()

# 例：用关键词找 SCr/BUN 的 itemid（实际以你数据字典为准）
targets = (
    d_lab.with_columns(pl.col("label").str.to_lowercase())
         .filter(pl.col("label").str.contains("creatin") | pl.col("label").str.contains(r"\bbun\b"))
         .select(["itemid","label"])
)
itemids = targets["itemid"].unique().to_list()

wanted_cols = ["subject_id","hadm_id","itemid","charttime","valuenum","valueuom"]
lab = (
    pl.scan_csv(LAB)
      .select(wanted_cols)
      .filter(pl.col("itemid").is_in(itemids))
      .filter(pl.col("valuenum").is_not_null())
      .collect(streaming=True)
)
```

### 4.3.4 `mimiciv_icu.icustays`（ICU 停留）

**关键列**：`subject_id`, `hadm_id`, `stay_id`, `intime`, `outtime`, `first_careunit`
**常见用法**：定义样本单位（多数分析以 `stay_id` 粒度）、时间锚点（`intime` 用于 0–24h/24–48h 窗口）
**注意**：一个住院可能多次 ICU；AKI 预测常以首次 ICU 为主，或按业务规则选择

```python
ICU = "data/raw/mimiciv_icu/icustays.csv.gz"
icu = (
    pl.scan_csv(ICU)
      .select(["subject_id","hadm_id","stay_id","intime","outtime","first_careunit"])
      .collect()
)
```

### 4.3.5 `mimiciv_icu.chartevents`（ICU 生命体征/护理）

**关键列**：`stay_id`, `itemid`, `charttime`, `valuenum`, `valueuom`
**字典表**：`mimiciv_icu.d_items`（`itemid → label/unitname/linksto`）
**常见用法**：心率/呼吸/血压/体温/SpO₂ 等高频序列 → **0–24h** 聚合（均值/极值/变异度/最后观测/缺失指示）
**注意**：表极大，务必**先筛 `itemid` 再做时间窗聚合**；单位与重复值需要清洗

```python
CHART   = "data/raw/mimiciv_icu/chartevents.csv.gz"
D_ITEMS = "data/raw/mimiciv_icu/d_items.csv.gz"

d_items = pl.scan_csv(D_ITEMS).select(["itemid","label","unitname"]).collect()
# 示例：根据字典中常见标签筛出心率/无创血压的 itemid（实际以你的字典为准）
vital_ids = (
    d_items.with_columns(pl.col("label").str.to_lowercase())
           .filter(pl.col("label").str.contains("heart rate") |
                   pl.col("label").str.contains("non invasive blood pressure"))
           .select("itemid")
           .to_series().to_list()
)

chart = (
    pl.scan_csv(CHART)
      .select(["stay_id","itemid","charttime","valuenum","valueuom"])
      .filter(pl.col("itemid").is_in(vital_ids))
      .filter(pl.col("valuenum").is_not_null())
      .collect(streaming=True)
)
```

### 4.3.6 诊断与字典：`diagnoses_icd` + `d_icd_diagnoses`

**`diagnoses_icd`**：`subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version`
**`d_icd_diagnoses`**：字典（`icd_code + icd_version → long_title`）
**常见用法**：计算 **Charlson/Elixhauser** 合并症；或病种亚组（CKD/糖尿病等）
**注意**：ICD-9 与 ICD-10 共存；映射需按 `icd_version` 选择规则

```python
DIAG  = "data/raw/mimiciv_hosp/diagnoses_icd.csv.gz"
D_DIAG= "data/raw/mimiciv_hosp/d_icd_diagnoses.csv.gz"
diag  = pl.scan_csv(DIAG).select(["subject_id","hadm_id","seq_num","icd_code","icd_version"]).collect()
d_diag= pl.scan_csv(D_DIAG).select(["icd_code","icd_version","long_title"]).collect()
```

### 4.3.7 其他常用表（速览）

* **`transfers`（住院转运）**：`hadm_id`, `careunit`, `intime/outtime`（可重建病区轨迹）
* **`procedures_icd`**：操作/手术（造影、透析等干预）
* **`prescriptions/pharmacy`**：给药与配药（利尿剂/血管活性药等暴露）
* **`microbiologyevents`**：微生物/药敏（感染/脓毒症亚组）
* **`omr`**：门诊/体检记录（`chartdate` + 值/单位；长期趋势）

---

## 4.3.x 典型连接与对齐范式（无 SQL）

### 4.3.x.1 “住院 → ICU → 事件”的三步法

**思路**：选 ICU 样本 → 对齐住院信息 → 拉取事件（按相对时间窗）

```python
import polars as pl

# 1) ICU 样本（示例：首次 ICU 停留）
icu = pl.scan_csv(ICU).select(["subject_id","hadm_id","stay_id","intime","outtime"]).collect()
icu_first = (
    icu.sort("intime")
       .groupby("subject_id", maintain_order=True)
       .agg([pl.all().first()])
       .explode(pl.all().exclude("subject_id"))
)

# 2) 住院信息（只取必要列）
cohort = (
    icu_first.join(
        pl.scan_csv(ADM).select(["subject_id","hadm_id","admittime","dischtime","deathtime"]).collect(),
        on=["subject_id","hadm_id"],
        how="left"
    )
    .select(["stay_id","hadm_id","intime","outtime","admittime","dischtime"])
)

# 3) 拉取 0–24h 化验（先按 hadm_id 半联结，再按 intime 做窗口过滤）
lab_sel = (
    pl.scan_csv(LAB)
      .select(["hadm_id","itemid","charttime","valuenum","valueuom"])
      .filter(pl.col("itemid").is_in(itemids))
      .filter(pl.col("valuenum").is_not_null())
)
lab_joined = lab_sel.join(cohort.select(["hadm_id","stay_id","intime"]), on="hadm_id", how="inner").collect(streaming=True)

lab_0_24h = (
    lab_joined
      .with_columns((pl.col("charttime") - pl.col("intime")).alias("dt"))
      .filter((pl.col("dt") >= pl.duration(hours=0)) & (pl.col("dt") < pl.duration(hours=24)))
      .groupby(["stay_id","itemid"])
      .agg([pl.col("valuenum").mean().alias("mean"),
            pl.col("valuenum").max().alias("max"),
            pl.col("valuenum").min().alias("min"),
            pl.col("valuenum").last().alias("last")])
      .collect()
)
```

### 4.3.x.2 字典映射与单位对齐

* 化验：连 `d_labitems` 获取 `label/loinc_code/category`，用于筛选项目与单位转换（如 **SCr μmol/L ↔ mg/dL**，常用换算：`mg/dL = μmol/L ÷ 88.4`）。
* ICU 项目：连 `d_items` 获取 `label/unitname`；同一生理量可能对应多个 `itemid`，先建立白名单映射。

```python
# itemid → 可读标签
lab_labeled = lab_0_24h.join(d_lab.select(["itemid","label"]).unique(), on="itemid", how="left")

# 示例：单位统一（按项目/单位条件转换）
# lab_labeled = lab_labeled.with_columns(
#     pl.when((pl.col("label").str.contains("creatin")) & (pl.col("valueuom") == "µmol/L"))
#       .then(pl.col("valuenum") / 88.4)
#       .otherwise(pl.col("valuenum"))
#       .alias("valuenum_std")
# )
```

### 4.3.x.3 稳健聚合与缺失处理

* **稳健统计**：`median/IQR`、截尾均值（winsorize）、重复值去重；
* **缺失指示器**：对完全缺失的项目增加布尔列，有助于模型与审计；
* **落地**：中间与特征统一写到 `data/interim/` 或 `data/derived/`（Parquet，必要时分区）。

```python
from pathlib import Path
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)
lab_labeled.write_parquet(INTERIM/"features_labs_0_24h.parquet")
```

### 4.3.x.4 性能小抄

* **先筛列/行，再落盘**：`scan_csv → select → filter → collect(streaming=True)`；
* **优先 Parquet**：中间层列存+压缩+统计信息，后续读取更快；
* **分区**：按常用过滤维度（如 `itemid`、时间桶、`subject_id % N`）分区写出，避免小文件风暴；
* **数据类型**：高基数字符串转 `Categorical`；时间列仅在必要时解析；
* **惰性 & 流式**：Polars lazy + streaming，降低内存峰值；
* **重复使用就物化**：会反复用的中间结果尽早写成 Parquet 做“缓存”。

---

## 小结与下一步

* **模块主干**：`hosp`（住院）与 `icu`（ICU）是主线；`ed/note/derived` 视研究扩展。
* **三层 ID + 相对时间** 是连接与窗口设计的核心。
* **常用表字典** 帮你快速定位值与单位：`d_labitems`、`d_items`、`d_icd_diagnoses`。
* **按需扫描 + 渐进式 Parquet** 可把 I/O 与内存开销压到最低，全程**无 SQL**。

如果你对本章风格与内容满意，我可以立刻给出**第5章：CSV 数据准备与质量校验**（含一键 **CSV→Parquet** 渐进转换脚本 + 质检 Notebook 模板），或者先按你的需求为某张大表（如 `labevents` / `chartevents`）写一份**从原始到 0–24h 聚合特征**的可运行模板。
