下面是**第6章：AKI（KDIGO）定义与实现路径**的详细学习笔记。
全章坚持你的技术约束：**DuckDB + CSV/Parquet + Python（pandas/Polars）+ Jupyter（VSCode）**，**不使用 SQL / Postgres**。示例均为**数据帧工作流**范式，可直接放进后续 Notebook 里运行与复用。

---

# 第6章 AKI（KDIGO）定义与实现路径

## 6.1 KDIGO 标准要点与算法选择

### 6.1.1 KDIGO 判定（面向实现的要点）

**以血清肌酐（SCr）与尿量（UO）两条准则判定，取更重的分期。**

* **“是否 AKI”判定**

  * 在 **48 小时** 内，SCr 较基线**上升 ≥ 0.3 mg/dL**；或
  * 在 **7 天** 内，SCr 升至基线的 **≥ 1.5 倍**；或
  * 尿量 **< 0.5 mL/kg/h 持续 ≥ 6 小时**。
* **分期（取更严重者）**

  * **Stage 1**：SCr 1.5–1.9 × 基线（或↑≥0.3 mg/dL）；或 UO < 0.5 mL/kg/h 持续 6–12h
  * **Stage 2**：SCr 2.0–2.9 × 基线；或 UO < 0.5 mL/kg/h 持续 ≥12h
  * **Stage 3**：SCr ≥3.0 × 基线，**或** SCr ≥ 4.0 mg/dL（不论倍数），**或** 启动肾脏替代治疗（RRT）；或 UO < 0.3 mL/kg/h 持续 ≥24h，**或** 无尿 ≥12h

> 我们的预测任务采用**相对窗口**：
>
> * **特征窗**：ICU 入科后 **0–24h**（`icustays.intime` 为锚点）
> * **标签窗**：ICU 入科后 **24–48h**
>   我们在**标签窗**内评估是否满足 KDIGO（≥Stage 1），并保存最重分期（0/1/2/3）。

### 6.1.2 实现策略（为何要两条路线）

* **路线 A（推荐，若可得）**：使用官方/社区的 **`mimic-iv-derived`** 中的 KDIGO 表，能最大限度对齐文献方法学（含尿量、RRT、边界细节）。
* **路线 B（教学向）**：当没有 derived 版本可用时，先做 **SCr-only** 的近似 KDIGO（必要时加入尿量），快速跑通端到端流程，之后再迭代完善。

---

## 6.2 两条路线

### 6.2.A 使用 `mimic-iv-derived`（CSV/Parquet 版）

> 思路：直接读取 KDIGO 结果表（例如 `kdigo_aki` 或类似命名），在**标签窗 \[24h,48h)** 内取**最大分期**，生成 `aki_stage_24_48h` 与 `aki_label`（≥1 为 1）。

**数据帧流程（示例，Polars）**

```python
from pathlib import Path
import polars as pl

RAW = Path("data/raw")
DER = Path("data/derived"); DER.mkdir(parents=True, exist_ok=True)

# 1) 读取 ICU 停留（锚点）
icu = (pl.scan_csv(RAW/"mimiciv_icu/icustays.csv.gz")
         .select(["subject_id","hadm_id","stay_id","intime","outtime"])
         .collect())

# 2) 读取 KDIGO 派生表（命名可能不同；这里假设包含 stay_id, charttime, aki_stage）
KDIGO = RAW/"mimiciv_derived/kdigo_aki.csv.gz"   # 若是 Parquet 则 read_parquet
kdigo = (pl.scan_csv(KDIGO)
           .select(["stay_id","charttime","aki_stage"])
           .collect())

# 3) 标签窗 [24h,48h)：按 stay_id + intime 对齐后取最大分期
lbl = (kdigo.join(icu.select(["stay_id","intime"]), on="stay_id", how="inner")
             .with_columns((pl.col("charttime") - pl.col("intime")).alias("dt"))
             .filter((pl.col("dt") >= pl.duration(hours=24)) & (pl.col("dt") < pl.duration(hours=48)))
             .groupby("stay_id")
             .agg(pl.col("aki_stage").max().alias("aki_stage_24_48h"))
             .with_columns((pl.col("aki_stage_24_48h") >= 1).cast(pl.Int8).alias("aki_label")))

lbl.write_parquet(DER/"label_kdigo_24_48h.parquet")
lbl.head()
```

**优势**：覆盖尿量、RRT、边界规则；对齐已有研究。
**注意**：不同版本的 derived 表可能有细节差异（列名、时间粒度、是否拆分 SCr/UO 两条分期）。在落地前，先通过 `head()/schema` 确认列名与含义。

---

### 6.2.B 教学向自建标签：基于 SCr（±尿量）

> 思路：纯用原始 `labevents`（SCr）与（可选）尿量记录，**独立实现**一个“窗口化 KDIGO 近似”。先用 **SCr-only** 跑通，再在资源允许时加入尿量与 RRT。

#### 6.2.B.1 准备：找出肌酐（SCr）的 `itemid` 并统一单位

* 单位换算：**SCr** 常见 `μmol/L` 与 `mg/dL`，换算：`mg/dL = μmol/L ÷ 88.4`
* 建议仅保留数值列 `valuenum`，丢弃纯字符串 `value`；并在转换后统一命名为 `valuenum_std`。

```python
import polars as pl
from pathlib import Path

RAW = Path("data/raw"); INTERIM = Path("data/interim"); INTERIM.mkdir(parents=True, exist_ok=True)

D_LAB = RAW/"mimiciv_hosp/d_labitems.csv.gz"
LAB   = RAW/"mimiciv_hosp/labevents.csv.gz"
ICU   = RAW/"mimiciv_icu/icustays.csv.gz"

# 1) 找 SCr 的 itemid（基于关键词；你也可以维护一份人工白名单）
d_lab = pl.scan_csv(D_LAB).select(["itemid","label"]).collect()
scr_ids = (
    d_lab.with_columns(pl.col("label").str.to_lowercase())
         .filter(pl.col("label").str.contains("creatin"))  # "creatinine"
         .select("itemid").to_series().to_list()
)

# 2) 读取 ICU 停留（锚点）
icu = (pl.scan_csv(ICU)
         .select(["stay_id","hadm_id","intime"])
         .collect())

# 3) 只取相关住院的 SCr 记录（减小 I/O），并做单位标准化
hadm_whitelist = icu["hadm_id"].unique().to_list()
scr_core = (
    pl.scan_csv(LAB)
      .select(["hadm_id","itemid","charttime","valuenum","valueuom"])
      .filter(pl.col("hadm_id").is_in(hadm_whitelist))
      .filter(pl.col("itemid").is_in(scr_ids))
      .filter(pl.col("valuenum").is_not_null())
      .collect(streaming=True)
)

def standardize_creatinine(df: pl.DataFrame) -> pl.DataFrame:
    vunit = pl.col("valueuom").str.to_lowercase()
    return df.with_columns(
        pl.when(vunit.is_in(["µmol/l","umol/l","μmol/l"]))
          .then(pl.col("valuenum") / 88.4)
          .when(vunit == "mg/dl")
          .then(pl.col("valuenum"))
          .otherwise(pl.col("valuenum"))    # 若遇到非常见单位，先原样保留（也可先过滤）
          .alias("valuenum_std")
    )

scr = standardize_creatinine(scr_core)
```

#### 6.2.B.2 基线 SCr 的选择（影响极大）

**备选策略（按推荐顺序降级）：**

1. **入科前 7 天**（`[intime-7d, intime)`）内的 **最小 SCr**（更贴近“近似稳定”的肾功能）；
2. 若无，取 **入院前 48 小时** 的最小值；
3. 若仍无，取 **入科后 0–24h 内的最小值**（有偏、但能兜底）；
4. **不推荐**用 eGFR 反推“假想基线”；必要时单独做敏感性分析。

```python
# 计算每个 stay 的基线 SCr（采用 7d 优先、否则 48h、否则 0–24h 最小值兜底）
stays = icu.select(["stay_id","hadm_id","intime"])

# 辅助：在给定窗口内取每个 stay 的最小 SCr
def min_scr_in_window(scr_df: pl.DataFrame, window_name: str, start_col: str, end_col: str):
    return (scr_df
        .groupby("stay_id")
        .agg(pl.col("valuenum_std").min().alias(f"scr_min_{window_name}"))
        .select(["stay_id", f"scr_min_{window_name}"])
    )

# 将 SCr 贴上 stay 与 intime（为窗口计算做准备）
scr_with_stay = (scr.join(stays, on="hadm_id", how="inner")
                    .with_columns((pl.col("charttime") - pl.col("intime")).alias("dt")))

# 7天窗口 [intime-7d, intime)
base7 = (scr_with_stay
         .filter((pl.col("dt") >= -pl.duration(days=7)) & (pl.col("dt") < pl.duration(hours=0)))
         .pipe(min_scr_in_window, "7d", "intime-7d", "intime"))

# 48h 窗口 [intime-48h, intime)
base48h = (scr_with_stay
         .filter((pl.col("dt") >= -pl.duration(hours=48)) & (pl.col("dt") < pl.duration(hours=0)))
         .pipe(min_scr_in_window, "48h", "intime-48h", "intime"))

# 0–24h 兜底窗口 [0h,24h)
base0_24 = (scr_with_stay
         .filter((pl.col("dt") >= pl.duration(hours=0)) & (pl.col("dt") < pl.duration(hours=24)))
         .pipe(min_scr_in_window, "0_24h", "intime", "intime+24h"))

# 合并并逐级回填（优先 7d，其次 48h，最后 0–24h）
base = (stays
        .join(base7, on="stay_id", how="left")
        .join(base48h, on="stay_id", how="left")
        .join(base0_24, on="stay_id", how="left")
        .with_columns(
            pl.coalesce(["scr_min_7d","scr_min_48h","scr_min_0_24h"]).alias("scr_baseline")
        )
        .select(["stay_id","scr_baseline"]))
```

#### 6.2.B.3 标签窗内的 SCr 峰值与分期

* **标签窗**：`[intime+24h, intime+48h)`
* 计算 `scr_peak_24_48h`，与 `scr_baseline` 比较，得到分期（SCr-only 近似）。
* 若 `baseline` 不可得，则该 `stay` 的标签设为缺失（或排除出训练集）。

```python
# 标签窗：[24h, 48h)
peak = (scr_with_stay
          .filter((pl.col("dt") >= pl.duration(hours=24)) & (pl.col("dt") < pl.duration(hours=48)))
          .groupby("stay_id")
          .agg(pl.col("valuenum_std").max().alias("scr_peak_24_48h"))
          .select(["stay_id","scr_peak_24_48h"]))

# 合并基线与峰值，生成分期（SCr-only）
label_scr = (base.join(peak, on="stay_id", how="left")
                 .with_columns([
                     (pl.col("scr_peak_24_48h") - pl.col("scr_baseline")).alias("abs_rise"),
                     (pl.col("scr_peak_24_48h") / pl.col("scr_baseline")).alias("ratio")
                 ])
)

def stage_from_scr(abs_rise, ratio):
    # 返回 0/1/2/3；None 表示无法判定（缺基线或缺峰值）
    if abs_rise is None or ratio is None:
        return None
    if ratio >= 3.0 or (abs_rise >= 4.0):  # SCr≥4.0 mg/dL 属于 Stage 3（近似：用“达到≥4.0”的条件）
        return 3
    if ratio >= 2.0:
        return 2
    if ratio >= 1.5 or abs_rise >= 0.3:
        return 1
    return 0

label_scr = label_scr.with_columns(
    pl.struct(["abs_rise","ratio"]).map_elements(lambda r: stage_from_scr(r["abs_rise"], r["ratio"])).alias("aki_stage_scr")
).with_columns(
    (pl.col("aki_stage_scr").fill_null(0) >= 1).cast(pl.Int8).alias("aki_label_scr")
)

label_scr.select(["stay_id","scr_baseline","scr_peak_24_48h","aki_stage_scr","aki_label_scr"]).head()
```

> **说明**：上面的 `abs_rise >= 4.0` 条件是对“SCr ≥ 4.0 mg/dL 即 Stage 3”的近似写法。严格实现应对**峰值本身是否 ≥ 4.0 mg/dL**做判定。你也可以用 `pl.when(pl.col("scr_peak_24_48h") >= 4.0).then(3)` 的方式更明确。

#### 6.2.B.4 可选：加入尿量与 RRT（改进版）

**尿量（UO）**

* 来源：`chartevents` 中的尿量相关 `itemid`（字典 `d_items` 中常以 “Urine Output”/“Urine Volume” 等字样）；或 `mimic-iv-derived` 的 `urine_output` 表。
* 处理：以 **mL** 为基础，合计 **6h/12h/24h** 窗口内的尿量；若能获得体重（`omr`/ICU 体重项），则换算为 **mL/kg/h**。
* 判定：在**标签窗**内每个连续窗段检查是否低于阈值（Stage 1/2/3 的时长门槛）。

**RRT（肾脏替代治疗）**

* 检出途径（可二选一或组合）：

  * 诊断/操作编码（`procedures_icd` 中透析相关代码）；
  * ICU 给药/置换液相关事件（不同版本表结构差异较大；若无把握，先用 ICD 操作代码）。
* 规则：**一旦检测到在标签窗内启动 RRT，直接判为 Stage 3（若未在窗内也可作为病程特征）**。

> 实作建议：先跑通 **SCr-only**，并记录“具备尿量/体重/RRT 信息的样本”比例，再迭代加入 UO + RRT 的分期覆盖。这样能清楚看到每步的样本数与性能变化。

---

## 6.3 风险与偏倚：基线肌酐、入院前记录缺失、RRT 干预

**为什么要这一节？**
AKI 标签的构建**不是纯客观“读数”**，里面有一系列选择，会直接影响阳性率、分期分布与模型表现。透明陈述这些选择与局限，是医学建模合规与可复现的关键。

### 6.3.1 基线肌酐的选择偏倚

* **“最小值”基线**（如 7 天最小）趋向保守（更容易判定为 AKI），但对本已处于上升期的患者可能低估。
* **“入院前缺失”** 会导致以 **0–24h 兜底**作为基线，从而**低估** AKI（因为基线已偏高）。
* **eGFR 反推基线**（假定正常肾功能）可能在老年/慢病患者中**系统性偏差**，建议仅作为**敏感性分析**。
  **建议**：在 QC 报告中输出**基线来源标记**（7d/48h/0–24h），分层报告 AKI 发生率。

### 6.3.2 采样频次与检测偏倚

* 病情更重的患者抽血更频繁，**更容易被“捕捉到”** 达到判定阈值；这会让标签与“被关注程度”相关。
  **建议**：统计特征窗/标签窗内的 **测量次数** 并作为协变量或进行分层分析。

### 6.3.3 SCr 的生理滞后 & 稀释效应

* SCr 对 GFR 的变化存在**时间滞后**；液体复苏/稀释可能**短期降低 SCr**。
  **建议**：补充尿量与液体平衡特征；对极端液体输入做敏感性分析。

### 6.3.4 尿量与利尿剂、导尿差异

* 利尿剂会改变尿量，导尿管管理也影响**记录完整性**。
  **建议**：在可用时纳入**药物暴露**与**护理流程**（导尿相关）信息；并在方法中说明限制。

### 6.3.5 RRT 的指征差异

* 启动 RRT 的门槛与医院/科室实践相关，**非纯粹生理阈值**。
  **建议**：将 RRT 作为 **Stage 3 的充分条件** 同时，进行“**排除 RRT** 再评估”的敏感性分析。

### 6.3.6 截尾与竞争风险

* 在标签窗 **提前出院/死亡** 的住院/ICU 停留，AKI 观察时间不足。
  **建议**：

  * 作为 **未知标签** 排除；或
  * 定义“在可观察时段内的 AKI”，并在论文中清晰说明。

---

## 产出与检查清单（本章落地结果）

**数据产出（Parquet）**

* `derived/label_kdigo_24_48h.parquet`（路线 A）：`stay_id`, `aki_stage_24_48h`, `aki_label`
* `derived/label_scronly_24_48h.parquet`（路线 B）：`stay_id`, `scr_baseline`, `scr_peak_24_48h`, `aki_stage_scr`, `aki_label_scr`, `baseline_source`（7d/48h/0–24h）

**QC 报告（JSON）**

* 基线来源分布（7d/48h/0–24h 的占比）；
* 标签窗内 SCr 测量次数分布、缺失率；
* AKI 阳性率（overall / 分层）；
* 若加入 UO/RRT：覆盖率与其对分期的贡献。

**自检清单**

* [ ] 标签窗定义是否严格为 `[24h,48h)`（未泄漏特征窗信息）
* [ ] 基线来源是否记录并在 EDA 中分层
* [ ] 单位是否统一为 `mg/dL`（SCr）
* [ ] 处理了**提前出院/死亡**等“观察不足”的样本
* [ ] 跑通小样本端到端，核对 AKI 阳性率是否在合理范围（避免 0% 或 90% 的极端）

---

如果你对第6章的笔记满意，我可以直接把\*\*“路线 B：SCr-only 标签生成”\*\*封装成一个可复用的 **Notebook/脚本骨架**（`build_labels_scr_only.ipynb`），并加上 **QC 输出** 与 **配置化**（窗口大小、基线策略、itemid 白名单）。接下来我们就能进入 **第7章：研究队列与时间窗**，把“样本选择 & 切分”与“特征窗/标签窗严格隔离”落成可执行流水线。
