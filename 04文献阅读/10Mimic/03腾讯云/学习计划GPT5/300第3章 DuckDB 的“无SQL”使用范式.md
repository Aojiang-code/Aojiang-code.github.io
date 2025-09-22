下面是**第3章：DuckDB 的“无SQL”使用范式**的详细学习笔记。整章坚持你的技术约束——**不使用 SQL / Postgres**，而是以 **DuckDB + CSV + Parquet + Python（pandas/Polars）+ Jupyter（VSCode）** 为主线，强调“数据帧工作流 + 渐进式转换 + 高效读写”。

---

# 第3章 DuckDB 的“无SQL”使用范式

## 3.1 为何选择 DuckDB：零服务、列存执行、向量化、并行

**核心优势（贴合 MIMIC-IV 大表场景）**

* **零服务**：嵌入式引擎（单文件/进程内），无守护进程、无端口管理；在 Notebook 中即开即用。
* **列式与向量化**：对列的批处理（vectorized execution）+ SIMD，加速扫描、过滤、聚合。
* **并行**：自动多线程读取与计算（尤其是 Parquet/CSV 扫描）。
* **原生 Arrow 生态**：与 pandas/Polars/pyarrow 互操作顺滑，适合“数据帧优先”的代码风格。
* **即席/延迟读取**：对 CSV/Parquet 的**按列/按条件**扫描（列裁剪 & 谓词下推），减少 I/O 与内存占用。
* **轻量缓存**：对同一 Parquet 的多次访问成本低（OS 页缓存 + 列式压缩），配合“中间层 Parquet”能显著提速。

> 我们的范式：**把 DuckDB 当作极速文件引擎与中间数据湖工具**；**主要变换在数据帧里做**（Polars/pandas），**不写 SQL**。

---

## 3.2 数据帧工作流：DuckDB ↔ pandas/Polars 的互操作

> 目标：**不写 SQL**，仍然能把 DuckDB 的高效 I/O 与数据帧的灵活变换结合起来。

### 3.2.1 最简连接与读写（Relation API + Arrow 桥）

```python
from pathlib import Path
import duckdb, polars as pl, pyarrow.parquet as pq

con = duckdb.connect(database=":memory:")  # 也可指定文件，如 "cache.duckdb"

# 读 Parquet → 转成 Polars（零拷/近零拷 Arrow 桥）
tbl = con.read_parquet("data/interim/patients.parquet").arrow()
pl_df = pl.from_arrow(tbl)

# 将 Polars/Pandas/Arrow 回写为 Parquet（统一由 Polars/pyarrow 完成）
pl_df.write_parquet("data/interim/patients_clean.parquet")
```

### 3.2.2 注册数据帧 → DuckDB（无需 SQL，只为高效 join/聚合）

DuckDB 可直接“接纳”**Arrow** 对象（Polars 可 `.to_arrow()`）：

```python
import polars as pl, duckdb
con = duckdb.connect()

df_left  = pl.DataFrame({"id":[1,2,3], "x":[0.1,0.2,0.3]})
df_right = pl.DataFrame({"id":[2,3,4], "y":[10,20,30]})

# 注册为 DuckDB 内部表（基于 Arrow，无需 SQL）
con.register("left_tbl",  df_left.to_arrow())
con.register("right_tbl", df_right.to_arrow())

# 用 DuckDB 做“大表 join + 聚合”时更快（这里仍可避免写 SQL：先让 DuckDB 负责 I/O，再回到 Polars变换）
joined = con.table("left_tbl").join(con.table("right_tbl"), ["id"], "inner")
# 取回 dataframe（pandas/Arrow/Polars均可）
out = pl.from_arrow(joined.arrow())
```

> 小结：**复杂变换尽量在 Polars/pandas 做**；DuckDB 负责**读写 & 大 join**；两者之间通过 Arrow 零拷桥接。

---

## 3.3 按需读取 CSV：列裁剪、行过滤、延迟执行、分块策略

> 大表（如 `labevents/chartevents`）的关键是：**只读你要的列，只拉你要的行**，并尽可能**惰性/流式**。

### 3.3.1 列裁剪 + 行过滤（Polars 的惰性扫描）

```python
import polars as pl
from pathlib import Path

csv_path = Path("data/raw/mimiciv_hosp/labevents.csv.gz")
wanted_cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
itemids = [50912, 50931]  # 举例：肌酐、尿素氮等的 itemid

lf = (
    pl.scan_csv(csv_path, has_header=True, ignore_errors=True)
      .select([c for c in wanted_cols])            # 列裁剪
      .filter(pl.col("itemid").is_in(itemids))     # 行过滤（谓词下推）
      .filter(pl.col("valuenum").is_not_null())    # 清洗
)

# 惰性：直到 collect() 才执行
df_small = lf.collect(streaming=True)  # 大文件用流式，降低内存峰值
```

**要点**

* `scan_csv` 是**惰性**的，配合 `.select/.filter` 会做**谓词下推与列裁剪**，读得更少更快。
* `collect(streaming=True)` 启动**流式执行**，适合超大 CSV。
* 对时间窗过滤：先把**候选行缩小**（例如只取目标 `itemid`），再做时间过滤（减轻解析成本）。

### 3.3.2 分片/分块策略（当单个 CSV 超大）

* **按键分片**：例如基于 `subject_id % N` 分成 N 份，逐块处理/输出 Parquet；或按 `anchor_year_group`/月份（若可用）。
* **按任务窗口**：以 ICU 入科时间为锚点，先抽取相关 `hadm_id/stay_id` 列表，再**半连接**（semi-join）过滤 `labevents`/`chartevents` 仅保留相关就诊。
* **Chunk 读取（pandas 备选）**：

  ```python
  import pandas as pd
  it = pd.read_csv(csv_path, chunksize=2_000_000, usecols=wanted_cols)
  for chunk in it:
      # 过滤/清洗/聚合后写入 Parquet（append 模式）
      ...
  ```

  > 建议优先 **Polars scan + streaming**；pandas chunksize 仅作兜底。

### 3.3.3 时间与类型解析

* 仅在**必要列**上做时间解析（例如 `charttime`），其他列保持原始数值/字符串。
* 明确 `dtype`，对大范围枚举（如 `valueuom`）可设为**分类/字符串**，避免 Python `object` 带来的性能损耗。

---

## 3.4 性能与容量：从 CSV 到 Parquet 的渐进式转换与缓存

**理念：** 原始数据只在 `raw/` 以 CSV(.gz) 存放；数据帧工作流中尽量把**中间层**与**下游特征**写成 **Parquet**（列存、压缩、含 schema），相当于“**渐进式数据湖**”。

### 3.4.1 一次性/增量转换器（CSV → Parquet）

```python
from pathlib import Path
import polars as pl

def csv_to_parquet(
    src_csv: Path, dst_parquet: Path,
    usecols: list[str] | None = None,
    predicate=None,          # 传入 pl 表达式，如 pl.col("itemid").is_in(itemids)
    row_group_size: int = 256_000,   # 写 Parquet 的行组大小，平衡速度/体积
    compression: str = "zstd"        # 压缩编码：zstd 通常更优
):
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_csv(src_csv)
    if usecols:
        lf = lf.select([c for c in usecols])
    if predicate is not None:
        lf = lf.filter(predicate)
    df = lf.collect(streaming=True)
    df.write_parquet(
        dst_parquet,
        compression=compression,
        statistics=True,        # 让下游引擎更好做谓词下推
        use_pyarrow=True
    )
    return dst_parquet

# 示例：只抽取所需列与 itemid → Parquet
csv_to_parquet(
    Path("data/raw/mimiciv_hosp/labevents.csv.gz"),
    Path("data/interim/labevents_core.parquet"),
    usecols=["subject_id","hadm_id","itemid","charttime","valuenum","valueuom"],
    predicate=pl.col("itemid").is_in([50912, 50931])
)
```

**Tips**

* **设置合理的 `row_group_size`**：行组更大 → 吞吐更高、但随机读取略差；反之亦然。
* 开启 **statistics**（默认由 Arrow 控制）：提升后续读取的谓词下推效果。
* 先**按需裁剪**再落 Parquet，显著减小磁盘占用与 I/O。

### 3.4.2 分区写出与组织（利于多进程/增量）

```python
def write_partitioned_parquet(df: pl.DataFrame, base_dir: Path, by_cols: list[str]):
    base_dir.mkdir(parents=True, exist_ok=True)
    # 以分区列写多文件（示例：按 anchor_year_group/或 subject_id_bucket）
    df.write_parquet(
        base_dir,
        use_pyarrow=True,
        pyarrow_options={"partition_cols": by_cols, "compression":"zstd"}
    )

# 举例：将已过滤的 labevents 按 valueuom 分区（仅做演示；实际按人群/时间/桶更常见）
filtered = pl.scan_parquet("data/interim/labevents_core.parquet").collect()
write_partitioned_parquet(filtered, Path("data/interim/labevents_part/"), ["valueuom"])
```

**如何选分区列？**

* 选择**高选择性、查询常用**的维度（如 `itemid`、近似时间分段、患者桶 `subject_id % 100`）。
* 保持**分区数量适中**（几十到几百），避免极端小文件风暴。

### 3.4.3 DuckDB + Parquet 的“缓存化”读取

* 只要你把中间结果写成 **Parquet**，后续反复读取的成本就很低（列式压缩 + OS 缓存）。
* 在 Notebook 中把**多个 Parquet**拼接为一个逻辑数据集，依旧无需 SQL：

  ```python
  import duckdb, polars as pl
  con = duckdb.connect()
  # 读一组分区：
  tbl = con.read_parquet("data/interim/labevents_part/*/*.parquet").arrow()
  df  = pl.from_arrow(tbl)
  ```
* 若某段计算会被反复使用，**提前物化**到 Parquet（例如“已对齐单位的化验子集”、“按 stay 的 0–24h 聚合结果”），后续章节直接读取即可。

### 3.4.4 内存与性能建议（实战清单）

* **先裁剪、后转换**：永远先 `select/usecols`，再写 Parquet。
* **尽量使用 Polars 惰性 + streaming**：让扫描与过滤都“推”到最前面。
* **数值类型与单位先统一**：化验/生命体征先做单位标准化，再聚合。
* **分类编码**：高基数字符串（如 `label`/`valueuom`）可转 `Categorical`，减少内存。
* **多进程/并行**：按分区（目录或哈希桶）并行处理，避免竞争一个巨大 CSV。
* **监控**：在关键 Notebook 加入“文件大小、行数、峰值内存、耗时”的打印，便于后续优化。

---

## 本章小结（可执行要点）

1. **把 DuckDB 当“文件加速器”**：CSV/Parquet 极速读写 + 与 Arrow/Polars 的零拷互转；复杂变换尽量在数据帧做。
2. **惰性/流式**是处理大 CSV 的关键：`pl.scan_csv(...).select(...).filter(...).collect(streaming=True)`。
3. **渐进式数据湖**：原始 CSV 保存在 `raw/`；任何中间与特征写 Parquet 到 `interim/derived/`；反复用、读取快。
4. **分区与行组**要均衡：按查询维度选择分区；合理设置行组大小；启用压缩与统计信息。
5. **不写 SQL 也不影响效率**：读写/合并由 DuckDB 和 Arrow 负责，变换由 Polars/pandas 实现。

---

如果你对本章满意，下一步我们进入 **第4章：MIMIC-IV 模块与关键表（概览）**，并在此基础上给出一套**即插即用的 CSV→Parquet 渐进转换脚本**（覆盖 `patients/admissions/icustays/labevents/...`）与示例 Notebook（无 SQL）。你也可以直接指定某个表（如 `labevents`）让我先写一份“**从 CSV 到 0–24h 聚合特征**”的可运行模板。
