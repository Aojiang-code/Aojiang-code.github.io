非常好，现在我们进入：

---

# 📁 阶段三：构建 AKI 标签（KDIGO 标准）

🎯 目标：根据 KDIGO 标准（Kidney Disease: Improving Global Outcomes），结合**血清肌酐（Scr）**和**尿量**，在 ICU 入科后的观察窗口内判断患者是否发生了 AKI，并构建二分类标签 `aki_label`。

本阶段完全复现论文《Predicting the risk factors of DKA-associated AKI》中的定义与方法。

---

## ✅ 一、KDIGO 诊断标准（论文方法依据）

AKI 的诊断标准包括以下任意一项达标：

| 标准                      | 说明                     |
| ----------------------- | ---------------------- |
| 血清肌酐（Scr）升高 ≥ 0.3 mg/dL | 在 **48 小时内**           |
| Scr ≥ 1.5 × baseline    | 在 **7 天内**             |
| 尿量 < 0.5 mL/kg/h        | 持续 **至少 6 小时**（部分研究可选） |

📌 文中以 **Scr 为主要判断依据**，我们也将基于此进行构建，尿量可作为补充。

---

## ✅ 二、步骤总览

| 步骤  | 任务                                  | 所用表         |
| --- | ----------------------------------- | ----------- |
| 3.1 | 获取 ICU 内 Scr 检测值                    | `labevents` |
| 3.2 | 计算每名患者的 baseline Scr 与 48h peak Scr | -           |
| 3.3 | 判断是否符合 AKI 条件，构建标签                  | -           |
| 3.4 | 将 AKI 标签合并进 `dka_cohort` 表          | -           |

---

## ✅ 三、详细步骤说明与代码

### 🔍 3.1：获取 ICU 内 Scr 检测值（ItemID = 50912）

```python
# Scr 项目编号（MIMIC-IV 中默认为 50912）
scr_itemid = 50912

# 提取 Scr 数据（入 ICU 后 7 天内）
query_scr = f"""
SELECT
    l.subject_id,
    l.hadm_id,
    l.charttime,
    l.valuenum AS scr,
    c.stay_id,
    c.intime
FROM labevents l
INNER JOIN dka_cohort c
    ON l.hadm_id = c.hadm_id
WHERE l.itemid = {scr_itemid}
"""
scr_df = con.execute(query_scr).df()
```

---

### ⏱ 3.2：计算时间差与观察窗口

```python
# 时间处理
scr_df['charttime'] = pd.to_datetime(scr_df['charttime'])
scr_df['intime'] = pd.to_datetime(scr_df['intime'])
scr_df['hours_from_icu'] = (scr_df['charttime'] - scr_df['intime']).dt.total_seconds() / 3600

# 筛选 ICU 后 0-168 小时（7 天）内的值
scr_df = scr_df[(scr_df['hours_from_icu'] >= 0) & (scr_df['hours_from_icu'] <= 168)]
```

---

### 📐 3.3：计算 baseline Scr 与 peak Scr（48h 内），判断 AKI

```python
aki_list = []

for stay_id, group in scr_df.groupby('stay_id'):
    baseline = group[group['hours_from_icu'] <= 12]['scr'].min()  # baseline 取最早 12h 最低值
    peak_48h = group[group['hours_from_icu'] <= 48]['scr'].max()  # 48h 峰值

    # AKI 判定逻辑
    if pd.notnull(baseline) and pd.notnull(peak_48h):
        if (peak_48h - baseline >= 0.3) or (peak_48h >= 1.5 * baseline):
            aki_list.append((stay_id, 1))
        else:
            aki_list.append((stay_id, 0))
    else:
        aki_list.append((stay_id, 0))  # 缺失数据视为未发生
```

---

### ✅ 3.4：构建标签 DataFrame 并合并进 `dka_cohort`

```python
aki_df = pd.DataFrame(aki_list, columns=['stay_id', 'aki_label'])

# 保存 AKI 标签临时表（供后续使用）
con.register('aki_temp', aki_df)

# 构建新 cohort，附带 AKI 标签
query_merge = """
CREATE OR REPLACE VIEW dka_with_aki AS
SELECT c.*, a.aki_label
FROM dka_cohort c
LEFT JOIN aki_temp a ON c.stay_id = a.stay_id
"""
con.execute(query_merge)
```

---

## 📊 五、检查构建结果

```python
# 查看新 cohort 表前几行
con.execute("SELECT * FROM dka_with_aki LIMIT 5").df()

# 查看标签分布
con.execute("SELECT aki_label, COUNT(*) FROM dka_with_aki GROUP BY aki_label").df()
```

---

## ✅ 阶段三成果总结

您现在已经成功构建并保存了 `dka_with_aki` 表：

| 字段   | 说明                                                     |
| ---- | ------------------------------------------------------ |
| 原始字段 | subject\_id, hadm\_id, stay\_id, age, gender, intime 等 |
| 新增字段 | `aki_label`：0（未发生），1（发生）                               |

---

下一步我们将进行：

### 📁 阶段四：提取 ICU 入科后 24 小时内的特征变量（实验室 + 生命体征 + 干预变量）

是否继续？我将提供按论文提取 BUN、WBC、PLT、Na、HR、RR 等变量的 DuckDB 查询与处理模板。
