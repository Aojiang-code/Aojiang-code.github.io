# AKI ICU队列定义（MIMIC-IV）

## 1. 人群与场景

- 人群：MIMIC-IV数据库中年龄≥18岁的成年患者；
- 场景：在研究时间范围内（2008–2019年）至少有一次ICU入住记录；
- 分析单位：**ICU首次住院（first ICU stay per patient）**。
## 2. 索引时间点（Index Time）

- Index time 定义为患者在研究中**首次ICU入住的入科时间（`icustays.intime`）**。
- 所有与“首日”相关的变量（如首日实验室指标）均以此时间点为起点，
  在入科后0–24小时内的记录中选取。
## 3. 纳入标准

患者需同时满足以下条件方可纳入AKI ICU队列：

1. 年龄 ≥ 18岁
   - 使用 `mimiciv_hosp.patients.anchor_age` 进行判断；
2. 至少有一条ICU入住记录
   - 在 `mimiciv_icu.icustays` 表中有记录；
3. 选取每位患者的**首次ICU入住**作为研究对象
   - 对每个 `subject_id` 仅保留按 `intime` 排序的第一条 `icustays` 记录；
4. 该次住院（`hadm_id`）期间存在急性肾损伤相关诊断
   - 在 `mimiciv_hosp.diagnoses_icd` 表中，满足以下任一条件：
     - ICD-9：`icd_code` 以 `584` 开头（急性肾衰竭/AKI）
     - ICD-10：`icd_code` 以 `N17` 开头（Acute kidney failure）
   - `icd_version = 9` 或 `10` 分别对应 ICD-9 / ICD-10。
## 4. 排除标准

满足以下任一条件的ICU入住将被排除：

1. 年龄缺失或无法推算
   - `anchor_age` 为空；
2. ICU停留时间极短（如 < 6小时）
   - 为避免因转科/行政原因导致的信息量不足；
   - `icustays.outtime - icustays.intime < 6小时`；
3. 住院信息缺失
   - 无法在 `admissions` 表中找到对应 `hadm_id`；
4. （可选，后期增加）明确的终末期肾病/长期透析患者
   - 通过特定ICD-9/10代码或透析程序代码识别；
   - 第一阶段可以暂不排除，后续建模时细化。
## 5. AKI定义版本（Versioning）

- 当前使用的AKI定义记为 **AKI Definition v1.0**：
  - 基于住院期间的ICD-9/ICD-10诊断码识别AKI；
  - ICD-9：`584*`；ICD-10：`N17*`；
  - 不区分严重程度分级（Stage 1–3）。

- 未来计划的扩展版本：
  - **v2.x**：基于KDIGO血肌酐标准的AKI定义；
  - **v3.x**：综合诊断码、肌酐变化和尿量等信息，进行更精细分层。

在项目代码中与文档中均明确标注当前使用的AKI定义版本。
## 6. 队列构建流程（拟定）

1. 所有ICU入住记录（`mimiciv_icu.icustays`）
2. 排除年龄 < 18岁
3. 仅保留首次ICU入住（每个subject_id一条）
4. 在 `admissions` + `diagnoses_icd` 中筛选住院期间有AKI相关诊断的住院
5. 排除ICU停留时间 < 6小时
6. 获得最终AKI ICU队列（N = 待填）

后续在完成SQL与数据提取后，将在此处补充具体数字。
