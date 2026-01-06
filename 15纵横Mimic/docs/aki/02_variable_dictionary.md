# AKI模块变量字典（版本 v1.0）

## 1. 人口学变量

| group      | var_name     | display_name     | source_table          | source_column  | time_window | type     | notes                              |
|-----------|--------------|------------------|-----------------------|----------------|-------------|----------|------------------------------------|
| demo      | subject_id   | 患者ID           | mimiciv_icu.icustays  | subject_id     | N/A         | id       | 研究内部唯一标识                   |
| demo      | hadm_id      | 住院ID           | mimiciv_icu.icustays  | hadm_id        | N/A         | id       |                                    |
| demo      | stay_id      | ICU入住ID        | mimiciv_icu.icustays  | stay_id        | N/A         | id       |                                    |
| demo      | age          | 年龄（岁）       | mimiciv_hosp.patients | anchor_age     | 入院时      | numeric  | 直接使用MIMIC提供的anchor_age     |
| demo      | gender       | 性别             | mimiciv_hosp.patients | gender         | N/A         | category | 'M'/'F'，前端展示为“男/女”        |
| demo      | ethnicity    | 种族             | mimiciv_hosp.admissions | ethnicity    | N/A         | category | 可根据需要合并小样本类别          |
| demo      | insurance    | 保险类型         | mimiciv_hosp.admissions | insurance    | N/A         | category | 商业保险/医保/自费等分类          |

## 2. ICU信息

| group  | var_name      | display_name  | source_table         | source_column   | time_window | type     | notes                                         |
|--------|---------------|---------------|----------------------|-----------------|-------------|----------|-----------------------------------------------|
| icu    | intime        | ICU入科时间   | mimiciv_icu.icustays | intime          | N/A         | datetime | 用作index time                               |
| icu    | outtime       | ICU出科时间   | mimiciv_icu.icustays | outtime         | N/A         | datetime |                                             |
| icu    | icu_los_hours | ICU停留时间(h)| 由 intime/outtime计算| N/A             | 全程        | numeric  | (outtime - intime) 以小时为单位              |
| icu    | first_careunit| 首次ICU类型   | mimiciv_icu.icustays | first_careunit  | N/A         | category | CCU/MICU/SICU等，前端可分组展示              |

## 3. 实验室指标（首日）

> 注意：这里先只定义“从哪里来、取哪段时间”，具体的 itemid 在步骤2再细化。

| group | var_name      | display_name          | source_table           | source_column | time_window            | type     | notes                                            |
|-------|---------------|-----------------------|------------------------|---------------|------------------------|----------|--------------------------------------------------|
| lab   | scr_firstday  | 首日血肌酐（Scr）     | mimiciv_hosp.labevents | valuenum      | ICU入科后0–24小时      | numeric  | 以特定itemid筛选肌酐记录，取中位数或最近一次    |
| lab   | bun_firstday  | 首日尿素氮（BUN）     | mimiciv_hosp.labevents | valuenum      | ICU入科后0–24小时      | numeric  | 同上                                             |
| lab   | na_firstday   | 首日血钠（Na）        | mimiciv_hosp.labevents | valuenum      | ICU入科后0–24小时      | numeric  |                                                  |
| lab   | k_firstday    | 首日血钾（K）         | mimiciv_hosp.labevents | valuenum      | ICU入科后0–24小时      | numeric  |                                                  |
| lab   | wbc_firstday  | 首日白细胞计数（WBC） | mimiciv_hosp.labevents | valuenum      | ICU入科后0–24小时      | numeric  |                                                  |
| lab   | hb_firstday   | 首日血红蛋白（Hb）    | mimiciv_hosp.labevents | valuenum      | ICU入科后0–24小时      | numeric  |                                                  |
| lab   | plt_firstday  | 首日血小板计数       | mimiciv_hosp.labevents | valuenum      | ICU入科后0–24小时      | numeric  |                                                  |

> 💡 这里最重要的是：  
> - 明确“**首日 = ICU入科后0–24h**”；  
> - 指出“具体 itemid 在步骤2 中列出”，这样 Step1 不被 itemid 细节拖住。

## 4. 结局变量

| group   | var_name          | display_name          | source_table              | source_column        | time_window  | type     | notes                                                                       |
|---------|-------------------|-----------------------|---------------------------|----------------------|--------------|----------|-----------------------------------------------------------------------------|
| outcome | icu_mortality     | ICU死亡               | 由icustays/admissions推导 | N/A                  | ICU住院期    | binary   | 若 `admissions.deathtime` 落在ICU入住至出科之间，则记为1                   |
| outcome | hosp_mortality    | 住院死亡              | mimiciv_hosp.admissions   | hospital_expire_flag | 住院期       | binary   | 直接使用 `hospital_expire_flag`                                            |
| outcome | icu_los_days      | ICU停留天数           | 由 intime/outtime 计算    | N/A                  | ICU住院期    | numeric  | ICU停留时间（小时数/24）                                                   |
| outcome | hosp_los_days     | 住院天数              | 由 admittime/dischtime 算 | N/A                  | 住院期       | numeric  | (dischtime - admittime)                                                    |
