-- backend/sql/aki_cohort.sql

-- 视情况添加 search_path
-- SET search_path TO mimiciv_icu, mimiciv_hosp, public;

DROP MATERIALIZED VIEW IF EXISTS aki_cohort CASCADE;

CREATE MATERIALIZED VIEW aki_cohort AS
WITH icu_firststay_adult AS (
    -- 1）每位患者的首次ICU入住 + 成年人
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        i.first_careunit,
        ROW_NUMBER() OVER (
            PARTITION BY i.subject_id
            ORDER BY i.intime
        ) AS rn
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_hosp.patients p
        ON i.subject_id = p.subject_id
    WHERE p.anchor_age >= 18  -- 成人
      AND p.anchor_age IS NOT NULL
),
icu_firststay_filtered AS (
    -- 2）只保留首次ICU + ICU住院时间 >= 6小时
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        f.intime,
        f.outtime,
        f.first_careunit,
        EXTRACT(EPOCH FROM (f.outtime - f.intime)) / 3600.0 AS icu_los_hours
    FROM icu_firststay_adult f
    WHERE f.rn = 1
      AND f.outtime IS NOT NULL
      AND EXTRACT(EPOCH FROM (f.outtime - f.intime)) / 3600.0 >= 6.0
),
aki_admissions AS (
    -- 3）在住院期间有 AKI ICD 诊断的住院
    SELECT DISTINCT
        d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE
        (
            (d.icd_version = 9  AND d.icd_code LIKE '584%')  -- ICD-9 AKI
         OR (d.icd_version = 10 AND d.icd_code LIKE 'N17%')  -- ICD-10 AKI
        )
),
base_cohort AS (
    -- 4）将ICU首次入住 + AKI诊断 + 患者/住院信息 合并
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.stay_id,
        icu.intime,
        icu.outtime,
        icu.first_careunit,
        icu.icu_los_hours,

        p.anchor_age AS age,
        p.gender,

        a.ethnicity,
        a.insurance,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.hospital_expire_flag
    FROM icu_firststay_filtered icu
    JOIN aki_admissions aki
        ON icu.hadm_id = aki.hadm_id
    JOIN mimiciv_hosp.patients p
        ON icu.subject_id = p.subject_id
    JOIN mimiciv_hosp.admissions a
        ON icu.hadm_id = a.hadm_id
)
SELECT
    -- ID 类变量
    subject_id,
    hadm_id,
    stay_id,

    -- 人口学
    age,
    gender,
    ethnicity,
    insurance,

    -- ICU信息
    intime,
    outtime,
    first_careunit,
    icu_los_hours,
    icu_los_hours / 24.0 AS icu_los_days,

    -- 住院信息
    admittime,
    dischtime,
    EXTRACT(EPOCH FROM (dischtime - admittime)) / 86400.0 AS hosp_los_days,

    -- 结局：ICU死亡 & 住院死亡
    CASE
        WHEN deathtime IS NOT NULL
             AND deathtime BETWEEN intime AND outtime
        THEN 1 ELSE 0
    END AS icu_mortality,

    hospital_expire_flag::int AS hosp_mortality
FROM base_cohort;
