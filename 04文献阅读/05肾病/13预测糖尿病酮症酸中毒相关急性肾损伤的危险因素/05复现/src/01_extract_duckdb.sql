-- src/01_extract_duckdb.sql
-- DuckDB SQL that expects in-memory registered tables.
-- It creates cohort & 24h-window first-measure aggregations.

WITH dka_hadm AS (
  SELECT DISTINCT hadm_id
  FROM diagnoses_icd
  WHERE icd_code IN (SELECT code FROM dka_icd_codes)
),
icu_first AS (
  SELECT subject_id, hadm_id, stay_id, intime,
         ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS rn
  FROM icustays
),
cohort AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime
  FROM icu_first i
  JOIN dka_hadm d USING (hadm_id)
  WHERE i.rn = 1
)
SELECT
  c.subject_id, c.hadm_id, c.stay_id, c.intime,
  (SELECT valuenum FROM labevents le
    WHERE le.hadm_id = c.hadm_id
      AND le.itemid IN (SELECT itemid FROM lab_bun_itemids)
      AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY le.charttime ASC
    LIMIT 1) AS bun,
  (SELECT valuenum FROM labevents le
    WHERE le.hadm_id = c.hadm_id
      AND le.itemid IN (SELECT itemid FROM lab_scr_itemids)
      AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY le.charttime ASC
    LIMIT 1) AS scr_first24h,
  (SELECT SUM(valuenum) FROM outputevents oe
    WHERE oe.stay_id = c.stay_id
      AND oe.itemid IN (SELECT itemid FROM urine_output_itemids)
      AND oe.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
  ) AS urine_24h,
  (SELECT SUM(valuenum) FROM inputevents ie
    WHERE ie.stay_id = c.stay_id
      AND ie.itemid IN (SELECT itemid FROM fluid_input_itemids)
      AND ie.starttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
  ) AS fluid_input_24h,
  (SELECT valuenum FROM chartevents ce
    WHERE ce.stay_id = c.stay_id
      AND ce.itemid IN (SELECT itemid FROM weight_itemids)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY ce.charttime ASC
    LIMIT 1) AS weight,
  (SELECT valuenum FROM chartevents ce
    WHERE ce.stay_id = c.stay_id
      AND ce.itemid IN (SELECT itemid FROM hr_itemids)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY ce.charttime ASC
    LIMIT 1) AS hr,
  (SELECT valuenum FROM chartevents ce
    WHERE ce.stay_id = c.stay_id
      AND ce.itemid IN (SELECT itemid FROM sbp_itemids)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY ce.charttime ASC
    LIMIT 1) AS sbp,
  (SELECT valuenum FROM chartevents ce
    WHERE ce.stay_id = c.stay_id
      AND ce.itemid IN (SELECT itemid FROM dbp_itemids)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY ce.charttime ASC
    LIMIT 1) AS dbp,
  (SELECT valuenum FROM chartevents ce
    WHERE ce.stay_id = c.stay_id
      AND ce.itemid IN (SELECT itemid FROM rr_itemids)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY ce.charttime ASC
    LIMIT 1) AS rr,
  (SELECT valuenum FROM chartevents ce
    WHERE ce.stay_id = c.stay_id
      AND ce.itemid IN (SELECT itemid FROM temp_itemids)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY ce.charttime ASC
    LIMIT 1) AS temperature,
  (SELECT valuenum FROM labevents le
    WHERE le.hadm_id = c.hadm_id
      AND le.itemid IN (SELECT itemid FROM lab_glucose_itemids)
      AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    ORDER BY le.charttime ASC
    LIMIT 1) AS glucose
FROM cohort c;
