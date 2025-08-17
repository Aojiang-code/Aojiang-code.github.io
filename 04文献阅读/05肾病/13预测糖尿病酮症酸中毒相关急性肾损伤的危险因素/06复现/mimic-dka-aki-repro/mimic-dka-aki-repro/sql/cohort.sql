-- sql/cohort.sql
-- Adult first ICU stays with DKA diagnosis (ICD9/10), linking hosp & icu tables.
WITH first_stay AS (
  SELECT
    ie.subject_id, ie.hadm_id, ie.stay_id,
    ie.intime AS icu_intime, ie.outtime AS icu_outtime,
    ROW_NUMBER() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS rn_hadm,
    ROW_NUMBER() OVER (PARTITION BY ie.subject_id ORDER BY ie.intime) AS rn_subject
  FROM icu.icustays ie
),
dka_codes AS (
  SELECT code, system
  FROM dka_codes_tmp
),
hadm_dka AS (
  SELECT DISTINCT dx.hadm_id
  FROM hosp.diagnoses_icd dx
  JOIN dka_codes dc ON
    (dc.system = 'ICD9CM' AND dx.icd9_code = dc.code)
    OR (dc.system = 'ICD10CM' AND dx.icd10_code = dc.code)
),
cohort0 AS (
  SELECT
    p.subject_id, a.hadm_id, fs.stay_id,
    p.anchor_age AS age,
    a.admittime, a.dischtime, fs.icu_intime, fs.icu_outtime,
    a.admission_type
  FROM hosp.patients p
  JOIN hosp.admissions a ON p.subject_id = a.subject_id
  JOIN first_stay fs ON a.hadm_id = fs.hadm_id
  JOIN hadm_dka hdk ON a.hadm_id = hdk.hadm_id
  WHERE p.anchor_age >= 18
    AND fs.rn_subject = 1  -- first ICU per patient
)
SELECT * FROM cohort0;
