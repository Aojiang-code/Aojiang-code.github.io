-- SELECT itemid, label, fluid, category
-- FROM mimiciv_hosp.d_labitems
-- WHERE LOWER(label) IN (
--     'creatinine',
--     'urea nitrogen',
--     'sodium',
--     'potassium',
--     'white blood cells',
--     'hemoglobin',
--     'platelet count'
-- )
-- ORDER BY label, itemid;




-- backend/sql/aki_labs_firstday.sql

DROP MATERIALIZED VIEW IF EXISTS aki_labs_firstday CASCADE;

CREATE MATERIALIZED VIEW aki_labs_firstday AS
WITH aki_base AS (
    -- 1）从已构建的 AKI 队列出发
    SELECT
        c.subject_id,
        c.hadm_id,
        c.stay_id,
        c.intime,
        c.outtime
    FROM aki_cohort c
),
lab_item_map AS (
    -- 2）手动填充或从 d_labitems 中选择需要的 itemid
    -- 下面的数字是占位符，请根据你在 d_labitems 查询的结果进行替换
    SELECT * FROM (VALUES
        (1001, 'scr_firstday'),  -- Creatinine
        (1002, 'bun_firstday'),  -- Urea nitrogen
        (1003, 'na_firstday'),   -- Sodium
        (1004, 'k_firstday'),    -- Potassium
        (1005, 'wbc_firstday'),  -- White blood cells
        (1006, 'hb_firstday'),   -- Hemoglobin
        (1007, 'plt_firstday')   -- Platelet count
    ) AS t(itemid, var_name)
),
aki_labs_raw AS (
    -- 3）提取 ICU入科后0-24小时内的实验室记录
    SELECT
        a.subject_id,
        a.hadm_id,
        a.stay_id,
        l.itemid,
        l.valuenum,
        l.charttime
    FROM aki_base a
    JOIN mimiciv_hosp.labevents l
        ON a.hadm_id = l.hadm_id
    JOIN lab_item_map m
        ON l.itemid = m.itemid
    WHERE
        l.valuenum IS NOT NULL
        AND l.charttime >= a.intime
        AND l.charttime <  a.intime + INTERVAL '24 hours'
),
aki_labs_agg AS (
    -- 4）对每个 stay_id + itemid 聚合：这里示例用中位数
    SELECT
        r.stay_id,
        r.itemid,
        percentile_disc(0.5) WITHIN GROUP (ORDER BY r.valuenum) AS lab_median
    FROM aki_labs_raw r
    GROUP BY r.stay_id, r.itemid
),
aki_labs_pivot AS (
    -- 5）将多行（不同itemid）转换为一行（列形式）
    SELECT
        b.subject_id,
        b.hadm_id,
        b.stay_id,

        -- 逐个变量 CASE WHEN + MAX 实现 pivot
        MAX(CASE WHEN m.var_name = 'scr_firstday' THEN a.lab_median END) AS scr_firstday,
        MAX(CASE WHEN m.var_name = 'bun_firstday' THEN a.lab_median END) AS bun_firstday,
        MAX(CASE WHEN m.var_name = 'na_firstday'  THEN a.lab_median END) AS na_firstday,
        MAX(CASE WHEN m.var_name = 'k_firstday'   THEN a.lab_median END) AS k_firstday,
        MAX(CASE WHEN m.var_name = 'wbc_firstday' THEN a.lab_median END) AS wbc_firstday,
        MAX(CASE WHEN m.var_name = 'hb_firstday'  THEN a.lab_median END) AS hb_firstday,
        MAX(CASE WHEN m.var_name = 'plt_firstday' THEN a.lab_median END) AS plt_firstday
    FROM aki_base b
    LEFT JOIN aki_labs_agg a
        ON b.stay_id = a.stay_id
    LEFT JOIN lab_item_map m
        ON a.itemid = m.itemid
    GROUP BY
        b.subject_id,
        b.hadm_id,
        b.stay_id
)
SELECT * FROM aki_labs_pivot;
