-- sql/itemids_lookup.sql
-- Look up vitals/lab itemids by case-insensitive patterns.
-- Replace :pattern with your search string.
-- Vitals (chartevents) dictionary is in icu.d_items; labs in hosp.d_labitems.
-- Example call is implemented in src/extract.py
-- Vitals
SELECT itemid, label, category FROM icu.d_items
WHERE LOWER(label) LIKE LOWER('%' || :pattern || '%')
ORDER BY label;

-- Labs
SELECT itemid, label, fluid, category FROM hosp.d_labitems
WHERE LOWER(label) LIKE LOWER('%' || :pattern || '%')
ORDER BY label;
