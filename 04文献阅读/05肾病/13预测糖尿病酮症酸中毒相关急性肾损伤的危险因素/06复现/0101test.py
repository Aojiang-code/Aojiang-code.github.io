import duckdb

# SQL 直接查询 CSV 文件
result = duckdb.query("""
    SELECT itemid, valuenum
    FROM '/public/home/aojiang/mimic/hosp/labevents.csv'
    WHERE valuenum IS NOT NULL
    LIMIT 10
""").to_df()

print(result)
