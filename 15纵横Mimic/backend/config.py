# backend/config.py

import os

# 建议在系统或 .env 中设置这些环境变量
DB_HOST = os.getenv("MIMIC_DB_HOST", "localhost")
DB_PORT = os.getenv("MIMIC_DB_PORT", "5432")
DB_NAME = os.getenv("MIMIC_DB_NAME", "mimiciv")
DB_USER = os.getenv("MIMIC_DB_USER", "postgres")
DB_PASSWORD = os.getenv("MIMIC_DB_PASSWORD", "your_password_here")  # TODO: 修改

# 如果你以后有多个数据库，可以在这里统一管理
