# backend/aki/run_aki_pipeline.py

from backend.aki.data_loader import get_engine, load_aki_full_dataset
from backend.aki.descriptive import (
    compute_basic_stats,
    age_group_distribution,
    categorical_distribution,
    mortality_distribution,
    generate_table1,
)
from backend.aki.export_results import save_json, save_csv


def main(limit=None):
    """
    运行 AKI 描述性分析的完整流程：
    1. 从数据库读取 aki_cohort + aki_labs_firstday 并合并
    2. 计算基本统计
    3. 生成年龄、性别、死亡率等分布
    4. 生成 Table 1
    5. 把所有结果导出到 outputs/aki/
    """
    print(">>> 连接数据库并加载 AKI 数据...")
    engine = get_engine()
    df = load_aki_full_dataset(engine, limit=limit)
    print(f"[INFO] AKI dataset loaded, n = {len(df)}")

    # 1. 基本统计
    print(">>> 计算基础统计指标...")
    basic_stats = compute_basic_stats(df)
    save_json(basic_stats, "aki_basic_stats.json")

    # 2. 年龄分布
    print(">>> 计算年龄分布...")
    age_dist = age_group_distribution(df)
    save_json(age_dist, "aki_age_distribution.json")

    # 3. 性别分布
    print(">>> 计算性别分布...")
    gender_display_map = {"M": "男", "F": "女"}
    gender_dist = categorical_distribution(
        df,
        column="gender",
        categories_order=["M", "F"],
        display_map=gender_display_map,
    )
    save_json(gender_dist, "aki_gender_distribution.json")

    # 4. 住院/ICU死亡率
    print(">>> 计算死亡率...")
    mort_dist = mortality_distribution(df)
    save_json(mort_dist, "aki_mortality.json")

    # 5. Table 1
    print(">>> 生成 Table 1（整体队列）...")
    numeric_vars = [
        "age",
        "icu_los_days",
        "hosp_los_days",
        # 如果你希望，也可以加上 scr_firstday, bun_firstday 等
        # "scr_firstday", "bun_firstday",
    ]
    categorical_vars = [
        "gender",
        "ethnicity",
        "insurance",
        "icu_mortality",
        "hosp_mortality",
    ]
    table1 = generate_table1(df, numeric_vars, categorical_vars)
    save_csv(table1, "aki_table1_overall.csv")

    print(">>> AKI 描述性分析流程完成 ✅")


if __name__ == "__main__":
    # 调试时可以设置 limit=1000，正式跑通后把 limit 改成 None
    main(limit=None)
