import os
import re
import pandas as pd

# 定义提取信息的函数
def extract_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 使用正则表达式提取信息
    title = re.search(r"标题\s*\|\s*(.+)", content)
    authors = re.search(r"作者\s*\|\s*(.+)", content)
    date = re.search(r"发表时间\s*\|\s*(\d{4}-\d{2}-\d{2})", content)
    country = re.search(r"国家\s*\|\s*(.+)", content)
    quartile = re.search(r"分区\s*\|\s*(.+)", content)
    impact_factor = re.search(r"影响因子\s*\|\s*(\d+\.\d+)", content)
    abstract = re.search(r"摘要\s*\|\s*(.+)", content)
    keywords = re.search(r"关键词\s*\|\s*(.+)", content)
    journal = re.search(r"期刊名称\s*\|\s*(.+)", content)
    volume_issue = re.search(r"卷号/期号\s*\|\s*(.+)", content)
    doi = re.search(r"DOI\s*\|\s*(.+)", content)
    method = re.search(r"研究方法\s*\|\s*(.+)", content)
    data_source = re.search(r"数据来源\s*\|\s*(.+)", content)
    results = re.search(r"研究结果\s*\|\s*(.+)", content)
    conclusion = re.search(r"研究结论\s*\|\s*(.+)", content)
    significance = re.search(r"研究意义\s*\|\s*(.+)", content)
    start_time = re.search(r"阅读开始时间\s*\|\s*(\d{8}\s*\d{2})", content)
    end_time = re.search(r"阅读结束时间\s*\|\s*(\d{8}\s*\d{2})", content)
    moment = re.search(r"时刻\s*\|\s*(.+)", content)
    weekday = re.search(r"星期\s*\|\s*(.+)", content)
    weather = re.search(r"天气\s*\|\s*(.+)", content)
    
    # 构建返回的字典，只包含成功提取到的信息
    info = {}
    if title:
        info["标题"] = title.group(1).strip()
    if authors:
        info["作者"] = authors.group(1).strip()
    if date:
        info["发表时间"] = date.group(1).strip()
    if country:
        info["国家"] = country.group(1).strip()
    if quartile:
        info["分区"] = quartile.group(1).strip()
    if impact_factor:
        info["影响因子"] = impact_factor.group(1).strip()
    if abstract:
        info["摘要"] = abstract.group(1).strip()
    if keywords:
        info["关键词"] = keywords.group(1).strip()
    if journal:
        info["期刊名称"] = journal.group(1).strip()
    if volume_issue:
        info["卷号/期号"] = volume_issue.group(1).strip()
    if doi:
        info["DOI"] = doi.group(1).strip()
    if method:
        info["研究方法"] = method.group(1).strip()
    if data_source:
        info["数据来源"] = data_source.group(1).strip()
    if results:
        info["研究结果"] = results.group(1).strip()
    if conclusion:
        info["研究结论"] = conclusion.group(1).strip()
    if significance:
        info["研究意义"] = significance.group(1).strip()
    if start_time:
        info["阅读开始时间"] = start_time.group(1).strip()
    if end_time:
        info["阅读结束时间"] = end_time.group(1).strip()
    if moment:
        info["时刻"] = moment.group(1).strip()
    if weekday:
        info["星期"] = weekday.group(1).strip()
    if weather:
        info["天气"] = weather.group(1).strip()
    
    return info

# 遍历文件夹并提取信息
def extract_all_info(folder_path):
    all_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower() == "readme.md":
                file_path = os.path.join(root, file)
                info = extract_info(file_path)
                if info:  # 只有当提取到信息时才添加到列表中
                    all_info.append(info)
    return all_info

# 转换为DataFrame并保存为CSV
folder_path = r"E:\JA\github_blogs\blogs-master\04文献阅读\05肾病"
output_folder = r"E:\JA\github_blogs\blogs-master\04文献阅读\05肾病"  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

all_info = extract_all_info(folder_path)
if all_info:  # 只有当提取到信息时才保存CSV文件
    df = pd.DataFrame(all_info)
    output_file = os.path.join(output_folder, "文献信息.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"信息已提取并保存到 {output_file}")
else:
    print("未找到任何有效信息，未生成CSV文件。")