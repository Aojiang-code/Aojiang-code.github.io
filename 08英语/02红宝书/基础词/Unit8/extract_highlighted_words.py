import re
import pandas as pd

# 请替换为你本地的 Markdown 文件路径
file_path = r"E:\JA\github_blogs\blogs-master\08英语\02红宝书\基础词\Unit8\Unit8_03速记.md"

# 读取 Markdown 文件
with open(file_path, encoding="utf-8") as f:
    markdown_text = f.read()

# 逐行提取已勾选的重点词
lines = markdown_text.splitlines()
highlighted_words = []

for i, line in enumerate(lines):
    if "[x]" in line and "是否需要特别注意" in line:
        for j in range(i, -1, -1):
            match = re.match(r"^####\s+\d+\.\s+\*\*(.*?)\*\*", lines[j])
            if match:
                highlighted_words.append(match.group(1))
                break

# 排序输出
highlighted_words = sorted(set(highlighted_words))
df = pd.DataFrame({"重点词汇": highlighted_words})

# 保存为 CSV 或打印
df.to_csv("Unit8_重点词汇.csv", index=False, encoding="utf-8-sig")
print("✅ 已成功提取重点词：")
print(df)
