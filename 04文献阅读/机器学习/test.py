import re

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def update_summary(file_content, doc_title, info_dict):
    # 根据正则表达式匹配需要更新的内容
    pattern = r'(#{2,3}\s*' + re.escape(doc_title) + r'(?s).*?)\n\n'
    matches = re.findall(pattern, file_content)
    
    # 替换匹配到的内容
    for match in matches:
        updated_content = match
        
        # 将info_dict中的键值对添加到更新的内容中
        for key, value in info_dict.items():
            updated_content = re.sub(r'^###\s*' + re.escape(key) + r'\s*:\s*(.*?)$', r'### ' + key + r': ' + value, updated_content, flags=re.MULTILINE)
        
        # 将更新后的内容替换到原始文件内容中
        file_content = re.sub(re.escape(match.strip()), updated_content.strip(), file_content)

    return file_content

# 读取汇总.md文件的内容
summary_content = read_file('汇总.md')

# 读取文档1.md文件的内容
doc1_content = read_file('文档1.md')

# 定义文档1的标题和信息字典
doc1_title = '文献标题：hhh'
doc1_info_dict = {
    '发表时间': '2020年',
    '阅读时间': '2024年',
    '分区': '1111',
    '国家': '111',
    '作者': '111',
    '影响因子': '111',
    '方法': '111',
    '创新点': '111',
    '不足之处': '111',
    '可借鉴之处': '111'
}

# 更新汇总.md的内容
summary_content = update_summary(summary_content, doc1_title, doc1_info_dict)

# 读取文档2.md文件的内容
doc2_content = read_file('文档2.md')

# 定义文档2的标题和信息字典
doc2_title = '文献标题：aaa'
doc2_info_dict = {
    '发表时间': '2021年',
    '阅读时间': '2025年',
    '分区': '222',
    '国家': '222',
    '作者': '222',
    '影响因子': '222',
    '方法': '222',
    '创新点': '222',
    '不足之处': '222',
    '可借鉴之处': '222'
}

# 更新汇总.md的内容
summary_content = update_summary(summary_content, doc2_title, doc2_info_dict)

# 将更新后的内容写入汇总.md文件
with open('汇总.md', 'w', encoding='utf-8') as file:
    file.write(summary_content)
