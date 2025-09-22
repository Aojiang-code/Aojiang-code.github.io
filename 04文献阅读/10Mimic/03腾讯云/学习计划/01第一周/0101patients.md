



```python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# 读取数据
patients = pd.read_csv("/workspace/mimic-iv-data/3.1/hosp/patients.csv")



```








```python
# 添加是否死亡字段
patients['is_dead'] = patients['dod'].notnull().astype(int)

# 添加年龄分组字段
patients['anchor_age_group'] = pd.cut(
    patients['anchor_age'],
    bins=[0, 18, 30, 45, 60, 75, 89, 200],
    labels=["0-18", "19-30", "31-45", "46-60", "61-75", "76-89", "90+"]
)

# 可视化 1：年龄分布
plt.figure(figsize=(10, 5))
sns.histplot(patients['anchor_age'], bins=50, kde=True)
plt.title("Age Distribution (Anchor Age)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()



```








```python
# 可视化 2：性别分布
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=patients)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()



```








```python
# 可视化 3：是否死亡分布
plt.figure(figsize=(6, 4))
sns.countplot(x='is_dead', data=patients)
plt.title("Patient Mortality Status")
plt.xlabel("Death Status (0 = Alive, 1 = Deceased)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()



```








```python
# 可视化 4：不同性别的死亡率
plt.figure(figsize=(6, 4))
sns.barplot(data=patients, x='gender', y='is_dead')
plt.title("Mortality Rate by Gender")
plt.ylabel("Mortality Rate")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.show()



```








```python
# 可视化 5：不同年龄段的死亡率
age_death_rate = patients.groupby('anchor_age_group')['is_dead'].mean().reset_index()
plt.figure(figsize=(8, 4))
sns.barplot(data=age_death_rate, x='anchor_age_group', y='is_dead')
plt.title("Mortality Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Mortality Rate")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.show()



```








```python

# 构建年龄×性别的死亡率统计表
summary_table = patients.groupby(['anchor_age_group', 'gender'])['is_dead'].agg(['count', 'sum', 'mean']).reset_index()
summary_table.rename(columns={"count": "Total", "sum": "Deaths", "mean": "Mortality Rate"}, inplace=True)

import ace_tools as tools; tools.display_dataframe_to_user(name="Mortality Summary by Age Group and Gender", dataframe=summary_table)
```








```python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 读取上传的文件
patients = pd.read_csv("/workspace/mimic-iv-data/3.1/hosp/patients.csv")

# 设置年龄分段（与参考代码保持一致）
age_bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
age_labels = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49', '50~59',
              '60~69', '70~79', '80~89', '>=90']

# 替换缺失年龄为中位数或删除缺失（此处假设 anchor_age 有效）
patients = patients.dropna(subset=['anchor_age'])
patients['age_group'] = pd.cut(patients['anchor_age'], bins=age_bins, labels=age_labels, right=False)

# 统计各年龄段男女人数
gender_map = {'M': 'Male', 'F': 'Female'}
patients['gender'] = patients['gender'].map(gender_map)
age_gender_counts = pd.crosstab(patients['age_group'], patients['gender'])

# 准备绘图数据
categories = age_gender_counts.index.tolist()
values_left = age_gender_counts['Male'].tolist() if 'Male' in age_gender_counts else [0] * len(categories)
values_right = age_gender_counts['Female'].tolist() if 'Female' in age_gender_counts else [0] * len(categories)
total_values = [l + r for l, r in zip(values_left, values_right)]

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)  # 整体背景透明
ax.patch.set_alpha(0)   # 坐标区背景透明
ax.grid(False)          # 禁用网格线

# 创建颜色映射（总数上色）
norm = mcolors.Normalize(vmin=min(total_values), vmax=max(total_values))
cmap = cm.RdYlGn_r
colors = [cmap(norm(value)) for value in total_values]

# 绘制柱状图
bars = ax.bar(categories, total_values, color=colors)

# 添加平滑曲线（三次样条插值）
x = np.arange(len(categories))
f = interp1d(x, total_values, kind='cubic')
x_smooth = np.linspace(0, len(categories) - 1, 300)
y_smooth = f(x_smooth)
ax.plot(x_smooth, y_smooth, "r--", linewidth=2, label='Trend Line')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height}', ha='center', va='bottom')

# 设置标签与标题（英文）
ax.set_ylabel('Number of Patients')
ax.set_xlabel('Age Group')
ax.set_title('Patient Count by Age Group')
ax.legend()

# 保存透明背景图像
plt.savefig('/workspace/output/01patients/0101age_distribution_plot.png', transparent=True)
plt.tight_layout()
plt.show()

```








```python

import matplotlib.pyplot as plt
import numpy as np

# 准备男女数量数据（前面已经从 patients 中获取过）
categories = age_gender_counts.index.tolist()
values_left = age_gender_counts['Male'].tolist() if 'Male' in age_gender_counts else [0] * len(categories)
values_right = age_gender_counts['Female'].tolist() if 'Female' in age_gender_counts else [0] * len(categories)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)  # 背景透明
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制男性（右侧）条形
ax.barh(categories, values_left, align='center', color='skyblue', label='Male')

# 绘制女性（左侧）条形（取负值）
ax.barh(categories, [-v for v in values_right], align='center', color='lightgreen', label='Female')

# 添加男性数据标签
for i, v in enumerate(values_left):
    ax.text(v + 2, i, str(v), va='center')

# 添加女性数据标签
for i, v in enumerate(values_right):
    ax.text(-v - 2, i, str(v), va='center', ha='right')

# 添加中心线
ax.axvline(x=0, color='black', linewidth=0.5)

# 添加均值线
left_mean = np.mean(values_left)
right_mean = np.mean(values_right)
ax.axvline(left_mean, color='blue', linestyle='--', linewidth=1.5, label='Mean (Male)')
ax.axvline(-right_mean, color='green', linestyle='--', linewidth=1.5, label='Mean (Female)')

# 设置刻度和标签
ax.set_xticks(np.arange(-max(values_right + values_left) - 10, max(values_right + values_left) + 11, 50))
ax.set_xticklabels([str(abs(x)) for x in ax.get_xticks()])
ax.set_xlabel('Number of Patients')
ax.set_ylabel('Age Group')
ax.set_title('Patient Count by Age Group and Gender')

# 添加图例
ax.legend()

# 保存图像
plt.savefig('//workspace/output/01patients/0201age_gender_distribution_plot.png', transparent=True)
plt.tight_layout()
plt.show()

```








```python
# 计算总人数与比例
total_per_age = [m + f for m, f in zip(values_left, values_right)]
male_ratios = [m / total * 100 if total > 0 else 0 for m, total in zip(values_left, total_per_age)]
female_ratios = [f / total * 100 if total > 0 else 0 for f, total in zip(values_right, total_per_age)]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制条形图（与上一步相同）
ax.barh(categories, values_left, align='center', color='skyblue', label='Male')
ax.barh(categories, [-v for v in values_right], align='center', color='lightgreen', label='Female')

# 添加数据标签（人数）和比例标签（%）
for i, (mv, fv, mp, fp) in enumerate(zip(values_left, values_right, male_ratios, female_ratios)):
    ax.text(mv + 2, i, f"{mv} ({mp:.1f}%)", va='center', fontsize=9)
    ax.text(-fv - 2, i, f"{fv} ({fp:.1f}%)", va='center', ha='right', fontsize=9)

# 中心线与均值线
ax.axvline(x=0, color='black', linewidth=0.5)
ax.axvline(np.mean(values_left), color='blue', linestyle='--', linewidth=1.5, label='Mean (Male)')
ax.axvline(-np.mean(values_right), color='green', linestyle='--', linewidth=1.5, label='Mean (Female)')

# 坐标轴设置
xticks = np.arange(-max(values_right + values_left) - 10, max(values_right + values_left) + 11, 50)
ax.set_xticks(xticks)
ax.set_xticklabels([str(abs(x)) for x in xticks])
ax.set_xlabel('Number of Patients')
ax.set_ylabel('Age Group')
ax.set_title('Patient Count and Gender Ratio by Age Group')

# 图例
ax.legend()

# 保存图像
plt.savefig('/workspace/output/01patients/0202age_gender_ratio_distribution_plot.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python
# 再次创建图形，补充文本统计信息
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制条形图
bars_male = ax.barh(categories, values_left, align='center', color='skyblue', label='Male')
bars_female = ax.barh(categories, [-v for v in values_right], align='center', color='lightgreen', label='Female')

# 添加人数和比例标签
for i, (mv, fv, mp, fp) in enumerate(zip(values_left, values_right, male_ratios, female_ratios)):
    ax.text(mv + 2, i, f"{mv} ({mp:.1f}%)", va='center', fontsize=9)
    ax.text(-fv - 2, i, f"{fv} ({fp:.1f}%)", va='center', ha='right', fontsize=9)

# 添加中心线和均值线
ax.axvline(x=0, color='black', linewidth=0.5)
ax.axvline(np.mean(values_left), color='blue', linestyle='--', linewidth=1.5, label='Mean (Male)')
ax.axvline(-np.mean(values_right), color='green', linestyle='--', linewidth=1.5, label='Mean (Female)')

# 添加文本统计信息框
left_mean = np.mean(values_left)
right_mean = np.mean(values_right)
left_std = np.std(values_left)
right_std = np.std(values_right)

stats_text = (
    f"Mean (Male): {left_mean:.2f}\n"
    f"Mean (Female): {right_mean:.2f}\n"
    f"SD (Male): {left_std:.2f}\n"
    f"SD (Female): {right_std:.2f}"
)

ax.text(
    0.97, 0.01, stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# 设置轴标签与标题
xticks = np.arange(-max(values_right + values_left) - 10, max(values_right + values_left) + 11, 50)
ax.set_xticks(xticks)
ax.set_xticklabels([str(abs(x)) for x in xticks])
ax.set_xlabel('Number of Patients')
ax.set_ylabel('Age Group')
ax.set_title('Patient Count and Gender Ratio by Age Group')

# 图例
ax.legend()

# 保存图像
plt.savefig('/workspace/output/01patients/0203age_gender_ratio_with_stats_plot.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python
# 绘图：特别调整“20~29岁”组别的标签位置与样式为黑色并向内偏移
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制男女条形图
bars_male = ax.barh(categories, values_left, align='center', color='skyblue', label='Male')
bars_female = ax.barh(categories, [-v for v in values_right], align='center', color='lightgreen', label='Female')

# 添加标签，特调“20~29”组
for i, (mv, fv, mp, fp) in enumerate(zip(values_left, values_right, male_ratios, female_ratios)):
    age_group = categories[i]
    
    if age_group == '20~29':
        ax.text(mv - 500, i, f"{mv} ({mp:.1f}%)", va='center', ha='right', fontsize=9, color='black')
        ax.text(-fv + 500, i, f"{fv} ({fp:.1f}%)", va='center', ha='left', fontsize=9, color='black')
    else:
        ax.text(mv + 2, i, f"{mv} ({mp:.1f}%)", va='center', fontsize=9, color='black')
        ax.text(-fv - 2, i, f"{fv} ({fp:.1f}%)", va='center', ha='right', fontsize=9, color='black')

# 添加中心线与均值虚线
ax.axvline(x=0, color='black', linewidth=0.5)
ax.axvline(np.mean(values_left), color='blue', linestyle='--', linewidth=1.5, label='Mean (Male)')
ax.axvline(-np.mean(values_right), color='green', linestyle='--', linewidth=1.5, label='Mean (Female)')

# 添加文本统计信息框
left_mean = np.mean(values_left)
right_mean = np.mean(values_right)
left_std = np.std(values_left)
right_std = np.std(values_right)

stats_text = (
    f"Mean (Male): {left_mean:.2f}\n"
    f"Mean (Female): {right_mean:.2f}\n"
    f"SD (Male): {left_std:.2f}\n"
    f"SD (Female): {right_std:.2f}"
)

ax.text(
    0.97, 0.01, stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# 设置坐标轴和标题
xticks = np.arange(-max(values_right + values_left) - 10, max(values_right + values_left) + 11, 50)
ax.set_xticks(xticks)
ax.set_xticklabels([str(abs(x)) for x in xticks])
ax.set_xlabel('Number of Patients')
ax.set_ylabel('Age Group')
ax.set_title('Patient Count and Gender Ratio by Age Group')

# 添加图例
ax.legend()

# 保存图像
# plt.savefig('/mnt/data/fixed_overlap_20_29_labels.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python
# 修改字体为默认英文，不再使用中文字体设置
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 准备数据
x = np.arange(len(categories))
male_values = np.array(values_left)
female_values = np.array(values_right)

# 平滑函数
def smooth_curve(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 柱状图参数
bar_width = 0.35
index = np.arange(len(categories))

# 绘制柱状图（红：女性，绿：男性）
ax.bar(index - bar_width / 2, female_values, width=bar_width, color='red', alpha=0.3, label='Female')
ax.bar(index + bar_width / 2, male_values, width=bar_width, color='green', alpha=0.3, label='Male')

# 绘制平滑曲线
x_f_smooth, y_f_smooth = smooth_curve(x, female_values)
x_m_smooth, y_m_smooth = smooth_curve(x, male_values)
ax.plot(x_f_smooth, y_f_smooth, 'r-', linewidth=3)
ax.plot(x_m_smooth, y_m_smooth, 'g-', linewidth=3)

# 添加数值标签
for i, val in enumerate(female_values):
    ax.text(i - bar_width / 2, val + 2, str(val), ha='center', va='bottom', fontsize=9, color='red')
for i, val in enumerate(male_values):
    ax.text(i + bar_width / 2, val + 2, str(val), ha='center', va='bottom', fontsize=9, color='green')

# 设置英文标题与标签
ax.set_title('Patient Count by Age Group and Gender', fontsize=16)
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_xticks(index)
ax.set_xticklabels(categories)

# 图例
ax.legend()

# 保存图像
plt.savefig('/workspace/output/01patients/0299age_gender_distribution_en.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 重新加载上传的患者数据
patients = pd.read_csv("/workspace/mimic-iv-data/3.1/hosp/patients.csv")
patients = patients.dropna(subset=['anchor_age'])

# 年龄分段
age_bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
age_labels = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49', '50~59',
              '60~69', '70~79', '80~89', '>=90']
patients['age_group'] = pd.cut(patients['anchor_age'], bins=age_bins, labels=age_labels, right=False)

# 是否死亡
patients['is_dead'] = patients['dod'].notnull().astype(int)

# 统计每个年龄段的总人数和死亡人数
total_counts = patients['age_group'].value_counts().sort_index()
death_counts = patients[patients['is_dead'] == 1]['age_group'].value_counts().sort_index()

# 过滤掉总人数为 0 的年龄段
filtered_categories = [label for label in age_labels if total_counts.get(label, 0) > 0]
total_values = [total_counts.get(label, 0) for label in filtered_categories]
death_values = [death_counts.get(label, 0) for label in filtered_categories]

# 设置颜色映射（根据总人数）
norm = mcolors.Normalize(vmin=min(total_values), vmax=max(total_values))
cmap = cm.RdYlGn_r
colors = [cmap(norm(value)) for value in total_values]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制总人口柱状图（浅色）
bars_total = ax.bar(filtered_categories, total_values, color=colors, alpha=0.3, label='Total Patients')

# 绘制死亡人口柱状图（深色）
bars_death = ax.bar(filtered_categories, death_values, color=colors, alpha=0.9, label='Deaths')

# 平滑曲线（基于总人口）
x = np.arange(len(filtered_categories))
f = interp1d(x, total_values, kind='cubic')
x_smooth = np.linspace(0, len(filtered_categories) - 1, 300)
y_smooth = f(x_smooth)
ax.plot(x_smooth, y_smooth, "r--", linewidth=2, label='Total Trend')

# 添加数值标签（死亡人数 + 总人数）
for i, (bar_t, bar_d) in enumerate(zip(bars_total, bars_death)):
    t_height = bar_t.get_height()
    d_height = bar_d.get_height()
    ax.text(bar_t.get_x() + bar_t.get_width() / 2., t_height + 2, f' {int(t_height)}', ha='center', va='bottom', fontsize=8)
    ax.text(bar_d.get_x() + bar_d.get_width() / 2., d_height + 2, f' {int(d_height)}', ha='center', va='bottom', fontsize=8, color='black')

# 标签与标题
ax.set_ylabel('Number of Patients')
ax.set_xlabel('Age Group')
ax.set_title('Total and Deceased Patients by Age Group')
ax.legend()

# 保存图像
plt.savefig('/workspace/output/01patients/0301filtered_age_group_total_vs_deaths.png', transparent=True)
plt.tight_layout()
plt.show()

```








```python
# 分别为总人数和死亡人数创建两个不同的颜色映射
norm_total = mcolors.Normalize(vmin=min(total_values), vmax=max(total_values))
norm_death = mcolors.Normalize(vmin=min(death_values), vmax=max(death_values))
cmap_total = cm.RdYlGn_r  # 高为红，低为绿（总人数）
cmap_death = cm.RdYlGn_r  # 高为红，低为绿（死亡人数）

colors_total = [cmap_total(norm_total(v)) for v in total_values]
colors_death = [cmap_death(norm_death(v)) for v in death_values]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制总人口柱状图（颜色1）
bars_total = ax.bar(filtered_categories, total_values, color=colors_total, alpha=0.3, label='Total Patients')

# 绘制死亡人口柱状图（颜色2）
bars_death = ax.bar(filtered_categories, death_values, color=colors_death, alpha=0.9, label='Deaths')

# 平滑曲线（基于总人数）
x = np.arange(len(filtered_categories))
f = interp1d(x, total_values, kind='cubic')
x_smooth = np.linspace(0, len(filtered_categories) - 1, 300)
y_smooth = f(x_smooth)
ax.plot(x_smooth, y_smooth, "r--", linewidth=2, label='Total Trend')

# 添加数值标签（死亡人数 + 总人数）
for i, (bar_t, bar_d) in enumerate(zip(bars_total, bars_death)):
    t_height = bar_t.get_height()
    d_height = bar_d.get_height()
    ax.text(bar_t.get_x() + bar_t.get_width() / 2., t_height + 2, f' {int(t_height)}', ha='center', va='bottom', fontsize=8)
    ax.text(bar_d.get_x() + bar_d.get_width() / 2., d_height + 2, f' {int(d_height)}', ha='center', va='bottom', fontsize=8, color='black')

# 设置标签与标题
ax.set_ylabel('Number of Patients')
ax.set_xlabel('Age Group')
ax.set_title('Total and Deceased Patients by Age Group (Separate Color Mapping)')
ax.legend()

# 保存图像
plt.savefig('/workspace/output/01patients/0302age_group_total_vs_deaths_separate_colors.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 仅绘制死亡人口分布图
# 使用之前计算的 filtered_categories 和 death_values

# 设置颜色映射（根据死亡人数）
norm = mcolors.Normalize(vmin=min(death_values), vmax=max(death_values))
cmap = cm.Reds
colors = [cmap(norm(value)) for value in death_values]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制死亡人数柱状图
bars_death_only = ax.bar(filtered_categories, death_values, color=colors, alpha=0.9, label='Deaths')

# 添加平滑曲线
x = np.arange(len(filtered_categories))
f = interp1d(x, death_values, kind='cubic')
x_smooth = np.linspace(0, len(filtered_categories) - 1, 300)
y_smooth = f(x_smooth)
ax.plot(x_smooth, y_smooth, "r--", linewidth=2, label='Death Trend')

# 添加死亡人数标签
for bar in bars_death_only:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 2, f'{int(height)}', ha='center', va='bottom', fontsize=8, color='black')

# 设置标签与标题
ax.set_ylabel('Number of Deaths')
ax.set_xlabel('Age Group')
ax.set_title('Deceased Patients by Age Group')
ax.legend()

# 保存图像
plt.savefig('/workspace/output/01patients/0401age_group_deaths_only.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python

# 使用 RdYlGn_r 颜色映射：高值为红，低值为绿
norm = mcolors.Normalize(vmin=min(death_values), vmax=max(death_values))
cmap = cm.RdYlGn_r
colors = [cmap(norm(value)) for value in death_values]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制死亡人数柱状图（使用红-绿渐变）
bars_death_only = ax.bar(filtered_categories, death_values, color=colors, alpha=0.9, label='Deaths')

# 添加趋势线
x = np.arange(len(filtered_categories))
f = interp1d(x, death_values, kind='cubic')
x_smooth = np.linspace(0, len(filtered_categories) - 1, 300)
y_smooth = f(x_smooth)
ax.plot(x_smooth, y_smooth, "r--", linewidth=2, label='Death Trend')

# 添加数值标签
for bar in bars_death_only:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 2, f'{int(height)}', ha='center', va='bottom', fontsize=8, color='black')

# 设置标签与标题
ax.set_ylabel('Number of Deaths')
ax.set_xlabel('Age Group')
ax.set_title('Deceased Patients by Age Group (Color = Magnitude)')
ax.legend()

# 保存图像
plt.savefig('/workspace/output/01patients/0402age_group_deaths_colored_red_green.png', transparent=True)
plt.tight_layout()
plt.show()

```








```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
patients = pd.read_csv("/workspace/mimic-iv-data/3.1/hosp/patients.csv")
patients = patients.dropna(subset=['anchor_age'])
patients['gender'] = patients['gender'].map({'M': 'Male', 'F': 'Female'})

# 年龄分组
age_bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
age_labels = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49', '50~59',
              '60~69', '70~79', '80~89', '>=90']
patients['age_group'] = pd.cut(patients['anchor_age'], bins=age_bins, labels=age_labels, right=False)

# 标记是否死亡
patients['is_dead'] = patients['dod'].notnull().astype(int)
dead_patients = patients[patients['is_dead'] == 1]

# 构建交叉表：死亡人数 × 年龄段 × 性别
death_counts_gender = pd.crosstab(dead_patients['age_group'], dead_patients['gender'])

# 构建绘图数据并过滤总死亡人数为0的年龄段
categories = []
values_left = []
values_right = []
male_ratios = []
female_ratios = []

for age_group in age_labels:
    male = death_counts_gender.loc[age_group, 'Male'] if age_group in death_counts_gender.index and 'Male' in death_counts_gender.columns else 0
    female = death_counts_gender.loc[age_group, 'Female'] if age_group in death_counts_gender.index and 'Female' in death_counts_gender.columns else 0
    total = male + female
    if total > 0:
        categories.append(age_group)
        values_left.append(male)
        values_right.append(female)
        male_ratios.append(male / total * 100)
        female_ratios.append(female / total * 100)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制左右条形图
bars_male = ax.barh(categories, values_left, align='center', color='green', label='Male Deaths')
bars_female = ax.barh(categories, [-v for v in values_right], align='center', color='red', label='Female Deaths')

# 添加标签，避免重叠处理
for i, (age_group, mv, fv, mp, fp) in enumerate(zip(categories, values_left, values_right, male_ratios, female_ratios)):
    if age_group == '10~19':
        ax.text(mv + 20, i, f"{mv} ({mp:.1f}%)", va='center', ha='left', fontsize=9, color='black')
        ax.text(-fv - 40, i, f"{fv} ({fp:.1f}%)", va='center', ha='right', fontsize=9, color='black')
    elif age_group == '20~29':
        ax.text(mv - 500, i, f"{mv} ({mp:.1f}%)", va='center', ha='right', fontsize=9, color='black')
        ax.text(-fv + 500, i, f"{fv} ({fp:.1f}%)", va='center', ha='left', fontsize=9, color='black')
    else:
        ax.text(mv + 2, i, f"{mv} ({mp:.1f}%)", va='center', fontsize=9, color='black')
        ax.text(-fv - 2, i, f"{fv} ({fp:.1f}%)", va='center', ha='right', fontsize=9, color='black')

# 添加中心线与均值线
ax.axvline(x=0, color='black', linewidth=0.5)
ax.axvline(np.mean(values_left), color='blue', linestyle='--', linewidth=1.5, label='Mean (Male Deaths)')
ax.axvline(-np.mean(values_right), color='purple', linestyle='--', linewidth=1.5, label='Mean (Female Deaths)')

# 添加统计信息框
left_mean = np.mean(values_left)
right_mean = np.mean(values_right)
left_std = np.std(values_left)
right_std = np.std(values_right)

stats_text = (
    f"Mean (Male): {left_mean:.2f}\n"
    f"Mean (Female): {right_mean:.2f}\n"
    f"SD (Male): {left_std:.2f}\n"
    f"SD (Female): {right_std:.2f}"
)

ax.text(
    0.97, 0.01, stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# 横坐标刻度：-5000 到 5000，每 500 一格，显示为绝对值
xtick_values = np.arange(-5000, 5001, 500)
xtick_labels = [str(abs(x)) for x in xtick_values]
ax.set_xticks(xtick_values)
ax.set_xticklabels(xtick_labels)

# 其余设置
ax.set_xlabel('')
ax.set_ylabel('Age Group')
ax.set_title('Deaths by Age Group and Gender (Filtered Zero)')
ax.legend()

# 保存图像
# plt.savefig('/mnt/data/final_gender_death_filtered_abs_ticks_500_full.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
patients = pd.read_csv("/workspace/mimic-iv-data/3.1/hosp/patients.csv")
patients = patients.dropna(subset=['anchor_age'])
patients['gender'] = patients['gender'].map({'M': 'Male', 'F': 'Female'})

# 年龄分组
age_bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
age_labels = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49', '50~59',
              '60~69', '70~79', '80~89', '>=90']
patients['age_group'] = pd.cut(patients['anchor_age'], bins=age_bins, labels=age_labels, right=False)

# 标记是否死亡
patients['is_dead'] = patients['dod'].notnull().astype(int)
dead_patients = patients[patients['is_dead'] == 1]

# 构建交叉表：死亡人数 × 年龄段 × 性别
death_counts_gender = pd.crosstab(dead_patients['age_group'], dead_patients['gender'])

# 构建绘图数据并过滤总死亡人数为0的年龄段
categories = []
values_left = []
values_right = []
male_ratios = []
female_ratios = []

for age_group in age_labels:
    male = death_counts_gender.loc[age_group, 'Male'] if age_group in death_counts_gender.index and 'Male' in death_counts_gender.columns else 0
    female = death_counts_gender.loc[age_group, 'Female'] if age_group in death_counts_gender.index and 'Female' in death_counts_gender.columns else 0
    total = male + female
    if total > 0:
        categories.append(age_group)
        values_left.append(male)
        values_right.append(female)
        male_ratios.append(male / total * 100)
        female_ratios.append(female / total * 100)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 绘制左右条形图
bars_male = ax.barh(categories, values_left, align='center', color='green', label='Male Deaths')
bars_female = ax.barh(categories, [-v for v in values_right], align='center', color='red', label='Female Deaths')

# 添加标签，避免重叠处理
for i, (age_group, mv, fv, mp, fp) in enumerate(zip(categories, values_left, values_right, male_ratios, female_ratios)):
    if age_group == '10~19':
        ax.text(mv + 20, i, f"{mv} ({mp:.1f}%)", va='center', ha='left', fontsize=9, color='black')
        ax.text(-fv - 40, i, f"{fv} ({fp:.1f}%)", va='center', ha='right', fontsize=9, color='black')
    elif age_group == '20~29':
        ax.text(mv + 1000, i, f"{mv} ({mp:.1f}%)", va='center', ha='right', fontsize=9, color='black')
        ax.text(-fv - 1000, i, f"{fv} ({fp:.1f}%)", va='center', ha='left', fontsize=9, color='black')
    elif age_group == '80~89':
        ax.text(mv - 20, i, f"{mv} ({mp:.1f}%)", va='center', ha='right', fontsize=9, color='black')
        ax.text(-fv + 20, i, f"{fv} ({fp:.1f}%)", va='center', ha='left', fontsize=9, color='black')
    else:
        ax.text(mv + 2, i, f"{mv} ({mp:.1f}%)", va='center', fontsize=9, color='black')
        ax.text(-fv - 2, i, f"{fv} ({fp:.1f}%)", va='center', ha='right', fontsize=9, color='black')

# 添加中心线与均值线
ax.axvline(x=0, color='black', linewidth=0.5)
ax.axvline(np.mean(values_left), color='blue', linestyle='--', linewidth=1.5, label='Mean (Male Deaths)')
ax.axvline(-np.mean(values_right), color='purple', linestyle='--', linewidth=1.5, label='Mean (Female Deaths)')

# 添加统计信息框
left_mean = np.mean(values_left)
right_mean = np.mean(values_right)
left_std = np.std(values_left)
right_std = np.std(values_right)

stats_text = (
    f"Mean (Male): {left_mean:.2f}\n"
    f"Mean (Female): {right_mean:.2f}\n"
    f"SD (Male): {left_std:.2f}\n"
    f"SD (Female): {right_std:.2f}"
)

ax.text(
    0.97, 0.01, stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# 横坐标刻度：-5000 到 5000，每 500 一格，显示为绝对值
xtick_values = np.arange(-5000, 5001, 500)
xtick_labels = [str(abs(x)) for x in xtick_values]
ax.set_xticks(xtick_values)
ax.set_xticklabels(xtick_labels)

# 其余设置
ax.set_xlabel('')
ax.set_ylabel('Age Group')
ax.set_title('Deaths by Age Group and Gender (Filtered Zero)')
ax.legend()

# 保存图像
# plt.savefig('/mnt/data/final_gender_death_filtered_abs_ticks_500_full.png', transparent=True)
plt.tight_layout()
plt.show()


```






```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 读取数据
patients = pd.read_csv("/workspace/mimic-iv-data/3.1/hosp/patients.csv")
patients = patients.dropna(subset=['anchor_age'])
patients['gender'] = patients['gender'].map({'M': 'Male', 'F': 'Female'})

# 年龄分组
age_bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
age_labels = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49', '50~59',
              '60~69', '70~79', '80~89', '>=90']
patients['age_group'] = pd.cut(patients['anchor_age'], bins=age_bins, labels=age_labels, right=False)

# 标记是否死亡
patients['is_dead'] = patients['dod'].notnull().astype(int)
dead_patients = patients[patients['is_dead'] == 1]

# 构建交叉表：死亡人数 × 年龄段 × 性别
death_counts_gender = pd.crosstab(dead_patients['age_group'], dead_patients['gender'])

# 构建绘图数据并过滤总死亡人数为0的年龄段
categories = []
values_left = []
values_right = []

for age_group in age_labels:
    male = death_counts_gender.loc[age_group, 'Male'] if age_group in death_counts_gender.index and 'Male' in death_counts_gender.columns else 0
    female = death_counts_gender.loc[age_group, 'Female'] if age_group in death_counts_gender.index and 'Female' in death_counts_gender.columns else 0
    total = male + female
    if total > 0:
        categories.append(age_group)
        values_left.append(male)
        values_right.append(female)

# 准备数据
x = np.arange(len(categories))
male_values = np.array(values_left)
female_values = np.array(values_right)

# 平滑函数
def smooth_curve(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 柱状图参数
bar_width = 0.35
index = np.arange(len(categories))

# 绘制柱状图（红：女性，绿：男性）
ax.bar(index - bar_width / 2, female_values, width=bar_width, color='red', alpha=0.3, label='Female')
ax.bar(index + bar_width / 2, male_values, width=bar_width, color='green', alpha=0.3, label='Male')

# 绘制平滑曲线
x_f_smooth, y_f_smooth = smooth_curve(x, female_values)
x_m_smooth, y_m_smooth = smooth_curve(x, male_values)
ax.plot(x_f_smooth, y_f_smooth, 'r-', linewidth=3)
ax.plot(x_m_smooth, y_m_smooth, 'g-', linewidth=3)

# 添加数值标签
for i, val in enumerate(female_values):
    ax.text(i - bar_width / 2, val + 2, str(val), ha='center', va='bottom', fontsize=9, color='red')
for i, val in enumerate(male_values):
    ax.text(i + bar_width / 2, val + 2, str(val), ha='center', va='bottom', fontsize=9, color='green')

# 设置英文标题与标签
ax.set_title('Deaths by Age Group and Gender', fontsize=16)
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Number of Deaths', fontsize=12)
ax.set_xticks(index)
ax.set_xticklabels(categories)

# 图例
ax.legend()

# 保存图像
# plt.savefig('/mnt/data/06_death_distribution_gender_curve.png', transparent=True)
plt.tight_layout()
plt.show()


```








```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 读取数据
patients = pd.read_csv("/workspace/mimic-iv-data/3.1/hosp/patients.csv")
patients = patients.dropna(subset=['anchor_age'])
patients['gender'] = patients['gender'].map({'M': 'Male', 'F': 'Female'})

# 年龄分组
age_bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
age_labels = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49', '50~59',
              '60~69', '70~79', '80~89', '>=90']
patients['age_group'] = pd.cut(patients['anchor_age'], bins=age_bins, labels=age_labels, right=False)

# 标记是否死亡
patients['is_dead'] = patients['dod'].notnull().astype(int)

# 构建总人数交叉表和死亡人数交叉表
total_counts_gender = pd.crosstab(patients['age_group'], patients['gender'])
dead_counts_gender = pd.crosstab(patients[patients['is_dead'] == 1]['age_group'], 
                                 patients[patients['is_dead'] == 1]['gender'])

# 构建绘图数据并过滤死亡率为 NaN 的年龄段
categories = []
male_rates = []
female_rates = []

for age_group in age_labels:
    total_m = total_counts_gender.loc[age_group, 'Male'] if age_group in total_counts_gender.index and 'Male' in total_counts_gender.columns else 0
    total_f = total_counts_gender.loc[age_group, 'Female'] if age_group in total_counts_gender.index and 'Female' in total_counts_gender.columns else 0
    dead_m = dead_counts_gender.loc[age_group, 'Male'] if age_group in dead_counts_gender.index and 'Male' in dead_counts_gender.columns else 0
    dead_f = dead_counts_gender.loc[age_group, 'Female'] if age_group in dead_counts_gender.index and 'Female' in dead_counts_gender.columns else 0
    if total_m + total_f > 0:
        categories.append(age_group)
        male_rates.append(dead_m / total_m * 100 if total_m > 0 else 0)
        female_rates.append(dead_f / total_f * 100 if total_f > 0 else 0)

# 准备数据
x = np.arange(len(categories))
male_values = np.array(male_rates)
female_values = np.array(female_rates)

# 平滑函数
def smooth_curve(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.grid(False)

# 柱状图参数
bar_width = 0.35
index = np.arange(len(categories))

# 绘制柱状图（红：女性，绿：男性）
ax.bar(index - bar_width / 2, female_values, width=bar_width, color='red', alpha=0.3, label='Female')
ax.bar(index + bar_width / 2, male_values, width=bar_width, color='green', alpha=0.3, label='Male')

# 绘制平滑曲线
x_f_smooth, y_f_smooth = smooth_curve(x, female_values)
x_m_smooth, y_m_smooth = smooth_curve(x, male_values)
ax.plot(x_f_smooth, y_f_smooth, 'r-', linewidth=3)
ax.plot(x_m_smooth, y_m_smooth, 'g-', linewidth=3)

# 添加数值标签
for i, val in enumerate(female_values):
    ax.text(i - bar_width / 2, val + 0.2, f"{val:.1f}%", ha='center', va='bottom', fontsize=9, color='red')
for i, val in enumerate(male_values):
    ax.text(i + bar_width / 2, val + 0.2, f"{val:.1f}%", ha='center', va='bottom', fontsize=9, color='green')

# 设置英文标题与标签
ax.set_title('Death Rate by Age Group and Gender', fontsize=16)
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Death Rate (%)', fontsize=12)
ax.set_xticks(index)
ax.set_xticklabels(categories)

# 图例
ax.legend()

# 保存图像
# plt.savefig('/mnt/data/07_death_rate_gender_curve.png', transparent=True)
plt.tight_layout()
plt.show()


```

