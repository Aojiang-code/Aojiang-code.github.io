# 第一部分代码

# 导入所需的库
import pandas as pd
import numpy as np
from datetime import datetime
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# 设置随机种子
np.random.seed(289)

# 定义输入参数
input_params = {
    'egfryears': [2, 4],
    'minegfrobs': 3,
    'minbpobs': 0,
    'minhrobs': 0,
    'train_pc': 80
}

# 预测模式设置为 'esrd' 或 'death'
predict_mode = 'esrd'

# 定义预测结果
outcomes = {
    'esrd': ['Non-ES', 'ESRD'],
    'death': ['Survival', 'Death']
}

# 目标设置为二元逻辑回归
objective = 'binary:logistic'

# 评估指标设置为 MCC（Matthews 相关系数）
eval_metric = 'mcc'

# SMOTE 标志设置为 False
smote_flag = False

# SMOTE 过采样和欠采样设置
smote_over = 2000
smote_under = 200

# 自动 XGBoost 标志设置为 False
autoxgb_flag = False

# 临床测试标志设置为 True
clintest_flag = True

# 重新抽样训练/测试集标志设置为 True
resample_flag = True

# 快速模式标志设置为 True
quick_flag = True

# 时间序列数量设置
num_ts = 2

# 特征列表
features = [
    'egfr', 'creatinine', 'glucose', 'height', 'bmi', 'weight',
    'sit_dia', 'sit_sys', 'sta_dia', 'sta_sys', 'sit_hr', 'sta_hr',
    'hgba1c', 'protcreatinine', 'proteinuria', 'diff_dia', 'diff_sys',
    'diff_hr', 'sit_pp', 'sta_pp', 'diff_pp'
]

# 根据 num_ts 的值定义额外特征
vars_extra = []
if num_ts == 19:
    vars_extra = features[:-2]  # 排除肌酐和身高
elif num_ts == 13:
    vars_extra = ['egfr', 'glucose', 'bmi', 'sta_sys', 'sta_dia', 'sta_hr', 'sit_sys', 'sit_dia', 'sit_hr', 'diff_hr', 'sit_pp', 'sta_pp', 'diff_pp']
elif num_ts == 9:
    vars_extra = ['egfr', 'glucose', 'bmi', 'sta_sys', 'sta_dia', 'sta_hr', 'sit_sys', 'sit_dia', 'sit_hr']
elif num_ts == 6:
    vars_extra = ['egfr', 'glucose', 'bmi', 'sta_sys', 'sta_dia', 'sta_hr']
elif num_ts == 2:
    vars_extra = ['egfr', 'glucose']

# 交叉验证折数设置
cv_fold = 10

# 卡方阈值设置
thresh_chisq = 0.999999999

# 颜色特征设置
color_feature = 'age.init'

# SHAP 特征数量设置
shap_top = 20

# SHAP 列数设置
shap_ncol = 5

# SHAP 分组设置
shap_groups = 6

# SHAP X 轴边界设置
shap_xbound = None

# 图例列数设置
legend_ncol = 5

# 分页设置
paging = True

# 在显示器上的图高设置
plot_height = "1024px"

# 在显示器上的轴文本大小设置
axis_text_size = 6

# 提取图高的数字值
plot_height_num = int(''.join(filter(str.isdigit, plot_height)))

# 定义颜色向量
col_vector = [
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#e6194b', '#3cb44b',
    '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#e8e8e8', '#000000'
]

# 定义非包含操作符
def ni(x, y):
    return not x in y

# 定义打印函数
def printly(m):
    print(m)

# 重要：普查日期 = 2016年11月15日 + 2年（2016年11月16日至2018年11月12日用于临床论文测试）
clin_init = pd.to_datetime('2016-11-15')
clin_last = clin_init + pd.DateOffset(years=2)

# 定义日期偏移函数
def offset_date(x):
    return x.mean() - (clin_last.year + (clin_last.month - 1) / 12)

# 阈值设置
thresh_p = 0.5
thresh_2y = 2.14
thresh_10y = 11.14
dig_suffix = ''

# 设置数据文件的本地目录
dir_path = '/mnt/appTrendal/data'

# 更改工作目录
import os
os.chdir(dir_path)

# 设置警告选项和表达式限制
pd.options.mode.chained_assignment = None  # default='warn'


# 第二部分代码
# 读取临床数据

# 从 'Death.csv' 文件读取数据，指定列名为 'id' 和 'date'，将 'NULL' 字符串视为缺失值
death = pd.read_csv('Death.csv', names=['id', 'date'], na_values='NULL')
# 将 'date' 列转换为 datetime 格式
death['date'] = pd.to_datetime(death['date'])
# 设置数据表的键为 'id' 和 'date'
death.set_index(['id', 'date'], inplace=True)

# 从 'Diabetes.csv' 文件读取数据，指定列名并将 'NULL' 视为缺失值
dt = pd.read_csv('Diabetes.csv', names=['id', 'date', 'glucose', 'glucose.str', 'hgba1c', 'hgba1c.str'], na_values='NULL')
# 将 'date' 列转换为 datetime 格式
dt['date'] = pd.to_datetime(dt['date'])
# 将数据重塑为长格式，只保留 'id'、'date'、'glucose' 和 'hgba1c' 列，去除缺失值
diabetes = dt.melt(id_vars=['id', 'date'], value_vars=['glucose', 'hgba1c'], var_name='variable', value_name='value').dropna()

# 从 'Exam.corrected.csv' 文件读取数据，指定列名并将 '\\N' 视为缺失值
dt = pd.read_csv('Exam.corrected.csv', names=['id', 'date', 'height', 'height_raw', 'weight', 'bmi_raw', 'bmi', 'sit_dia', 'sit_sys', 'sit_hr', 'sta_dia', 'sta_sys', 'sta_hr'], na_values='\\N')
# 将 'date' 列转换为 datetime 格式
dt['date'] = pd.to_datetime(dt['date'])
# 删除不需要的 'height_raw' 和 'bmi_raw' 列
dt.drop(columns=['height_raw', 'bmi_raw'], inplace=True)
# 计算坐位和站位的收缩压差
dt['diff_dia'] = dt['sit_dia'] - dt['sta_dia']
# 计算坐位和站位的舒张压差
dt['diff_sys'] = dt['sit_sys'] - dt['sta_sys']
# 计算坐位和站位的心率差
dt['diff_hr'] = dt['sit_hr'] - dt['sta_hr']
# 计算坐位的脉压
dt['sit_pp'] = dt['sit_sys'] - dt['sit_dia']
# 计算站位的脉压
dt['sta_pp'] = dt['sta_sys'] - dt['sta_dia']
# 计算坐位和站位的脉压差
dt['diff_pp'] = dt['sit_pp'] - dt['sta_pp']
# 将数据重塑为长格式，去除缺失值，抑制警告
exam = dt.melt(id_vars=['id', 'date'], value_vars=['height', 'weight', 'bmi', 'diff_dia', 'diff_sys', 'diff_hr', 'sit_pp', 'sta_pp'], var_name='variable', value_name='value').dropna()
exam['date'] = pd.to_datetime(exam['date'])

# 从 'KidneyFunction.csv' 文件读取数据，指定列名并将 'NULL' 视为缺失值
dt = pd.read_csv('KidneyFunction.csv', names=['id', 'date', 'creatinine', 'egfr'], na_values='NULL')
# 将 'date' 列转换为 datetime 格式
dt['date'] = pd.to_datetime(dt['date'])
# 将数据重塑为长格式，去除缺失值
kidney = dt.melt(id_vars=['id', 'date'], value_vars=['creatinine', 'egfr'], var_name='variable', value_name='value').dropna()

# 从 'UrineProteinCreatinineRatios.csv' 文件读取数据，指定列名并将 'NULL' 视为缺失值
dt = pd.read_csv('UrineProteinCreatinineRatios.csv', names=['id', 'date', 'test', 'other.test', 'result', 'other.result', 'protcreatinine', 'result.str', 'norm.range', 'units', 'provider'], na_values='NULL')
# 将 'date' 列转换为 datetime 格式
dt['date'] = pd.to_datetime(dt['date'])
# 将数据重塑为长格式，只保留 'id'、'date' 和 'protcreatinine' 列，去除缺失值
protcreatinine = dt.melt(id_vars=['id', 'date'], value_vars=['protcreatinine'], var_name='variable', value_name='value').dropna()

# 从 'Urine24hrProtein.csv' 文件读取数据，指定列名并将 'NULL' 视为缺失值
dt = pd.read_csv('Urine24hrProtein.csv', names=['id', 'date', 'test', 'other.test', 'result', 'other.result', 'proteinuria', 'result.str', 'norm.range', 'units', 'provider'], na_values='NULL')
# 将 'date' 列转换为 datetime 格式
dt['date'] = pd.to_datetime(dt['date'])
# 将数据重塑为长格式，只保留 'id'、'date' 和 'proteinuria' 列，去除缺失值
proteinuria = dt.melt(id_vars=['id', 'date'], value_vars=['proteinuria'], var_name='variable', value_name='value').dropna()

# 从 'PtID.csv' 文件读取数据，指定列名并将 'NULL' 视为缺失值
patient = pd.read_csv('PtID.csv', names=['id', 'birth.year', 'gender'], na_values='NULL')
# 设置数据表的键为 'id'
patient.set_index('id', inplace=True)

# 从 'Modality.csv' 文件读取数据，指定列名并将 'NULL' 视为缺失值
dt = pd.read_csv('Modality.csv', names=['id', 'date', 'end.date', 'modality', 'setting'], na_values='NULL')
# 将 'date' 列转换为 datetime 格式
dt['date'] = pd.to_datetime(dt['date'])
# 将 'end.date' 列转换为 datetime 格式
dt['end.date'] = pd.to_datetime(dt['end.date'])
# 设置数据表的键为 'id'
dt.set_index('id', inplace=True)
# 按照 'id' 和 'date' 对数据进行排序
dt.sort_values(by=['id', 'date'], inplace=True)
modality = dt.copy()  # 复制数据表以备后用

# 提取移植、供体和非 RRT 的记录，按 'id' 分组，获取最小日期和模态
transplant = modality[modality['modality'].isin(['Transplant', 'Donor', 'Non RRT'])].groupby('id').agg({'date': 'min', 'modality': 'first'}).reset_index()
# 提取透析记录，按 'id' 分组，获取最小日期和模态
dialysis = modality[modality['modality'].isin(['HD', 'PD', 'Nocturnal HD', 'Haemodiafiltration', 'Haemofiltration'])].groupby('id').agg({'date': 'min', 'modality': 'first'}).reset_index()


# 第三部分代码
# 处理数据

# 合并多个数据表（肾脏、糖尿病、检查、尿液蛋白和24小时尿液蛋白），并去除重复行
dt = pd.concat([kidney, diabetes, exam, protcreatinine, proteinuria]).drop_duplicates()
# 按照 'variable'、'id' 和 'date' 对数据进行排序
dt.sort_values(by=['variable', 'id', 'date'], inplace=True)

# 将数据重塑为长格式，并计算每个变量的异常值
melted = dt.groupby('variable').apply(lambda x: x.assign(outlier=scores(x['value'], type='chisq', prob=thresh_chisq))).reset_index(drop=True)

# 提取 'egfr' 变量的数据
m = melted[melted['variable'] == 'egfr']

# 获取每个 'id' 的最小日期对应的 'egfr' 值
egfr_init = m.loc[m.groupby('id')['date'].idxmin()]
egfr_init['variable'] = 'egfr.init'  # 将变量名设置为 'egfr.init'

# 获取每个 'id' 的最大日期对应的 'egfr' 值
egfr_last = m.loc[m.groupby('id')['date'].idxmax()]
egfr_last['variable'] = 'egfr.last'  # 将变量名设置为 'egfr.last'

# 获取第一个 'egfr' 值小于 10 的记录
esrd_hit = m[m['value'] < 10].iloc[0]
esrd_hit['id'] = egfr_init['id']
esrd_hit['variable'] = 'esrd.hit'  # 将变量名设置为 'esrd.hit'
esrd_hit['outlier'] = False  # 将异常值标记设置为 False

# 获取第一个 'egfr' 值小于 15 的记录
esrd_close = m[m['value'] < 15].iloc[0]
esrd_close['id'] = egfr_init['id']
esrd_close['variable'] = 'esrd.close'  # 将变量名设置为 'esrd.close'
esrd_close['outlier'] = False  # 将异常值标记设置为 False

# 合并所有 'egfr' 相关数据，并去除缺失值
egfr_melt = pd.concat([egfr_init, egfr_last, esrd_hit, esrd_close]).dropna()

# 将数据重塑为宽格式，按 'id' 分组，使用 'variable' 作为列名
dt_wide = egfr_melt.pivot(index='id', columns='variable', values=['date', 'value'])

# 计算 'egfr' 的年数差
dt_wide['egfr.y'] = (dt_wide['date.egfr.last'] - dt_wide['date.egfr.init']).dt.days / 365.25
dt_wide['egfr.y'].attrs['units'] = 'years'  # 设置 'egfr.y' 的单位为年

# 将 'egfr.y' 赋值给 'preesrd.y'
dt_wide['preesrd.y'] = dt_wide['egfr.y']
# 如果存在 'esrd.hit' 日期，则计算 'preesrd.y'
dt_wide.loc[dt_wide['date.esrd.hit'].notna(), 'preesrd.y'] = (dt_wide['date.esrd.hit'] - dt_wide['date.egfr.init']).dt.days / 365.25

# 将 'egfr.y' 赋值给 'preclose.y'
dt_wide['preclose.y'] = dt_wide['egfr.y']
# 如果存在 'esrd.close' 日期，则计算 'preclose.y'
dt_wide.loc[dt_wide['date.esrd.close'].notna(), 'preclose.y'] = (dt_wide['date.esrd.close'] - dt_wide['date.egfr.init']).dt.days / 365.25

# 设置数据表的键为 'id'
dt_wide.set_index('id', inplace=True)

# 计算相对年份
melted['relyear'] = (melted['date'] - dt_wide.loc[melted['id'], 'date.egfr.init']).dt.days / 365.25
melted['relyear'].attrs['units'] = 'years'  # 设置 'relyear' 的单位为年

# 处理特征
for v in features:
    m = melted[melted['variable'] == v]  # 提取当前特征的数据
    obs = m.groupby('id').size().reset_index(name='N')  # 计算每个 'id' 的观察数
    obs.rename(columns={'N': v}, inplace=True)  # 将观察数列重命名为当前特征名
    obs.set_index('id', inplace=True)  # 设置键为 'id'
    dt_wide = dt_wide.join(obs, how='left')  # 将观察数合并到主数据表中

# 将死亡数据合并到主数据表中
dt_wide.rename(columns={'date': 'date.death'}, inplace=True)  # 将 'date' 列重命名为 'date.death'
dt_wide = patient.join(dt_wide, how='left')  # 将患者数据合并到主数据表中

# 定义分类序列
cats = np.arange(1, 12.5, 0.5)  # 定义分类序列
cats_col = ['cat' + str(int(cat)) for cat in cats]  # 生成分类列名

# 根据条件设置分类
dt_wide['category'] = np.where(dt_wide['date.esrd.hit'].notna() & (dt_wide['preesrd.y'] <= input_params['egfryears'][1]),
                                outcomes[predict_mode][1], outcomes[predict_mode][0])

# 为每个分类列设置值
for i in range(len(cats)):
    dt_wide[cats_col[i]] = np.where(dt_wide['date.esrd.hit'].notna() & (dt_wide['preesrd.y'] <= cats[i]),
                                     outcomes[predict_mode][1], outcomes[predict_mode][0])

# 计算初始年龄
dt_wide['age.init'] = dt_wide['date.egfr.init'].dt.year - dt_wide['birth.year']

# 计算死亡年数
dt_wide['death.y'] = (dt_wide['date.death'] - dt_wide['date.egfr.init']).dt.days / 365.25
dt_wide['death.y'].attrs['units'] = 'years'  # 设置 'death.y' 的单位为年

# 设置数据表的列顺序
col_order = ["id", "preesrd.y", "preclose.y", "egfr.y", "death.y"] + features + ["gender", "age.init", "value.egfr.init", "value.esrd.close", "value.esrd.hit", "value.egfr.last", "birth.year", "date.egfr.init", "date.esrd.close", "date.esrd.hit", "date.egfr.last", "date.death"]
dt_wide = dt_wide[col_order]

# 复制数据表以备后用
meta = dt_wide.copy()

# 生成样本运行名称
sample_run = f"samples-{input_params['egfryears'][0]}-{input_params['egfryears'][1]}-{input_params['minegfrobs']}-{input_params['minbpobs']}-{input_params['minhrobs']}"

# 将元数据保存为 CSV 文件
meta.to_csv(f"{sample_run}.meta.csv", index=False)

# 生成样本文件名
sample_fn = f"{sample_run}.csv"

# 如果样本文件存在，则读取数据
if os.path.exists(sample_fn):
    dt_wide = pd.read_csv(sample_fn)  # 读取数据
    if resample_flag:
        training = np.random.choice(dt_wide['id'], size=int(dt_wide.shape[0] * input_params['train_pc'] / 100), replace=False)  # 随机选择训练样本
        dt_wide['train'] = dt_wide['id'].isin(training)  # 标记训练样本
else:
    # 如果样本文件不存在
    if clintest_flag:
        dt_wide = meta[(dt_wide['preclose.y'] >= input_params['egfryears'][0]) &
                       ((dt_wide['egfr.y'] >= input_params['egfryears'][1]) | dt_wide['date.esrd.hit'].notna()) &
                       ((input_params['minegfrobs'] == 0) | (dt_wide['egfr'] >= input_params['minegfrobs'])) &
                       ((input_params['minbpobs'] == 0) | (dt_wide['sta_dia'] >= input_params['minbpobs'])) &
                       ((input_params['minhrobs'] == 0) | (dt_wide['sta_hr'] >= input_params['minhrobs']))]  # 选择临床测试样本
        training = np.random.choice(np.setdiff1d(dt_wide['id'], clinician_test['id']), size=int(dt_wide.shape[0] * input_params['train_pc'] / 100), replace=False)  # 随机选择训练样本
    else:
        dt_wide = meta[(dt_wide['preesrd.y'] >= input_params['egfryears'][0]) &
                       ((input_params['minegfrobs'] == 0) | (dt_wide['egfr'] >= input_params['minegfrobs'])) &
                       ((input_params['minbpobs'] == 0) | (dt_wide['sta_dia'] >= input_params['minbpobs'])) &
                       ((input_params['minhrobs'] == 0) | (dt_wide['sta_hr'] >= input_params['minhrobs']))]  # 选择样本
        training = np.random.choice(dt_wide['id'], size=int(dt_wide.shape[0] * input_params['train_pc'] / 100), replace=False)  # 随机选择训练样本

    dt_wide['train'] = dt_wide['id'].isin(training)  # 标记训练样本
    dt_wide.to_csv(sample_fn, index=False)  # 将数据保存为 CSV 文件

# 复制数据表
filtered = dt_wide.copy()

# 获取训练集
train_set = filtered[filtered['train'] == True]

# 获取测试集
test_set = filtered[filtered['train'] == False]

# 合并多个数据表并去除重复行
dt = pd.concat([kidney, diabetes, exam, protcreatinine, proteinuria]).drop_duplicates()
# 按照 'variable'、'id' 和 'date' 对数据进行排序
dt.sort_values(by=['variable', 'id', 'date'], inplace=True)

# 将数据重塑为长格式，并计算每个变量的异常值
melted = dt.groupby('variable').apply(lambda x: x.assign(outlier=scores(x['value'], type='chisq', prob=thresh_chisq))).reset_index(drop=True)

# 定义异常值标记函数
def flag_outliers(vari):
    mv = melted[melted['variable'] == vari]  # 提取当前变量的数据
    print(f"{vari}: ")  # 打印变量名
    vals = mv['value']  # 获取值
    summ = pd.DataFrame([{'variable': vari, 'points': len(vals)}] + [vals.describe().to_dict()])  # 计算统计信息
    global stats
    stats = summ if 'stats' not in globals() else pd.concat([stats, summ], ignore_index=True)  # 更新统计信息
    if vari == 'weight':
        outidx = mv[mv['value'] > 500].index  # 处理体重异常值
    elif vari == 'bmi':
        outidx = mv[mv['value'] > 200].index  # 处理 BMI 异常值
    elif vari == 'height':
        outidx = mv[mv['value'] > 250].index  # 处理身高异常值
    else:
        outs = scores(vals, type='chisq', prob=thresh_chisq)  # 计算异常值
        outvals = vals[outs].unique()  # 获取异常值
        outidx = mv[mv['value'].isin(outvals)].index  # 获取异常值索引
    print(melted.loc[outidx, 'value'])  # 打印异常值
    return outidx  # 返回异常值索引

# 检查特征，排除特定变量
check = [feature for feature in features if feature not in ['creatinine', 'glucose', 'hgba1c', 'proteinuria', 'protcreatinine']]
stats = None  # 初始化统计信息
outliers = np.concatenate([flag_outliers(var) for var in check])  # 获取异常值

# 保存统计信息到 TSV 文件
pd.DataFrame(stats).to_csv('stats.tsv', sep='\t', index=False)

# 保存异常值到 TSV 文件
melted.loc[outliers].to_csv('outliers.tsv', sep='\t', index=False)

# 去除异常值
remains = melted.drop(outliers)
remains.set_index(['id', 'date'], inplace=True)  # 设置键为 'id' 和 'date'

# 获取每个 'id' 的最小日期对应的 'egfr'
egfr_init = remains[remains['variable'] == 'egfr'].groupby('id')['date'].min().reset_index()
egfr_init.rename(columns={'date': 'egfr.init'}, inplace=True)  # 重命名列

# 计算相对年份
outed = remains[remains['variable'] == 'egfr'].copy()
outed['rel.year'] = (outed['date'] - egfr_init['egfr.init']).dt.days / 365.25  # 计算相对年份
outed['rel.year'].attrs['units'] = 'years'  # 设置单位为年

# 删除不需要的列
outed.drop(columns=['egfr.init', 'outlier'], inplace=True)

# 设置键为 'id'
outed.set_index('id', inplace=True)

# 提取在过滤后的数据中存在的 'id'
j = outed[outed.index.isin(filtered['id'])].copy()

# 复制数据，筛选出相对年份小于等于输入的 egfryears
joined = j[j['rel.year'] <= input_params['egfryears'][0]].copy()

# 处理特征，排除肌酐
vars = [feature for feature in features if feature != 'creatinine']
combi = [vars, vars_extra, j]  # 组合特征

# 计算摘要
import hashlib
dig = hashlib.md5(str(combi).encode()).hexdigest()  # 计算摘要
print(dig)  # 打印摘要

# 生成摘要文件名
dig_file = f"{dig}.csv"

# 定义输入参数
inputs = ['minegfrobs', 'minbpobs', 'minhrobs', 'egfryears', 'train_pc']
input_filt = input_params.copy()  # 复制输入参数
input_filt['egfryears'] = f"{input_filt['egfryears'][0]}-{input_filt['egfryears'][1]}"  # 格式化 egfryears
input_filt['train_pc'] = input_filt['train_pc']  # 复制训练比例
input_filt['num_ts'] = num_ts  # 复制时间序列数
input_filt['filtered'] = filtered.shape[0]  # 计算过滤后的样本数
input_filt['train'] = train_set.shape[0]  # 计算训练集样本数
input_filt['test'] = test_set.shape[0]  # 计算测试集样本数

# 转换为数据表
filt = pd.DataFrame(input_filt)
print(filt)  # 打印过滤后的信息

# 保存过滤后的信息到 CSV 文件
filt.to_csv(f"{dig}.filt.csv", index=False)

# 保存过滤后的元数据到 CSV 文件
filtered.to_csv(f"{dig}.meta.csv", index=False)

# 如果摘要文件存在，则读取数据
if os.path.exists(dig_file):
    fresh = pd.read_csv(dig_file)  # 读取数据
else:
    fresh = pd.DataFrame()  # 初始化数据表
    keep = ['id', 'rel.year', 'value']  # 保留的列
    for var in vars:
        dt = joined[joined['variable'] == var]  # 提取特征数据
        dt_n = dt.shape[0]  # 计算观察数
        if dt_n == 0:
            continue  # 如果没有数据则跳过
        dt = dt[keep].dropna()  # 去除 NA
        dt.rename(columns={'value': var}, inplace=True)  # 重命名列
        print(f"{var}: {dt_n}")  # 打印特征名和观察数
        if var in vars_extra:
            f = extract_features(dt, column_id='id', column_sort='rel.year', disable_progressbar=True)  # 提取特征
        else:
            f = extract_features(dt, column_id='id', column_sort='rel.year', disable_progressbar=True, default_fc_parameters=tsfresh.feature_extraction.settings.MinimalFCParameters())  # 提取特征
        f['rn'] = f.index.astype(int)  # 将行名转换为数字
        fresh = f if var == vars[0] else fresh.merge(f, on='rn', how='outer')  # 合并数据

    fresh['rn'] = fresh.index  # 将行名转换为数字
    fresh.columns = [col.replace('__', ': ').replace('_', ' ') for col in fresh.columns]  # 替换列名
    for j in range(fresh.shape[1]):
        fresh.iloc[:, j] = fresh.iloc[:, j].replace([np.inf, -np.inf, np.nan], np.nan)  # 处理无穷大和 NaN
    fresh.to_csv(dig_file, index=False)  # 保存数据

# 设置键为 'id'
fresh.set_index('id', inplace=True)

# 确保列名唯一
fresh.columns = pd.io.parsers.ParserBase({'names': fresh.columns})._maybe_dedup_names(fresh.columns)

# 处理无穷大和 NaN
for j in range(fresh.shape[1]):
    fresh.iloc[:, j] = fresh.iloc[:, j].replace([np.inf, -np.inf, np.nan], np.nan)

# 删除不需要的列
fresh = fresh.loc[:, ~fresh.columns.str.contains("__sum_values")]
fresh = fresh.loc[:, ~fresh.columns.str.contains("__length")]

# 返回处理后的数据
fresh

# 第四部分代码
# 测试数据集

# 从 meta2 中提取 fresh2 对应的行，生成 test2 数据表
test2 = meta2[fresh2]

# 创建 test2.label，标记 pats2 中类别为预测模式的第二类的样本，1 表示属于该类别，0 表示不属于
test2_label = (pats2['category'] == outcomes[predict_mode][1]).astype(int)

# 提取 test2 中的 'id' 列
test2_id = test2['id']

# 从 test2 中删除 'id' 列
test2.drop(columns=['id'], inplace=True)

# 将 test2 转换为 XGBoost 的 DMatrix 格式，包含特征数据和标签
dtest2 = xgb.DMatrix(data=test2.values, label=test2_label)

# 进行比例检验，比较训练集和测试集的正类样本比例
pt2 = prop.test([sum(train_label), sum(test2_label)], [len(train_label), len(test2_label)])

# 输出比例检验的结果
print(pt2)

# 打印比例检验的结果
print(pt2)

# 使用训练好的 XGBoost 模型对 dtest2 进行预测，得到预测概率
pred2 = model.predict(dtest2)

# 创建数据表 pred2.dt，包含样本 id 和对应的预测概率
pred2_dt = pd.DataFrame({'id': test2_id, 'p': pred2})

# 设置 pred2.dt 的键为 'id'
pred2_dt.set_index('id', inplace=True)

# 将预测概率转换为标签，概率大于等于 0.5 的标记为 1，否则为 0
pred2_label = (pred2 >= 0.5).astype(int)

# 从 pats2 中提取 id 在 test2_id 中的样本，生成 test.pats2 数据框
test_pats2 = pats2[pats2['id'].isin(test2_id)].copy()

# 创建 endstage_hit 列，标记是否为预测模式的类别
test_pats2['endstage_hit'] = (test_pats2['category'] == outcomes[predict_mode][1]).astype(int)

# 创建 endstage_pred 列，存储预测标签
test_pats2['endstage_pred'] = pred2_label

# 计算预测错误，得到真实标签与预测标签的差值
error2 = test2_label - pred2_label

# 将错误值添加到 test.pats2 数据框中
test_pats2['error'] = error2

# 第五部分代码
# 第五部分代码

# 进行模型评估

# 计算混淆矩阵
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(test_pats2['endstage_hit'], test_pats2['endstage_pred'])

# 打印混淆矩阵
print("Confusion Matrix:")
print(conf_matrix)

# 计算准确率
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")

# 计算精确率
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
print(f"Precision: {precision:.2f}")

# 计算召回率
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
print(f"Recall: {recall:.2f}")

# 计算F1分数
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"F1 Score: {f1_score:.2f}")

# 生成 ROC 曲线
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(test_pats2['endstage_hit'], pred2)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# 第六部分代码
# 第六部分代码

# 生成最终报告

# 创建一个报告数据框，包含模型评估的结果
report = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [accuracy, precision, recall, f1_score, roc_auc]
})

# 保存报告到 CSV 文件
report.to_csv('model_evaluation_report.csv', index=False)

# 打印报告
print("Model Evaluation Report:")
print(report)

# 生成特征重要性图
import xgboost as xgb

# 获取特征重要性
importance = model.get_score(importance_type='weight')
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # 反转 y 轴
plt.show()

# 保存特征重要性到 CSV 文件
importance_df.to_csv('feature_importance.csv', index=False)

# 结束报告
print("Report generation completed.")

# 第七部分代码
# 第七部分代码

# 进行模型保存和加载

import joblib

# 保存训练好的模型
model_filename = 'xgboost_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# 加载模型
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# 使用加载的模型进行预测
loaded_pred = loaded_model.predict(dtest2)

# 创建数据框，包含样本 id 和对应的预测概率
loaded_pred_dt = pd.DataFrame({'id': test2_id, 'loaded_prediction': loaded_pred})

# 设置 loaded_pred_dt 的键为 'id'
loaded_pred_dt.set_index('id', inplace=True)

# 打印加载模型的预测结果
print("Loaded Model Predictions:")
print(loaded_pred_dt.head())

# 结束模型保存和加载过程
print("Model saving and loading process completed.")

# 第八部分代码

# 进行结果可视化

# 绘制预测结果与真实标签的对比图
plt.figure(figsize=(12, 6))
plt.scatter(test_pats2['id'], test_pats2['endstage_hit'], color='blue', label='True Labels', alpha=0.5)
plt.scatter(test_pats2['id'], test_pats2['endstage_pred'], color='red', label='Predicted Labels', alpha=0.5)
plt.xlabel('Sample ID')
plt.ylabel('Label')
plt.title('True vs Predicted Labels')
plt.legend()
plt.xticks(rotation=90)  # 旋转 x 轴标签
plt.tight_layout()
plt.show()

# 绘制预测概率的直方图
plt.figure(figsize=(10, 6))
plt.hist(pred2, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.show()

# 结束结果可视化过程
print("Result visualization completed.")


# 第九部分代码

# 进行模型的超参数调优

from sklearn.model_selection import GridSearchCV

# 定义 XGBoost 模型
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 定义超参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 使用 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)

# 拟合模型
grid_search.fit(train_set.drop(columns=['endstage_hit', 'endstage_pred', 'error']), train_set['endstage_hit'])

# 输出最佳参数
print("Best Parameters:")
print(grid_search.best_params_)

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_

# 在测试集上进行预测
best_pred = best_model.predict(dtest2)

# 创建数据框，包含样本 id 和对应的最佳预测标签
best_pred_dt = pd.DataFrame({'id': test2_id, 'best_prediction': best_pred})

# 设置 best_pred_dt 的键为 'id'
best_pred_dt.set_index('id', inplace=True)

# 打印最佳模型的预测结果
print("Best Model Predictions:")
print(best_pred_dt.head())

# 结束超参数调优过程
print("Hyperparameter tuning completed.")

# 第十部分代码

# 进行模型的最终评估和保存

# 计算最佳模型的混淆矩阵
best_conf_matrix = confusion_matrix(test_pats2['endstage_hit'], best_pred)

# 打印最佳模型的混淆矩阵
print("Best Model Confusion Matrix:")
print(best_conf_matrix)

# 计算最佳模型的准确率
best_accuracy = (best_conf_matrix[0, 0] + best_conf_matrix[1, 1]) / np.sum(best_conf_matrix)
print(f"Best Model Accuracy: {best_accuracy:.2f}")

# 计算最佳模型的精确率
best_precision = best_conf_matrix[1, 1] / (best_conf_matrix[1, 1] + best_conf_matrix[0, 1]) if (best_conf_matrix[1, 1] + best_conf_matrix[0, 1]) > 0 else 0
print(f"Best Model Precision: {best_precision:.2f}")

# 计算最佳模型的召回率
best_recall = best_conf_matrix[1, 1] / (best_conf_matrix[1, 1] + best_conf_matrix[1, 0]) if (best_conf_matrix[1, 1] + best_conf_matrix[1, 0]) > 0 else 0
print(f"Best Model Recall: {best_recall:.2f}")

# 计算最佳模型的 F1 分数
best_f1_score = 2 * (best_precision * best_recall) / (best_precision + best_recall) if (best_precision + best_recall) > 0 else 0
print(f"Best Model F1 Score: {best_f1_score:.2f}")

# 计算最佳模型的 ROC AUC
best_fpr, best_tpr, best_thresholds = roc_curve(test_pats2['endstage_hit'], best_pred)
best_roc_auc = auc(best_fpr, best_tpr)

# 打印最佳模型的 ROC AUC
print(f"Best Model ROC AUC: {best_roc_auc:.2f}")

# 保存最佳模型
best_model_filename = 'best_xgboost_model.joblib'
joblib.dump(best_model, best_model_filename)
print(f"Best model saved to {best_model_filename}")

# 结束最终评估和保存过程
print("Final evaluation and model saving completed.")

# 第十一部分代码

# 进行结果的总结和报告生成

# 创建一个最终报告数据框，包含最佳模型评估的结果
final_report = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [best_accuracy, best_precision, best_recall, best_f1_score, best_roc_auc]
})

# 保存最终报告到 CSV 文件
final_report.to_csv('final_model_evaluation_report.csv', index=False)

# 打印最终报告
print("Final Model Evaluation Report:")
print(final_report)

# 生成特征重要性图
importance_best = best_model.get_score(importance_type='weight')
importance_best_df = pd.DataFrame(importance_best.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

# 绘制最佳模型的特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(importance_best_df['Feature'], importance_best_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Best Model Feature Importance')
plt.gca().invert_yaxis()  # 反转 y 轴
plt.show()

# 保存最佳模型的特征重要性到 CSV 文件
importance_best_df.to_csv('best_model_feature_importance.csv', index=False)

# 结束结果总结和报告生成过程
print("Result summary and report generation completed.")


# 第十二部分代码

# 进行模型的部署准备

# 定义一个函数，用于预测新样本
def predict_new_samples(new_data):
    """
    使用训练好的模型对新样本进行预测。
    
    参数:
    new_data (DataFrame): 包含新样本特征的数据框
    
    返回:
    DataFrame: 包含样本 ID 和预测结果的数据框
    """
    # 确保新数据的特征与训练时一致
    new_data = new_data[best_model.get_booster().feature_names]  # 选择模型所需的特征
    predictions = best_model.predict(new_data)  # 进行预测
    return pd.DataFrame({'id': new_data.index, 'prediction': predictions})

# 示例：准备新样本数据
# new_samples = pd.DataFrame(...)  # 这里应填入新样本数据的准备过程

# 使用模型对新样本进行预测
# predicted_results = predict_new_samples(new_samples)

# 打印预测结果
# print(predicted_results)

# 保存预测结果到 CSV 文件
# predicted_results.to_csv('new_samples_predictions.csv', index=False)

# 结束模型部署准备过程
print("Model deployment preparation completed.")


# 第十三部分代码

# 进行模型的监控和维护

import time

# 定义一个函数，用于监控模型性能
def monitor_model_performance(new_data, true_labels):
    """
    监控模型性能，计算并输出最新的评估指标。
    
    参数:
    new_data (DataFrame): 包含新样本特征的数据框
    true_labels (Series): 新样本的真实标签
    """
    # 使用模型对新数据进行预测
    predictions = best_model.predict(new_data)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # 计算准确率
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
    
    # 计算精确率
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
    
    # 计算召回率
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
    
    # 计算 F1 分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 打印监控结果
    print("Model Performance Monitoring:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

# 示例：定期监控模型性能
# while True:
#     new_data = pd.DataFrame(...)  # 获取新样本数据
#     true_labels = pd.Series(...)  # 获取新样本的真实标签
#     monitor_model_performance(new_data, true_labels)
#     time.sleep(3600)  # 每小时监控一次

# 结束模型监控和维护过程
print("Model monitoring and maintenance preparation completed.")


# 第十四部分代码

# 进行模型的更新和再训练

# 定义一个函数，用于更新模型
def update_model(new_data, new_labels):
    """
    使用新数据更新模型。
    
    参数:
    new_data (DataFrame): 包含新样本特征的数据框
    new_labels (Series): 新样本的真实标签
    """
    # 将新数据与旧数据合并
    global train_set
    combined_data = pd.concat([train_set.drop(columns=['endstage_hit']), new_data], ignore_index=True)
    combined_labels = pd.concat([train_set['endstage_hit'], new_labels], ignore_index=True)

    # 重新训练模型
    updated_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    updated_model.fit(combined_data, combined_labels)

    # 保存更新后的模型
    updated_model_filename = 'updated_xgboost_model.joblib'
    joblib.dump(updated_model, updated_model_filename)
    print(f"Updated model saved to {updated_model_filename}")

# 示例：获取新样本数据和标签
# new_data = pd.DataFrame(...)  # 获取新样本数据
# new_labels = pd.Series(...)  # 获取新样本的真实标签

# 更新模型
# update_model(new_data, new_labels)

# 结束模型更新和再训练过程
print("Model update and retraining completed.")