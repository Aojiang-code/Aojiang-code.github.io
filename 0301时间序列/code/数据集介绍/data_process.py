import pandas as pd
from functools import reduce


# 将月份向后递推一个月
def month_process(date):
    year = int(date[0:4])
    month = int(date[5:7]) + 1

    if month == 13:
        year += 1
        month = 1

    year = str(year)
    month = str(month) if month > 9 else '0' + str(month)

    return year + '-' + month


# 加载数据, 将5张数据表中的数据按照时间维度合并
def load_data(hs300_file, econ_file, credit_file, us_index_file, shibor_file):
    hs300 = pd.read_csv(hs300_file)
    econ = pd.read_csv(econ_file)
    credit = pd.read_csv(credit_file)
    us_index = pd.read_csv(us_index_file)
    shibor = pd.read_csv(shibor_file)

    dfs = [hs300, us_index, shibor]
    df = reduce(lambda left, right: pd.merge(left, right, on='date', how='left'), dfs)

    # 美元指数、10年期美债收益率数据交易时间不完全与A股同步（节假日等因素）
    # 因此需要补齐缺失值, 这里补值的方法是采用前后一天的均值补充
    us_cols = ['ust_closing_price', 'ust_extent', 'usdx_closing_price', 'usdx_extent']

    us_null_indexs = df[df.isnull().T.any()].index.tolist()
    for us_null_index in us_null_indexs:
        mean_value = df.loc[[us_null_index - 1, us_null_index + 1]][us_cols].mean().round(4)
        df.loc[us_null_index, us_cols] = mean_value

    # econ的指标是按月统计的, 下个月公布上个月的统计结果
    # 比如在预测沪深300 2021-07-20的指标时, 只能够使用CPI、PPI 2021-06的统计结果
    # 因此在做数据对齐时, 需要往后推一个月
    econ['month'] = econ.apply(lambda row: month_process(row['date']), axis=1)
    del econ['date']

    # 与econ相同, credit需要往后推一个月
    credit['month'] = credit.apply(lambda row: month_process(row['date']), axis=1)
    del credit['date']

    df['month'] = df['date'].str[0:4] + '-' + df['date'].str[5:7]

    df = pd.merge(df, econ, on='month', how='left')
    df = pd.merge(df, credit, on='month', how='left')

    df = df.round(4)
    df.index = pd.to_datetime(df['date'])

    del df['month']

    df.sort_index(inplace=True)

    return df


hs300_file = '../data/hs300.csv'
econ_file = '../data/econ.csv'
credit_file = '../data/credit.csv'
us_index_file = '../data/us_index.csv'
shibor_file = '../data/shibor.csv'

df = load_data(hs300_file, econ_file, credit_file, us_index_file, shibor_file)
df.to_csv('../data/informations.csv', index=False)