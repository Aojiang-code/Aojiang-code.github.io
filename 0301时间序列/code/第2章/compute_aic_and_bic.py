import warnings
import pandas as pd
import statsmodels.api as sm

# 忽略警告信息
warnings.filterwarnings("ignore")

# 根据AIC、BIC准则计算p、q
def select_p_q(df, col):
    # 使用statsmodels的arma_order_select_ic函数选择最佳的ARMA模型参数(p, q)
    ic_val = sm.tsa.arma_order_select_ic(df[col], ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
    
    # 打印AIC准则下的最佳(p, q)组合
    print(col + ' AIC', ic_val.aic_min_order)
    # 打印BIC准则下的最佳(p, q)组合
    print(col + ' BIC', ic_val.bic_min_order)

# 定义日期解析函数，将字符串转换为日期对象
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')

# 读取CSV文件，将'date'列解析为日期，并将其设为索引
df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 对沪深300指数做一阶差分，并对起始位置补0
df['hs300_closing_price_diff1'] = df['hs300_closing_price'].diff(1)
df['hs300_closing_price_diff1'].fillna(0, inplace=True)

# 计算并打印沪深300收益率的最佳ARMA模型参数
select_p_q(df, 'hs300_yield_rate')
# 计算并打印沪深300收盘价一阶差分的最佳ARMA模型参数
select_p_q(df, 'hs300_closing_price_diff1')