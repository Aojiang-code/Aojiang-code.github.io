import warnings
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# 根据AIC、BIC准则计算p、q
def select_p_q(df, col):
    ic_val = sm.tsa.arma_order_select_ic(df[col], ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)

    print(col + ' AIC', ic_val.aic_min_order)
    print(col + ' BIC', ic_val.bic_min_order)


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 对沪深300指数做一阶差分, 并对起始位置补0
df['hs300_closing_price_diff1'] = df['hs300_closing_price'].diff(1)
df['hs300_closing_price_diff1'].fillna(0, inplace=True)

select_p_q(df, 'hs300_yield_rate')
select_p_q(df, 'hs300_closing_price_diff1')