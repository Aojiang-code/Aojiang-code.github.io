import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 计算自相关图、偏自相关图
def autocorrelation(df, col1, col2, lags):
    fig = plt.figure(figsize=(9, 6))
    fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)
    ax1 = fig.add_subplot(221)
    sm.graphics.tsa.plot_acf(df[col1], lags=lags, ax=ax1, title=col1 + ' ACF')
    ax2 = fig.add_subplot(223)
    sm.graphics.tsa.plot_pacf(df[col1], lags=lags, ax=ax2, title=col1 + ' PACF')
    ax1 = fig.add_subplot(222)
    sm.graphics.tsa.plot_acf(df[col2], lags=lags, ax=ax1, title=col2 + ' ACF')
    ax2 = fig.add_subplot(224)
    sm.graphics.tsa.plot_pacf(df[col2], lags=lags, ax=ax2, title=col2 + ' PACF')
    plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 对沪深300指数做一阶差分, 并对起始位置补0
df['hs300_closing_price_diff1'] = df['hs300_closing_price'].diff(1)
df['hs300_closing_price_diff1'].fillna(0, inplace=True)

autocorrelation(df, 'hs300_closing_price_diff1', 'hs300_yield_rate', 20)