import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 相关系数
def autocorrelation(df, col1, col2, lags):
    fig = plt.figure(figsize=(9, 6))
    fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.92, wspace=None, hspace=0.3)

    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(df[col1], lags=lags, ax=ax1, title=col1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_acf(df[col2], lags=lags, ax=ax2, title=col2)
    plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

autocorrelation(df, 'hs300_closing_price', 'hs300_yield_rate', 20)