import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test


# Ljung-Box检验
def ljungbox(df, col1, col2, lags, alpha):
    _, pvalue1 = lb_test(df[col1], lags=lags)
    _, pvalue2 = lb_test(df[col2], lags=lags)

    pvalue1 = [round(x, 3) for x in pvalue1]
    index1 = [i for i in range(len(pvalue1)) if pvalue1[i] < alpha][0] + 1
    pvalue2 = [round(x, 3) for x in pvalue2]
    index2 = [i for i in range(len(pvalue2)) if pvalue2[i] < alpha][0] + 1

    print(col1 + ' p-value < alpha index: ' + str(index1))
    print('p-value: ', pvalue1)
    print(col2 + ' p-value < alpha index: ' + str(index2))
    print('p-value: ', pvalue2)

    indexs = [i + 1 for i in range(10)]
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)

    ax1 = plt.subplot(211)
    ax1.set_title(col1)
    plt.plot(indexs, pvalue1)

    ax2 = plt.subplot(212)
    ax2.set_title(col2)
    plt.plot(indexs, pvalue2)

    plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

ljungbox(df, 'hs300_closing_price', 'hs300_yield_rate', 10, 0.05)