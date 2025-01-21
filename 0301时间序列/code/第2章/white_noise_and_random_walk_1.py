import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 随机产生一个高斯白噪声序列
gauss_white_noise = np.random.normal(0, 1, 1000)

fig = plt.figure(figsize=(9, 6))
fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)
plt.subplot(221)
plt.title('hs300_yield_rate')
plt.plot(df['hs300_yield_rate'])
ax1 = fig.add_subplot(222)
sm.graphics.tsa.plot_acf(df['hs300_yield_rate'], lags=50, ax=ax1, title='hs300_yield_rate')

plt.subplot(223)
plt.title('gauss_white_noise')
plt.plot(gauss_white_noise)
ax2 = fig.add_subplot(224)
sm.graphics.tsa.plot_acf(gauss_white_noise, lags=50, ax=ax2, title='gauss_white_noise')
plt.show()