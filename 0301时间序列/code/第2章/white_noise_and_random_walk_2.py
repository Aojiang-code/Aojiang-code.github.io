import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 随机产生一个高斯白噪声序列
gauss_white_noise = np.random.normal(0, 1, 1000)
# 产生一个随机游走序列
random_walk = [sum(gauss_white_noise[0:i]) for i in range(len(gauss_white_noise))]

fig = plt.figure(figsize=(9, 6))
fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)
plt.subplot(221)
plt.title('hs300_closing_price')
plt.plot(df['hs300_closing_price'])
ax1 = fig.add_subplot(222)
sm.graphics.tsa.plot_acf(df['hs300_closing_price'], lags=50, ax=ax1, title='hs300_closing_price')

plt.subplot(223)
plt.title('random_walk')
plt.plot(random_walk)
ax2 = fig.add_subplot(224)
sm.graphics.tsa.plot_acf(random_walk, lags=50, ax=ax2, title='random_walk')
plt.show()