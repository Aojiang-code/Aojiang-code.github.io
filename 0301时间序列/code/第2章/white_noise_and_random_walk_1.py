import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 定义日期解析函数，将字符串转换为日期对象
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')

# 读取CSV文件，将'date'列解析为日期，并将其设为索引
df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 随机产生一个高斯白噪声序列，均值为0，标准差为1，长度为1000
gauss_white_noise = np.random.normal(0, 1, 1000)

# 创建一个图形对象，大小为9x6
fig = plt.figure(figsize=(9, 6))
# 调整子图之间的间距
fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)

# 在图形中添加第一个子图，用于绘制沪深300收益率
plt.subplot(221)
plt.title('hs300_yield_rate')
plt.plot(df['hs300_yield_rate'])

# 在图形中添加第二个子图，用于绘制沪深300收益率的自相关图
ax1 = fig.add_subplot(222)
sm.graphics.tsa.plot_acf(df['hs300_yield_rate'], lags=50, ax=ax1, title='hs300_yield_rate')

# 在图形中添加第三个子图，用于绘制高斯白噪声序列
plt.subplot(223)
plt.title('gauss_white_noise')
plt.plot(gauss_white_noise)

# 在图形中添加第四个子图，用于绘制高斯白噪声序列的自相关图
ax2 = fig.add_subplot(224)
sm.graphics.tsa.plot_acf(gauss_white_noise, lags=50, ax=ax2, title='gauss_white_noise')

# 显示图形
plt.show()