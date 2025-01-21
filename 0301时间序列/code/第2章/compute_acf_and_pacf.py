import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 定义函数以计算自相关图（ACF）和偏自相关图（PACF）
def autocorrelation(df, col1, col2, lags):
    # 创建一个图形对象，大小为9x6
    fig = plt.figure(figsize=(9, 6))
    # 调整子图之间的间距
    fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)
    
    # 在图形中添加第一个子图，用于绘制第一个列的自相关图
    ax1 = fig.add_subplot(221)
    sm.graphics.tsa.plot_acf(df[col1], lags=lags, ax=ax1, title=col1 + ' ACF')
    
    # 在图形中添加第二个子图，用于绘制第一个列的偏自相关图
    ax2 = fig.add_subplot(223)
    sm.graphics.tsa.plot_pacf(df[col1], lags=lags, ax=ax2, title=col1 + ' PACF')
    
    # 在图形中添加第三个子图，用于绘制第二个列的自相关图
    ax1 = fig.add_subplot(222)
    sm.graphics.tsa.plot_acf(df[col2], lags=lags, ax=ax1, title=col2 + ' ACF')
    
    # 在图形中添加第四个子图，用于绘制第二个列的偏自相关图
    ax2 = fig.add_subplot(224)
    sm.graphics.tsa.plot_pacf(df[col2], lags=lags, ax=ax2, title=col2 + ' PACF')
    
    # 显示图形
    plt.show()


# 定义日期解析函数，将字符串转换为日期对象
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')

# 读取CSV文件，将'date'列解析为日期，并将其设为索引
df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 对沪深300指数做一阶差分, 并对起始位置补0
df['hs300_closing_price_diff1'] = df['hs300_closing_price'].diff(1)
df['hs300_closing_price_diff1'].fillna(0, inplace=True)

# 调用自相关函数，分析两个列的自相关性和偏自相关性
autocorrelation(df, 'hs300_closing_price_diff1', 'hs300_yield_rate', 20)