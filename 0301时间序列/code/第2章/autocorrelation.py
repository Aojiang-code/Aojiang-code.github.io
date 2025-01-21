from datetime import datetime
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 定义自相关函数
def autocorrelation(df, col1, col2, lags):
    # 创建一个图形对象，大小为9x6
    fig = plt.figure(figsize=(9, 6))
    # 调整子图之间的间距
    fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.92, wspace=None, hspace=0.3)

    # 在图形中添加第一个子图
    ax1 = fig.add_subplot(211)
    # 绘制第一个列的自相关图
    sm.graphics.tsa.plot_acf(df[col1], lags=lags, ax=ax1, title=col1)
    
    # 在图形中添加第二个子图
    ax2 = fig.add_subplot(212)
    # 绘制第二个列的自相关图
    sm.graphics.tsa.plot_acf(df[col2], lags=lags, ax=ax2, title=col2)
    
    # 显示图形
    plt.show()


# 读取CSV文件，将'date'列解析为日期，并将其设为索引
df = pd.read_csv('../../data/informations.csv', parse_dates=['date'], index_col='date')

# 调用自相关函数，分析两个列的自相关性
autocorrelation(df, 'hs300_closing_price', 'hs300_yield_rate', 20)