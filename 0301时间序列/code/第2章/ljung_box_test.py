import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test


# 定义Ljung-Box检验函数
def ljungbox(df, col1, col2, lags, alpha):
    # 对第一个列进行Ljung-Box检验，返回p值
    _, pvalue1 = lb_test(df[col1], lags=lags)
    # 对第二个列进行Ljung-Box检验，返回p值
    _, pvalue2 = lb_test(df[col2], lags=lags)

    # 将p值四舍五入到小数点后三位
    pvalue1 = [round(x, 3) for x in pvalue1]
    # 找到第一个p值小于alpha的索引
    index1 = [i for i in range(len(pvalue1)) if pvalue1[i] < alpha][0] + 1
    pvalue2 = [round(x, 3) for x in pvalue2]
    index2 = [i for i in range(len(pvalue2)) if pvalue2[i] < alpha][0] + 1

    # 打印第一个列的结果
    print(col1 + ' p-value < alpha index: ' + str(index1))
    print('p-value: ', pvalue1)
    # 打印第二个列的结果
    print(col2 + ' p-value < alpha index: ' + str(index2))
    print('p-value: ', pvalue2)

    # 创建一个索引列表，用于绘图
    indexs = [i + 1 for i in range(10)]
    # 创建一个图形对象，大小为9x6
    plt.figure(figsize=(9, 6))
    # 调整子图之间的间距
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)

    # 在图形中添加第一个子图，用于绘制第一个列的p值
    ax1 = plt.subplot(211)
    ax1.set_title(col1)
    plt.plot(indexs, pvalue1)

    # 在图形中添加第二个子图，用于绘制第二个列的p值
    ax2 = plt.subplot(212)
    ax2.set_title(col2)
    plt.plot(indexs, pvalue2)

    # 显示图形
    plt.show()


# 定义日期解析函数，将字符串转换为日期对象
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')

# 读取CSV文件，将'date'列解析为日期，并将其设为索引
df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 调用Ljung-Box检验函数，分析两个列的自相关性
ljungbox(df, 'hs300_closing_price', 'hs300_yield_rate', 10, 0.05)