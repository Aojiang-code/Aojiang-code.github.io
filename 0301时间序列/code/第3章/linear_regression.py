import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler


# 线性回归处理函数
def lr_process(df_train, df_test, features, label):
    # 创建线性回归模型
    lr_model = LinearRegression()
    # 使用训练数据拟合模型
    lr_model.fit(df_train[features], df_train[label])

    # 获取回归系数
    w = lr_model.coef_
    # 获取截距
    b = lr_model.intercept_

    # 打印回归系数和截距
    print('w: ', w)
    print('b: ', b)

    # 预测训练集和测试集的标签
    df_train[label + '_pred'] = lr_model.predict(df_train[features])
    df_test[label + '_pred'] = lr_model.predict(df_test[features])

    # 计算训练集和测试集的得分
    train_score = lr_model.score(df_train[features], df_train[label])
    test_score = lr_model.score(df_test[features], df_test[label])

    # 计算训练集和测试集的平均绝对误差
    train_mae = mean_absolute_error(df_train[label], df_train[label + '_pred'])
    test_mae = mean_absolute_error(df_test[label], df_test[label + '_pred'])

    # 打印得分和误差
    print("train score: ", train_score, '   train mae: ', train_mae)
    print("test score: ", test_score, '   test mae: ', test_mae)

    # 显示预测结果
    show(df_train[label], df_train[label + '_pred'], df_test[label], df_test[label + '_pred'], label)


# Lasso线性回归处理函数
def lasso_process(df_train, df_test, features, label):
    # 使用LassoCV选择最佳alpha
    lassocv_model = LassoCV(cv=15).fit(df_train[features], df_train[label])
    alpha = lassocv_model.alpha_
    # 打印最佳alpha值
    print('best alpha: ', alpha)

    # 创建Lasso模型并使用最佳alpha值
    lr_model = Lasso(max_iter=10000, alpha=alpha)
    # 使用训练数据拟合模型
    lr_model.fit(df_train[features], df_train[label])

    # 获取回归系数
    w = lr_model.coef_
    # 获取截距
    b = lr_model.intercept_

    # 打印回归系数和截距
    print('w: ', w)
    print('b: ', b)

    # 预测训练集和测试集的标签
    df_train[label + '_pred'] = lr_model.predict(df_train[features])
    df_test[label + '_pred'] = lr_model.predict(df_test[features])

    # 计算训练集和测试集的得分
    train_score = lr_model.score(df_train[features], df_train[label])
    test_score = lr_model.score(df_test[features], df_test[label])

    # 计算训练集和测试集的平均绝对误差
    train_mae = mean_absolute_error(df_train[label], df_train[label + '_pred'])
    test_mae = mean_absolute_error(df_test[label], df_test[label + '_pred'])

    # 打印得分和误差
    print("train score: ", train_score, '   train mae: ', train_mae)
    print("test score: ", test_score, '   test mae: ', test_mae)

    # 显示预测结果
    show(df_train[label], df_train[label + '_pred'], df_test[label], df_test[label + '_pred'], label)


# 显示预测结果的函数
def show(train_seq, train_seq_pred, test_seq, test_seq_pred, label):
    # 创建一个图形对象，大小为9x6
    plt.figure(figsize=(9, 6))
    # 调整子图之间的间距
    plt.subplots_adjust(left=0.09, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)

    # 绘制训练集的真实值和预测值
    plt.subplot(2, 1, 1)
    plt.plot(train_seq, label=label + '(train)')
    plt.plot(train_seq_pred, label=label + '_pred' + '(train)')
    plt.legend(loc='best')

    # 绘制测试集的真实值和预测值
    plt.subplot(2, 1, 2)
    plt.plot(test_seq, label=label + '(test)')
    plt.plot(test_seq_pred, label=label + '_pred' + '(test)')
    plt.legend(loc='best')

    # 显示图形
    plt.show()


# 定义日期解析函数，将字符串转换为日期对象
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')

# 读取CSV文件，将'date'列解析为日期，并将其设为索引
df = pd.read_csv('../data/informations.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

# 筛选数据的时间范围
df = df['2019-01-01':'2021-07-20']
# 删除缺失值
df.dropna(axis=0, how='any', inplace=True)

# 计算新的特征
df['CPI-PPI_YoY'] = df['CPI_YoY'] - df['PPI_YoY']
df['M1-M2_YoY'] = df['M1_YoY'] - df['M2_YoY']
df['PMI_MI-NMI_YoY'] = df['PMI_MI_YoY'] - df['PMI_NMI_YoY']

# 定义特征列表
features = ['financing_balance', 'financing_balance_ratio', 'financing_buy', 'financing_net_buy', '1M', '6M', '1Y', 'ust_closing_price', 'usdx_closing_price', 'CPI-PPI_YoY', 'PMI_MI-NMI_YoY', 'M1-M2_YoY', 'credit_mon_YoY', 'credit_acc_YoY']

# 特征放缩到同一尺度
scaler = MinMaxScaler(feature_range=(0, 100))
scaled_features = scaler.fit_transform(df[features])

# 更新数据框中的特征
df[features] = scaled_features

# 划分训练集和测试集
df_train = df['2019-01-01':'2020-12-31']
df_test = df['2021-01-01':'2021-07-20']

# 定义标签
label = 'hs300_closing_price'
# 调用线性回归处理函数
lr_process(df_train, df_test, features, label)
# 调用Lasso线性回归处理函数
lasso_process(df_train, df_test, features, label)

# 更改标签
label = 'hs300_yield_rate'
# 调用线性回归处理函数
lr_process(df_train, df_test, features, label)
# 调用Lasso线性回归处理函数
lasso_process(df_train, df_test, features, label)