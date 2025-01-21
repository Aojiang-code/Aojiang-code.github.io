import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# 使用逻辑回归函数进行预测，必须设置最大渐进值cap
def prophet_with_logistic(df, cap, holidays, periods):
    # 创建Prophet模型，使用逻辑增长
    model = Prophet(growth='logistic',
                    n_changepoints=60, changepoint_range=0.9,
                    changepoint_prior_scale=0.1,
                    holidays=holidays, holidays_prior_scale=5)

    # 添加季节性
    model.add_seasonality(name='weekly', period=14, fourier_order=3, prior_scale=0.5)

    # 设置cap列
    df['cap'] = cap

    # 拟合模型
    model.fit(df)
    # 创建未来数据框
    future = model.make_future_dataframe(periods=periods, freq='D')
    future['cap'] = cap
    # 进行预测
    forecast = model.predict(future)

    # 绘制预测结果
    model.plot(forecast, figsize=(9, 5))
    plt.show()

    return forecast


# 使用线性函数进行预测
def prophet_with_linear(df, holidays, periods):
    # 创建Prophet模型，使用线性增长
    model = Prophet(growth='linear',
                    n_changepoints=5, changepoint_range=0.9,
                    changepoint_prior_scale=0.1,
                    holidays=holidays, holidays_prior_scale=5)

    # 添加季节性
    model.add_seasonality(name='weekly', period=14, fourier_order=3, prior_scale=5)

    # 拟合模型
    model.fit(df)
    # 创建未来数据框
    future = model.make_future_dataframe(periods=periods, freq='D')
    # 进行预测
    forecast = model.predict(future)

    # 绘制预测结果
    model.plot(forecast, figsize=(9, 5))
    plt.show()

    return forecast


# 数据展示函数
def show(forecast, col, yseq, periods, mean=0, std=1):
    # 只取预测的最后periods个数据
    forecast = forecast[-periods:]

    x = forecast['ds']

    # 反标准化
    yseq = yseq * std + mean
    yhat = forecast['yhat'] * std + mean
    yhat_lower = forecast['yhat_lower'] * std + mean
    yhat_upper = forecast['yhat_upper'] * std + mean

    # 计算平均绝对误差
    test_mae = mean_absolute_error(yhat, yseq)

    # 打印误差
    print(col + ' forecast mae: ', test_mae)

    # 创建一个图形对象，大小为9x4
    plt.figure(figsize=(9, 4))
    # 调整子图之间的间距
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.1, hspace=0.2, wspace=0.05)

    # 绘制真实值和预测值
    plt.plot(x, yseq)
    plt.plot(x, yhat)
    plt.plot(x, yhat_lower)
    plt.plot(x, yhat_upper)

    plt.legend(['yseq', 'yhat', 'yhat lower', 'yhat upper'])

    # 显示图形
    plt.show()


# 构造节假日函数
# lower_window 节假日前影响范围
# upper_window 节假日后影响范围
def build_holidays(lower_window, upper_window):
    # 元旦
    new_year_day = pd.DataFrame({
        'holiday': 'new_year_day',
        'ds': pd.to_datetime(['2018-01-01', '2018-12-30', '2018-12-31', '2019-01-01',
                              '2020-01-01', '2021-01-01', '2021-01-02', '2021-01-03']),
        'lower_window': lower_window,
        'upper_window': upper_window,
    })

    # 春节
    cn_new_year = pd.DataFrame({
        'holiday': 'cn_new_year',
        'ds': pd.to_datetime(
            ['2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
             '2018-02-19', '2018-02-20', '2018-02-21',
             '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07',
             '2019-02-08', '2019-02-09', '2019-02-10',
             '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27',
             '2020-01-28', '2020-01-29', '2020-01-30',
             '2020-01-31', '2020-02-01', '2020-02-02',
             '2021-02-11', '2021-02-12', '2021-02-13', '2021-02-14', '2021-02-15', '2021-02-16', '2021-02-17']),
        'lower_window': lower_window,
        'upper_window': upper_window,
    })

    # 清明
    clear_and_bright = pd.DataFrame({
        'holiday': 'clear_and_bright',
        'ds': pd.to_datetime(
            ['2018-04-05', '2018-04-06', '2018-04-07',
             '2019-04-05', '2019-04-06', '2019-04-07',
             '2020-04-04', '2020-04-05', '2020-04-06',
             '2021-04-03', '2021-04-04', '2021-04-05'
             ]),
        'lower_window': lower_window,
        'upper_window': upper_window,
    })

    # 五一
    labor_day = pd.DataFrame({
        'holiday': 'labor_day',
        'ds': pd.to_datetime(
            ['2018-04-29', '2018-04-30', '2018-05-01',
             '2019-05-01', '2019-05-02', '2019-05-03', '2019-05-04',
             '2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04',
             '2020-05-05',
             '2021-05-01', '2021-05-02', '2021-05-03', '2021-05-04',
             '2021-05-05'
             ]),
        'lower_window': lower_window,
        'upper_window': upper_window,
    })

    # 端午
    dragon_boat = pd.DataFrame({
        'holiday': 'dragon_boat',
        'ds': pd.to_datetime(
            ['2018-06-16', '2018-06-17', '2018-06-18',
             '2019-06-07', '2019-06-08', '2019-06-09',
             '2020-06-25', '2020-06-26', '2020-06-27',
             '2021-06-12', '2021-06-13', '2021-06-14'
             ]),
        'lower_window': lower_window,
        'upper_window': upper_window,
    })

    # 中秋
    mid_autumn = pd.DataFrame({
        'holiday': 'mid_autumn',
        'ds': pd.to_datetime(
            ['2018-09-22', '2018-09-23', '2018-09-24',
             '2019-09-13', '2019-09-14', '2019-09-15',
             # 2020年中秋国庆连在一起, 因此设置在影响因素更大的国庆假日中
             '2021-09-19', '2021-09-21'
             ]),
        'lower_window': lower_window,
        'upper_window': upper_window,
    })

    # 国庆
    national_day = pd.DataFrame({
        'holiday': 'national_day',
        'ds': pd.to_datetime(
            ['2018-10-01', '2018-10-02', '2018-10-03', '2018-10-04',
             '2018-10-05', '2018-10-06', '2018-10-07',
             '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04',
             '2019-10-05', '2019-10-06', '2019-10-07',
             '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-04',
             '2020-10-05', '2020-10-06', '2020-10-07', '2020-10-08'
             ]),
        'lower_window': lower_window,
        'upper_window': upper_window,
    })

    # 合并所有节假日数据
    holidays = pd.concat([new_year_day, cn_new_year, clear_and_bright,
                          labor_day, dragon_boat, mid_autumn, national_day])

    return holidays


# 定义日期解析函数，将字符串转换为日期对象
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')

# 读取CSV文件，将'date'列解析为日期
df = pd.read_csv('../data/informations.csv', parse_dates=['date'], date_parser=dateparse)

# 筛选数据的时间范围
df = df[df['date'] >= '2018-01-01']
# 重命名列名
df.rename(columns={'date': 'ds'}, inplace=True)

# 预测步长
periods = 30

# 沪深300指数建模
col = 'hs300_closing_price'
# 计算均值和标准差
mean = df[col].mean()
std = df[col].std()
# 序列标准化
df[col] = (df[col] - mean) / std
# 划分训练集、测试集
train_df = df[:-periods]
test_df = df[-periods:]
# 重命名列名
train_df.rename(columns={col: 'y'}, inplace=True)
# 使用线性增长模型进行预测
forecast = prophet_with_linear(train_df, build_holidays(-7, 7), periods)
# 显示预测结果
show(forecast, col, test_df[col], periods, mean, std)

# 删除训练集中的'y'列
del train_df['y']

# 沪深300日收益率建模
col = 'hs300_yield_rate'
# 重命名列名
train_df.rename(columns={col: 'y'}, inplace=True)
# 使用逻辑增长模型进行预测
forecast = prophet_with_logistic(train_df, 0.05, build_holidays(-2, 1), periods)
# 显示预测结果
show(forecast, col, test_df[col], periods)



# 在这段代码中，使用 `Prophet` 模型进行时间序列预测。代码首先定义了两个预测函数 `prophet_with_logistic` 和 `prophet_with_linear`，分别用于逻辑增长和线性增长。然后，定义了一个展示预测结果的函数 `show` 和一个构建节假日数据的函数 `build_holidays`。代码读取数据文件并进行预处理，创建并训练 `Prophet` 模型，最后进行预测并展示结果。
