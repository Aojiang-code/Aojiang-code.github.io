import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# 趋势项采用逻辑回归函数, 采用逻辑回归必须设置最大渐进值C(t):cap
def prophet_with_logistic(df, cap, holidays, periods):
    model = Prophet(growth='logistic',
                    n_changepoints=60, changepoint_range=0.9,
                    changepoint_prior_scale=0.1,
                    holidays=holidays, holidays_prior_scale=5)

    model.add_seasonality(name='weekly', period=14, fourier_order=3, prior_scale=0.5)

    df['cap'] = cap

    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='D')
    future['cap'] = cap
    forecast = model.predict(future)

    model.plot(forecast, figsize=(9, 5))
    plt.show()

    return forecast


# 趋势项采用分段线性函数
def prophet_with_linear(df, holidays, periods):
    model = Prophet(growth='linear',
                    n_changepoints=5, changepoint_range=0.9,
                    changepoint_prior_scale=0.1,
                    holidays=holidays, holidays_prior_scale=5)

    model.add_seasonality(name='weekly', period=14, fourier_order=3, prior_scale=5)

    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    model.plot(forecast, figsize=(9, 5))
    plt.show()

    return forecast


# 数据展示
def show(forecast, col, yseq, periods, mean=0, std=1):
    forecast = forecast[-periods:]

    x = forecast['ds']

    yseq = yseq * std + mean
    yhat = forecast['yhat'] * std + mean
    yhat_lower = forecast['yhat_lower'] * std + mean
    yhat_upper = forecast['yhat_upper'] * std + mean

    test_mae = mean_absolute_error(yhat, yseq)

    print(col + ' forecast mae: ', test_mae)

    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.1, hspace=0.2, wspace=0.05)

    plt.plot(x, yseq)
    plt.plot(x, yhat)
    plt.plot(x, yhat_lower)
    plt.plot(x, yhat_upper)

    plt.legend(['yseq', 'yhat', 'yhat lower', 'yhat upper'])

    plt.show()


# 构造节假日
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

    holidays = pd.concat([new_year_day, cn_new_year, clear_and_bright,
                          labor_day, dragon_boat, mid_autumn, national_day])

    return holidays


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('../data/informations.csv', parse_dates=['date'], date_parser=dateparse)
df = df[df['date'] >= '2018-01-01']
df.rename(columns={'date': 'ds'}, inplace=True)

# 预测步长
periods = 30

# 沪深300指数建模
col = 'hs300_closing_price'
mean = df[col].mean()
std = df[col].std()
# 序列标准化
df[col] = (df[col] - mean) / std
# 划分训练集、测试集
train_df = df[:-periods]
test_df = df[-periods:]
train_df.rename(columns={col: 'y'}, inplace=True)
forecast = prophet_with_linear(train_df, build_holidays(-7, 7), periods)
show(forecast, col, test_df[col], periods, mean, std)

del train_df['y']

# 沪深300日收益率建模
col = 'hs300_yield_rate'
train_df.rename(columns={col: 'y'}, inplace=True)
forecast = prophet_with_logistic(train_df, 0.05, build_holidays(-2, 1), periods)
show(forecast, col, test_df[col], periods)
