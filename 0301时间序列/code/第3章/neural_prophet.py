import pandas as pd
import matplotlib.pyplot as plt
from neural_prophet import NeuralProphet
from sklearn.metrics import mean_absolute_error


# 结果展示
def show(col, x, yseq, ypred):
    yseq = yseq
    ypred = ypred

    mae = mean_absolute_error(yseq, ypred)

    print(col + ' forecast mae: ', mae)

    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.1, hspace=0.2, wspace=0.05)

    plt.plot(x, yseq)
    plt.plot(x, ypred)
    plt.legend(['yseq', 'ypred'])

    plt.show()


# 节假日与事件设置
def build_holidays():
    # 只考虑影响最大的两个节假日, 春节与国庆
    holiday_history = pd.DataFrame({
        'event': 'h',
        'ds': pd.to_datetime([
            # 春节
            '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
            '2018-02-19', '2018-02-20', '2018-02-21',
            '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07',
            '2019-02-08', '2019-02-09', '2019-02-10',
            '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27',
            '2020-01-28', '2020-01-29', '2020-01-30',
            '2020-01-31', '2020-02-01', '2020-02-02',
            '2021-02-11', '2021-02-12', '2021-02-13', '2021-02-14',
            '2021-02-15', '2021-02-16', '2021-02-17',

            # 端午
            '2018-06-16', '2018-06-17', '2018-06-18',
            '2019-06-07', '2019-06-08', '2019-06-09',
            '2020-06-25', '2020-06-26', '2020-06-27',

            # 国庆
            '2018-10-01', '2018-10-02', '2018-10-03', '2018-10-04',
            '2018-10-05', '2018-10-06', '2018-10-07',
            '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04',
            '2019-10-05', '2019-10-06', '2019-10-07',
            '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-04',
            '2020-10-05', '2020-10-06', '2020-10-07', '2020-10-08'
        ])
    })

    holiday_feature = pd.DataFrame({
        'event': 'h',
        'ds': pd.to_datetime([
            # 端午
            '2021-06-12', '2021-06-13', '2021-06-14'
        ])
    })

    return holiday_history, holiday_feature


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('../data/informations.csv', parse_dates=['date'], date_parser=dateparse)
df = df[df['date'] >= '2018-01-01']

# 模型的输入必须为ds与y
df.rename(columns={'date': 'ds', 'hs300_yield_rate': 'y'}, inplace=True)
df = df[['ds', 'y']]

# 预测步长
periods = 30
train_df = df[:-periods]
test_df = df[-periods:]

model = NeuralProphet(
    # 趋势项、季节参数, 趋势项只能为linear（growth=off没有趋势）
    growth="linear",
    n_changepoints=150,
    changepoints_range=0.9,
    trend_reg=0,
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_reg=1,

    # 自回归参数, 针对AR-Net模型
    n_forecasts=30,
    n_lags=60,
    ar_reg=1,

    # 模型训练参数
    batch_size=32,
    epochs=600,
    learning_rate=0.01
)

# 加入季节项
model.add_seasonality('week', 14, 5)

# 加入事件（节假日）
model.add_events(['h'], lower_window=-1, upper_window=3, regularization=1)

holiday_history, holiday_feature = build_holidays()

history_df = model.create_df_with_events(train_df, holiday_history)

metrics = model.fit(history_df, freq="D")

future = model.make_future_dataframe(history_df, holiday_feature, periods=periods, n_historic_predictions=False)
forecast = model.predict(future)

# 注意这里如果只采用趋势、季节、事件, 预测列为yhat1; 如果采用AR-Net, 预测列为yhat + 预测步长
show('hs300_yield_rate', forecast[-periods:]['ds'], test_df['y'], forecast[-periods:]['yhat30'])
