import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler


# 线性回归
def lr_process(df_train, df_test, features, label):
    lr_model = LinearRegression()
    lr_model.fit(df_train[features], df_train[label])

    # 回归系数
    w = lr_model.coef_
    # 截距
    b = lr_model.intercept_

    print('w: ', w)
    print('b: ', b)

    df_train[label + '_pred'] = lr_model.predict(df_train[features])
    df_test[label + '_pred'] = lr_model.predict(df_test[features])

    # 残差评估方法
    train_score = lr_model.score(df_train[features], df_train[label])
    test_score = lr_model.score(df_test[features], df_test[label])

    train_mae = mean_absolute_error(df_train[label], df_train[label + '_pred'])
    test_mae = mean_absolute_error(df_test[label], df_test[label + '_pred'])

    print("train score: ", train_score, '   train mae: ', train_mae)
    print("test score: ", test_score, '   test mae: ', test_mae)

    show(df_train[label], df_train[label + '_pred'], df_test[label], df_test[label + '_pred'], label)


# Lasso线性回归
def lasso_process(df_train, df_test, features, label):
    lassocv_model = LassoCV(cv=15).fit(df_train[features], df_train[label])
    alpha = lassocv_model.alpha_
    print('best alpha: ', alpha)

    # 调节alpha可以实现对拟合的程度
    lr_model = Lasso(max_iter=10000, alpha=alpha)
    lr_model.fit(df_train[features], df_train[label])

    # 回归系数
    w = lr_model.coef_
    # 截距
    b = lr_model.intercept_

    print('w: ', w)
    print('b: ', b)

    df_train[label + '_pred'] = lr_model.predict(df_train[features])
    df_test[label + '_pred'] = lr_model.predict(df_test[features])

    # 残差评估方法
    train_score = lr_model.score(df_train[features], df_train[label])
    test_score = lr_model.score(df_test[features], df_test[label])

    train_mae = mean_absolute_error(df_train[label], df_train[label + '_pred'])
    test_mae = mean_absolute_error(df_test[label], df_test[label + '_pred'])

    print("train score: ", train_score, '   train mae: ', train_mae)
    print("test score: ", test_score, '   test mae: ', test_mae)

    show(df_train[label], df_train[label + '_pred'], df_test[label], df_test[label + '_pred'], label)


def show(train_seq, train_seq_pred, test_seq, test_seq_pred, label):
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(left=0.09, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)

    plt.subplot(2, 1, 1)
    plt.plot(train_seq, label=label + '(train)')
    plt.plot(train_seq_pred, label=label + '_pred' + '(train)')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(test_seq, label=label + '(test)')
    plt.plot(test_seq_pred, label=label + '_pred' + '(test)')
    plt.legend(loc='best')

    plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('../data/informations.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

df = df['2019-01-01':'2021-07-20']
df.dropna(axis=0, how='any', inplace=True)

df['CPI-PPI_YoY'] = df['CPI_YoY'] - df['PPI_YoY']
df['M1-M2_YoY'] = df['M1_YoY'] - df['M2_YoY']
df['PMI_MI-NMI_YoY'] = df['PMI_MI_YoY'] - df['PMI_NMI_YoY']

features = ['financing_balance', 'financing_balance_ratio', 'financing_buy', 'financing_net_buy', '1M', '6M', '1Y', 'ust_closing_price', 'usdx_closing_price', 'CPI-PPI_YoY', 'PMI_MI-NMI_YoY', 'M1-M2_YoY', 'credit_mon_YoY', 'credit_acc_YoY']

# 特征放缩到同一尺度
scaler = MinMaxScaler(feature_range=(0, 100))
scaled_features = scaler.fit_transform(df[features])

df[features] = scaled_features

df_train = df['2019-01-01':'2020-12-31']
df_test = df['2021-01-01':'2021-07-20']

label = 'hs300_closing_price'
lr_process(df_train, df_test, features, label)
lasso_process(df_train, df_test, features, label)

label = 'hs300_yield_rate'
lr_process(df_train, df_test, features, label)
lasso_process(df_train, df_test, features, label)