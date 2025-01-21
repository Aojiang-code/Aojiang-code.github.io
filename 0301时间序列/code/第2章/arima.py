import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test

import warnings

warnings.filterwarnings("ignore")


# 模型阶数选择
def select_p_q(seq, ic='aic'):
    trend_evaluate = sm.tsa.arma_order_select_ic(seq, ic=['aic', 'bic'], trend='nc', max_ar=6, max_ma=6)

    print('AIC', trend_evaluate.aic_min_order)
    print('BIC', trend_evaluate.bic_min_order)

    if ic == 'aic':
        return trend_evaluate.aic_min_order
    elif ic == 'bic':
        return trend_evaluate.bic_min_order
    else:
        return None


# 采用ARIMA模型预测
def arima_pred(seq, train_step, pred_step, order):
    seq_len = len(seq)
    index = seq.index.tolist()[train_step:]
    # 在使用ARIMA模型时, 建议使用连续索引
    seq.index = [i for i in range(seq_len)]

    pred_seq = []
    # 采用滑动法, 每次预测pred_step个步长
    for i in range((seq_len - train_step) // pred_step):
        train_seq = seq[i * pred_step:i * pred_step + train_step]
        train_seq.index = [i for i in range(train_step)]

        model = sm.tsa.SARIMAX(train_seq, order=order).fit(disp=0, trend='c')
        start = train_step
        end = train_step + pred_step - 1
        pred_seq.extend(model.predict(start=start, end=end, dynamic=True))

    # 对末尾无法预测pred_step个步长做处理
    resi_step = (seq_len - train_step) % pred_step
    if resi_step != 0:
        train_seq = seq[-resi_step - train_step: -resi_step]
        train_seq.index = [i for i in range(train_step)]

        model = sm.tsa.SARIMAX(train_seq, order=order).fit(disp=0, trend='c')
        start = train_step
        end = train_step + resi_step - 1
        pred_seq.extend(model.predict(start=start, end=end, dynamic=True))

    return pd.Series(index=index, data=seq.tolist()[train_step:]), pd.Series(index=index, data=pred_seq)


# 白噪声检验（Ljung-Box检验）
def wn_test(seq, lags, alpha):
    _, pvalues = lb_test(seq, lags)

    for pvalue in pvalues:
        if pvalue < alpha:
            return False

    return True


# 模型效果展示
def show(col, origin_seq, pred_seq):
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(left=0.09, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)

    residual_seq = origin_seq - pred_seq

    plt.subplot(2, 1, 1)
    plt.plot(origin_seq)
    plt.plot(pred_seq)
    plt.legend([col + '(origin seq)', col + '(pred seq)'])

    plt.subplot(2, 1, 2)
    plt.plot(residual_seq)
    plt.legend([col + '(residual seq)'])

    plt.show()

    # 校验结果
    is_wn = wn_test(residual_seq, 1, 0.05)
    mae = mean_absolute_error(origin_seq, pred_seq)
    print('residual seq is white noise: ', is_wn)
    print('pred mae: ', mae)


# arima预测
def arima_run(df, col, is_diff, train_step, pred_step):
    seq = df[col]
    if is_diff:
        index = df.index.tolist()
        start_val = seq[0]
        # 一阶差分后第一个元素为NaN, 需要剔除
        seq_diff = seq.diff(1)[1:]

        # 差分序列预测
        p, q = select_p_q(seq_diff)
        origin_seq_diff, pred_seq_diff = arima_pred(seq_diff, train_step, pred_step, (p, 0, q))
        origin_seq_diff.index = index[-len(origin_seq_diff):]
        pred_seq_diff.index = index[-len(pred_seq_diff):]
        show(col + '_diff', origin_seq_diff, pred_seq_diff)

        # 还原为原序列
        origin_seq = origin_seq_diff.cumsum() + start_val
        pred_seq = pred_seq_diff.cumsum() + start_val
        origin_seq.index = index[-len(origin_seq):]
        pred_seq.index = index[-len(pred_seq):]
        show(col, origin_seq, pred_seq)

    else:
        p, q = select_p_q(seq)
        origin_seq, pred_seq = arima_pred(seq, train_step, pred_step, (p, 0, q))
        show(col, origin_seq, pred_seq)


# 随机游走预测
def random_walk_run(df, col):
    df[col + '_random_walk'] = [0] + df[col].tolist()[:-1]

    index = df.index.tolist()[1:]
    origin_seq = pd.Series(index=index, data=df[col][1:].tolist())
    pred_seq = pd.Series(index=index, data=df[col + '_random_walk'][1:].tolist())

    show(col, origin_seq, pred_seq)


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('./data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 获取2019-07-20开始的数据
df = df[df.index >= '2019-07-20']

# 滑窗法, 拟合train_step长度的数据, 向后预测pred_step长度的数据
train_step = 30
pred_step = 1

# 沪深300日收益率预测
col = 'hs300_yield_rate'
print(col + '(arima)')
arima_run(df, col, False, train_step, pred_step)

# 沪深300指数预测（AR）
col = 'hs300_closing_price'
print(col + '(ar)')
arima_run(df, col, False, train_step, pred_step)

# 沪深300指数预测（一阶差分ARIMA）
print(col + '(arima diff=1)')
arima_run(df, col, True, train_step, pred_step)

# 沪深300指数预测（随机游走）
print(col + '(random walk)')
random_walk_run(df, col)