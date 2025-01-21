import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


class TSCI_Model:
    def __init__(self, seq, col, trend_step, season_step, cycle_step):
        self.trend_step = trend_step
        self.season_step = season_step
        self.cycle_step = cycle_step
        self.index = seq.index
        self.tsci_seq = seq[col].tolist()

    # 趋势拟合
    def build_trend_seq(self, seq):
        trend_seq = []
        if self.trend_step % 2 == 0:
            ma_seq = []
            for i in range(len(seq) - self.trend_step):
                values = seq[i:i + self.trend_step + 1]
                values[0] = values[0] / 2
                values[-1] = values[-1] / 2
                cma_value = sum(values) / self.trend_step
                ma_seq.append(cma_value)

            head_value = ma_seq[0]
            tail_value = ma_seq[-1]
            trend_seq.extend([head_value for i in range(int(self.trend_step / 2))])
            trend_seq.extend(ma_seq)
            trend_seq.extend([tail_value for i in range(int(self.trend_step / 2))])

        if self.trend_step % 2 == 1:
            ma_seq = []
            for i in range(len(seq) - self.trend_step + 1):
                values = seq[i:i + self.trend_step]
                cma_value = sum(values) / self.trend_step
                ma_seq.append(cma_value)

            head_value = ma_seq[0]
            tail_value = ma_seq[-1]
            trend_seq.extend([head_value for i in range(int((self.trend_step - 1) / 2))])
            trend_seq.extend(ma_seq)
            trend_seq.extend([tail_value for i in range(int((self.trend_step - 1) / 2))])

        return trend_seq

    # 季节拟合
    def build_season_seq(self, seq):
        start_index = int(self.season_step / 2)
        end_index = int(len(seq) - self.season_step / 2)
        cma_index = self.index[start_index:end_index]
        cma_ratio_seq = []
        for i in range(len(seq) - self.season_step):
            values = seq[i:i + self.season_step + 1]
            values[0] = values[0] / 2
            values[-1] = values[-1] / 2
            cma_value = sum(values) / self.season_step
            seq_value = seq[start_index + i]
            cma_ratio = seq_value / cma_value
            cma_ratio_seq.append(cma_ratio)

        season_bucket = [[] for i in range(self.season_step)]
        for i in range(len(cma_index)):
            month = str(cma_index[i]).split('-')[1]
            value = cma_ratio_seq[i]
            index = int(month) - 1
            season_bucket[index].append(value)

        season_avg = []
        for season in season_bucket:
            season_avg.append(sum(season) / len(season))

        season_kpi = []
        avg_season_value = sum(season_avg) / self.season_step
        for i in season_avg:
            season_kpi.append(i / avg_season_value)

        season_seq = []
        for date in self.index:
            month = str(date).split('-')[1]
            index = int(month) - 1
            value = season_kpi[index]
            season_seq.append(value)

        return season_seq

    # 循环变动拟合
    def build_cycle_seq(self, seq):
        ma_model = ARIMA(seq, order=(0, 0, self.cycle_step))
        ma_model_fit = ma_model.fit(disp=0)
        cycle_seq = ma_model_fit.fittedvalues

        return cycle_seq

    def fit(self):
        season_seq = self.build_season_seq(self.tsci_seq)
        tci_seq = []
        for i in range(len(self.tsci_seq)):
            tci_seq.append(self.tsci_seq[i] / season_seq[i])
        trend_seq = self.build_trend_seq(tci_seq)

        ci_seq = []
        for i in range(len(self.tsci_seq)):
            ci_seq.append(tci_seq[i] / trend_seq[i])
        cycle_seq = self.build_cycle_seq(ci_seq)

        irregular_seq = []
        for i in range(len(self.tsci_seq)):
            irregular_seq.append(ci_seq[i] / cycle_seq[i])

        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, hspace=0.4, wspace=0.2)
        plt.subplot(311)
        plt.plot(self.index, self.tsci_seq)
        plt.xlabel('date')
        plt.ylabel('origin seq')

        plt.subplot(323)
        plt.plot(self.index, trend_seq)
        plt.xlabel('date')
        plt.ylabel('trend')

        plt.subplot(324)
        plt.plot(self.index, season_seq)
        plt.xlabel('date')
        plt.ylabel('seasonal')

        plt.subplot(325)
        plt.plot(self.index, cycle_seq)
        plt.xlabel('date')
        plt.ylabel('cyclical')

        plt.subplot(326)
        plt.plot(self.index, irregular_seq)
        plt.xlabel('date')
        plt.ylabel('irregular')

        plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

seq = df[['hs300_closing_price']]
model = TSCI_Model(df, 'hs300_closing_price', 12, 12, 5)
model.fit()


import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def decomposing(seq):
    add_decomposition = seasonal_decompose(seq, model='additive', period=90)
    add_trend = add_decomposition.trend
    add_seasonal = add_decomposition.seasonal
    add_residual = add_decomposition.resid

    multi_decomposition = seasonal_decompose(seq, model='multiplicative', period=90)
    multi_trend = multi_decomposition.trend
    multi_seasonal = multi_decomposition.seasonal
    multi_residual = multi_decomposition.resid
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, hspace=0.4, wspace=0.2)

    plt.subplot(411)
    plt.plot(seq)
    plt.xlabel('date')
    plt.ylabel('origin seq')

    plt.subplot(423)
    plt.plot(add_trend)
    plt.xlabel('date')
    plt.ylabel('add trend')

    plt.subplot(425)
    plt.plot(add_seasonal)
    plt.xlabel('date')
    plt.ylabel('add seasonal')

    plt.subplot(427)
    plt.plot(add_residual)
    plt.xlabel('date')
    plt.ylabel('add residual')

    plt.subplot(424)
    plt.plot(multi_trend)
    plt.xlabel('date')
    plt.ylabel('multi trend')

    plt.subplot(426)
    plt.plot(multi_seasonal)
    plt.xlabel('date')
    plt.ylabel('multi seasonal')

    plt.subplot(428)
    plt.plot(multi_residual)
    plt.xlabel('date')
    plt.ylabel('multi residual')

    plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

seq = df[['hs300_closing_price']]
decomposing(seq)