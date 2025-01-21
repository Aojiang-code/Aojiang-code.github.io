#    pip install statsmodels

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# 定义时间序列分解模型类
class TSCI_Model:
    def __init__(self, seq, col, trend_step, season_step, cycle_step):
        # 初始化模型参数
        self.trend_step = trend_step  # 趋势步长
        self.season_step = season_step  # 季节步长
        self.cycle_step = cycle_step  # 循环步长
        self.index = seq.index  # 时间序列索引
        self.tsci_seq = seq[col].tolist()  # 将指定列的数据转换为列表形式

    # 构建趋势序列
    def build_trend_seq(self, seq):
        trend_seq = []  # 初始化趋势序列
        # 如果趋势步长为偶数
        if self.trend_step % 2 == 0:
            ma_seq = []  # 移动平均序列
            for i in range(len(seq) - self.trend_step):
                values = seq[i:i + self.trend_step + 1]  # 获取当前窗口的值
                values[0] = values[0] / 2  # 首尾值减半
                values[-1] = values[-1] / 2
                cma_value = sum(values) / self.trend_step  # 计算中心移动平均
                ma_seq.append(cma_value)

            head_value = ma_seq[0]  # 获取头部值
            tail_value = ma_seq[-1]  # 获取尾部值
            trend_seq.extend([head_value for i in range(int(self.trend_step / 2))])  # 填充头部
            trend_seq.extend(ma_seq)  # 添加移动平均序列
            trend_seq.extend([tail_value for i in range(int(self.trend_step / 2))])  # 填充尾部

        # 如果趋势步长为奇数
        if self.trend_step % 2 == 1:
            ma_seq = []  # 移动平均序列
            for i in range(len(seq) - self.trend_step + 1):
                values = seq[i:i + self.trend_step]  # 获取当前窗口的值
                cma_value = sum(values) / self.trend_step  # 计算中心移动平均
                ma_seq.append(cma_value)

            head_value = ma_seq[0]  # 获取头部值
            tail_value = ma_seq[-1]  # 获取尾部值
            trend_seq.extend([head_value for i in range(int((self.trend_step - 1) / 2))])  # 填充头部
            trend_seq.extend(ma_seq)  # 添加移动平均序列
            trend_seq.extend([tail_value for i in range(int((self.trend_step - 1) / 2))])  # 填充尾部

        return trend_seq  # 返回趋势序列

    # 构建季节序列
    def build_season_seq(self, seq):
        start_index = int(self.season_step / 2)  # 计算起始索引
        end_index = int(len(seq) - self.season_step / 2)  # 计算结束索引
        cma_index = self.index[start_index:end_index]  # 中心移动平均索引
        cma_ratio_seq = []  # 中心移动平均比率序列
        for i in range(len(seq) - self.season_step):
            values = seq[i:i + self.season_step + 1]  # 获取当前窗口的值
            values[0] = values[0] / 2  # 首尾值减半
            values[-1] = values[-1] / 2
            cma_value = sum(values) / self.season_step  # 计算中心移动平均
            seq_value = seq[start_index + i]  # 获取当前序列值
            cma_ratio = seq_value / cma_value  # 计算比率
            cma_ratio_seq.append(cma_ratio)

        season_bucket = [[] for i in range(self.season_step)]  # 初始化季节桶
        for i in range(len(cma_index)):
            month = str(cma_index[i]).split('-')[1]  # 提取月份
            value = cma_ratio_seq[i]  # 获取比率值
            index = int(month) - 1  # 计算索引
            season_bucket[index].append(value)  # 将值添加到对应的季节桶中

        season_avg = []  # 季节平均值
        for season in season_bucket:
            season_avg.append(sum(season) / len(season))  # 计算每月平均值

        season_kpi = []  # 季节关键绩效指标
        avg_season_value = sum(season_avg) / self.season_step  # 计算平均季节值
        for i in season_avg:
            season_kpi.append(i / avg_season_value)  # 归一化

        season_seq = []  # 季节序列
        for date in self.index:
            month = str(date).split('-')[1]  # 提取月份
            index = int(month) - 1  # 计算索引
            value = season_kpi[index]  # 获取季节KPI值
            season_seq.append(value)  # 添加到季节序列中

        return season_seq  # 返回季节序列

    # 构建循环序列
    def build_cycle_seq(self, seq):
        ma_model = ARIMA(seq, order=(0, 0, self.cycle_step))  # 使用新的ARIMA模型
        ma_model_fit = ma_model.fit()  # 拟合模型
        cycle_seq = ma_model_fit.fittedvalues  # 获取拟合值

        return cycle_seq  # 返回循环序列

    # 拟合模型
    def fit(self):
        season_seq = self.build_season_seq(self.tsci_seq)  # 构建季节序列
        tci_seq = []  # 去除季节成分后的序列
        for i in range(len(self.tsci_seq)):
            tci_seq.append(self.tsci_seq[i] / season_seq[i])  # 去除季节成分
        trend_seq = self.build_trend_seq(tci_seq)  # 构建趋势序列

        ci_seq = []  # 去除趋势成分后的序列
        for i in range(len(self.tsci_seq)):
            ci_seq.append(tci_seq[i] / trend_seq[i])  # 去除趋势成分
        cycle_seq = self.build_cycle_seq(ci_seq)  # 构建循环序列

        irregular_seq = []  # 去除循环成分后的序列
        for i in range(len(self.tsci_seq)):
            irregular_seq.append(ci_seq[i] / cycle_seq[i])  # 去除循环成分

        # 绘制图形
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, hspace=0.4, wspace=0.2)
        plt.subplot(311)
        plt.plot(self.index, self.tsci_seq)  # 绘制原始序列
        plt.xlabel('date')
        plt.ylabel('origin seq')

        plt.subplot(323)
        plt.plot(self.index, trend_seq)  # 绘制趋势序列
        plt.xlabel('date')
        plt.ylabel('trend')

        plt.subplot(324)
        plt.plot(self.index, season_seq)  # 绘制季节序列
        plt.xlabel('date')
        plt.ylabel('seasonal')

        plt.subplot(325)
        plt.plot(self.index, cycle_seq)  # 绘制循环序列
        plt.xlabel('date')
        plt.ylabel('cyclical')

        plt.subplot(326)
        plt.plot(self.index, irregular_seq)  # 绘制不规则序列
        plt.xlabel('date')
        plt.ylabel('irregular')

        plt.show()  # 显示图形

# 日期解析函数
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')

# 读取数据
df = pd.read_csv('../../data/informations.csv', parse_dates=['date'], index_col='date')

# 如果需要自定义日期解析，可以在读取后使用
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

# 提取时间序列
seq = df[['hs300_closing_price']]
# 创建并拟合模型
model = TSCI_Model(df, 'hs300_closing_price', 12, 12, 5)
model.fit()

# 导入必要的库
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 定义分解函数
def decomposing(seq):
    # 加性分解
    add_decomposition = seasonal_decompose(seq, model='additive', period=90)
    add_trend = add_decomposition.trend  # 获取加性分解的趋势成分
    add_seasonal = add_decomposition.seasonal  # 获取加性分解的季节成分
    add_residual = add_decomposition.resid  # 获取加性分解的残差成分

    # 乘性分解
    multi_decomposition = seasonal_decompose(seq, model='multiplicative', period=90)
    multi_trend = multi_decomposition.trend  # 获取乘性分解的趋势成分
    multi_seasonal = multi_decomposition.seasonal  # 获取乘性分解的季节成分
    multi_residual = multi_decomposition.resid  # 获取乘性分解的残差成分

    # 绘制分解结果
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, hspace=0.4, wspace=0.2)

    plt.subplot(411)
    plt.plot(seq)  # 绘制原始序列
    plt.xlabel('date')
    plt.ylabel('origin seq')

    plt.subplot(423)
    plt.plot(add_trend)  # 绘制加性分解的趋势成分
    plt.xlabel('date')
    plt.ylabel('add trend')

    plt.subplot(425)
    plt.plot(add_seasonal)  # 绘制加性分解的季节成分
    plt.xlabel('date')
    plt.ylabel('add seasonal')

    plt.subplot(427)
    plt.plot(add_residual)  # 绘制加性分解的残差成分
    plt.xlabel('date')
    plt.ylabel('add residual')

    plt.subplot(424)
    plt.plot(multi_trend)  # 绘制乘性分解的趋势成分
    plt.xlabel('date')
    plt.ylabel('multi trend')

    plt.subplot(426)
    plt.plot(multi_seasonal)  # 绘制乘性分解的季节成分
    plt.xlabel('date')
    plt.ylabel('multi seasonal')

    plt.subplot(428)
    plt.plot(multi_residual)  # 绘制乘性分解的残差成分
    plt.xlabel('date')
    plt.ylabel('multi residual')

    plt.show()  # 显示图形

# 日期解析函数
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

# 读取数据
df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 提取时间序列
seq = df[['hs300_closing_price']]
# 调用分解函数
decomposing(seq)