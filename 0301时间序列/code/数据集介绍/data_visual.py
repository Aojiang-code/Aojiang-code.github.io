import pandas as pd
import matplotlib.pyplot as plt


# hs300数据表展示
def hs300_show(hs300_file):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df = pd.read_csv(hs300_file, parse_dates=['date'], date_parser=dateparse, index_col='date')

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.08, hspace=0.4, wspace=0.05)
    plt.subplot(4, 1, 1)
    plt.plot(df['hs300_closing_price'])
    plt.xlabel('date')
    plt.ylabel('hs300_closing_price')

    plt.subplot(4, 1, 2)
    plt.plot(df['hs300_yield_rate'])
    plt.xlabel('date')
    plt.ylabel('hs300_yield_rate')

    plt.subplot(4, 1, 3)
    plt.plot(df['financing_balance'])
    plt.xlabel('date')
    plt.ylabel('financing_balance')

    plt.subplot(4, 1, 4)
    plt.plot(df['financing_balance_ratio'])
    plt.xlabel('date')
    plt.ylabel('financing_balance_ratio')

    plt.show()


# econ数据表展示
def econ_show(econ_file):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    df = pd.read_csv(econ_file, parse_dates=['date'], date_parser=dateparse, index_col='date')

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.4, wspace=0.05)
    plt.subplot(3, 1, 1)
    plt.plot(df['CPI_YoY'])
    plt.plot(df['PPI_YoY'])
    plt.legend(['CPI_YoY', 'PPI_YoY'])
    plt.xlabel('date')
    plt.ylabel('YoY')

    plt.subplot(3, 1, 2)
    plt.plot(df['PMI_MI_YoY'])
    plt.plot(df['PMI_NMI_YoY'])
    plt.legend(['PMI_MI_YoY', 'PMI_NMI_YoY'])
    plt.xlabel('date')
    plt.ylabel('YoY')

    plt.subplot(3, 1, 3)
    plt.plot(df['M1_YoY'], linestyle='-')
    plt.plot(df['M2_YoY'], linestyle='--')
    plt.legend(['M1_YoY', 'M2_YoY'])
    plt.xlabel('date')
    plt.ylabel('YoY')

    plt.show()


# credit数据表展示
def credit_show(credit_file):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    df = pd.read_csv(credit_file, parse_dates=['date'], date_parser=dateparse, index_col='date')

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.4, wspace=0.05)
    plt.subplot(3, 1, 1)
    plt.plot(df['credit_mon_val'])
    plt.xlabel('date')
    plt.ylabel('credit_mon_val')

    plt.subplot(3, 1, 2)
    plt.plot(df['credit_acc_val'])
    plt.xlabel('date')
    plt.ylabel('credit_acc_val')

    plt.subplot(3, 1, 3)
    plt.plot(df['credit_mon_YoY'], linestyle='-')
    plt.plot(df['credit_acc_YoY'], linestyle='--')
    plt.legend(['credit_mon_YoY', 'credit_acc_YoY'])
    plt.xlabel('date')
    plt.ylabel('YoY')

    plt.show()


# us_index数据表展示
def us_index_show(us_index_file):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df = pd.read_csv(us_index_file, parse_dates=['date'], date_parser=dateparse, index_col='date')

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.2, wspace=0.05)
    plt.subplot(2, 1, 1)
    plt.plot(df['ust_closing_price'])
    plt.xlabel('date')
    plt.ylabel('ust_closing_price')

    plt.subplot(2, 1, 2)
    plt.plot(df['usdx_closing_price'])
    plt.xlabel('date')
    plt.ylabel('usdx_closing_price')

    plt.show()


# shibor数据表展示
def shibor_show(shibor_file):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df = pd.read_csv(shibor_file, parse_dates=['date'], date_parser=dateparse, index_col='date')

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.08, hspace=0.3, wspace=0.05)
    plt.subplot(3, 1, 1)
    plt.plot(df['O/N'], label='O/N')
    plt.xlabel('date')
    plt.ylabel('O/N')

    plt.subplot(3, 1, 2)
    plt.plot(df['6M'], label='6M')
    plt.xlabel('date')
    plt.ylabel('6M')

    plt.subplot(3, 1, 3)
    plt.plot(df['1Y'], label='1Y')
    plt.xlabel('date')
    plt.ylabel('1Y')

    plt.show()


hs300_file = '../data/hs300.csv'
econ_file = '../data/econ.csv'
credit_file = '../data/credit.csv'
us_index_file = '../data/us_index.csv'
shibor_file = '../data/shibor.csv'

hs300_show(hs300_file)
econ_show(econ_file)
credit_show(credit_file)
us_index_show(us_index_file)
shibor_show(shibor_file)