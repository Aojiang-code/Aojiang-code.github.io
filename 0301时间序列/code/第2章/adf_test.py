import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('../data/informations.csv', parse_dates=['date'],
                 index_col='date', date_parser=dateparse)

# 做一阶差分
df['hs300_closing_price_diff1'] = df['hs300_closing_price'].diff(1)

closing_price_adf = ADF(df['hs300_closing_price'].tolist())
closing_price_diff1_adf = ADF(df['hs300_closing_price_diff1'].tolist()[1:])
yield_rate_adf = ADF(df['hs300_yield_rate'].tolist())

print('hs300_closing_price_adf : ', closing_price_adf)
print('hs300_closing_price_diff1_adf : ', closing_price_diff1_adf)
print('hs300_yield_rate_adf : ', yield_rate_adf)

fig = plt.figure(figsize=(9, 6))
fig.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=None, hspace=0.3)
plt.subplot(211)
plt.title('hs300_closing_price')
plt.plot(df['hs300_closing_price'])

plt.subplot(212)
plt.title('hs300_closing_price_diff1')
plt.plot(df['hs300_closing_price_diff1'])

plt.show()