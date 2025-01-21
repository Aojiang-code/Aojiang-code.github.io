from scipy import stats

t_value = 0.93
crit = stats.t.ppf(q=0.95, df=4)
P_value = 1 - stats.t.cdf(x=t_value, df=4)

print('T统计量：', t_value)
print('T临界值：', crit)
print('P-value：', P_value)