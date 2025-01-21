import numpy as np
from scipy import stats

observed_value = np.array([732, 467, 363, 321])
expected_value = np.array([700, 499, 395, 289])
chi_squared_value = ((observed_value - expected_value) ** 2 / expected_value).sum()

# 卡方检验的显著性水平为5%，自由度为1
crit = stats.chi2.ppf(q=0.95, df=1)
P_value = 1 - stats.chi2.cdf(x=chi_squared_value, df=1)

print('卡方统计量：', chi_squared_value)
print('卡方临界值：', crit)
print('P-value：', P_value)