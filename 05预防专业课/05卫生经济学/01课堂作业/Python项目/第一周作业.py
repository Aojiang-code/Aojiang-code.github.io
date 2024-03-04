pip install wordcloud matplotlib

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.dates as mdates

# 准备词云数据
words = defaultdict(int)
words['Health Expenditure'] = 1950
words['Healthcare'] = 1950
words['Public Health'] = 1950
words['Cost-Benefit Analysis'] = 1960
words['Resource Allocation'] = 1960
words['Medical Security'] = 1960
# ... 添加更多年代的词汇和频率

# 创建词云
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)

# 显示词云
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# 添加时间轴
plt.gca().xaxis.set_major_locator(mdates.YearLocator(yearinterval=10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_tick_params(pad=10)
plt.gca().xaxis.set_tick_params(labelrotation=45)

# 显示图表
plt.show()