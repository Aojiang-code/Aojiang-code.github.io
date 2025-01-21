import numpy as np
import matplotlib.pyplot as plt

# 定义逻辑函数
def logistic(x, c, k, m):
    # 返回逻辑函数的计算结果
    return c / (1 + np.exp(-k * (x - m)))

# 生成从-6到6的500个等间距的点
x = np.linspace(-6, 6, 500)

# 计算不同参数下的逻辑函数值
y0 = logistic(x, c=1, k=1, m=0)
y1 = logistic(x, c=2, k=1, m=0)
y2 = logistic(x, c=1, k=2, m=0)
y3 = logistic(x, c=1, k=1, m=1)

# 创建一个图形对象，大小为9x5
plt.figure(figsize=(9, 5))
# 调整子图之间的间距
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.2, wspace=0.2)

# 在图形中添加第一个子图，绘制y0
plt.subplot(2, 2, 1)
plt.plot(x, y0)
plt.legend(['y0: C=1, k=1, m=0'])

# 在图形中添加第二个子图，绘制y1
plt.subplot(2, 2, 2)
plt.plot(x, y1)
plt.legend(['y1: C=2, k=1, m=0'])

# 在图形中添加第三个子图，绘制y2
plt.subplot(2, 2, 3)
plt.plot(x, y2)
plt.legend(['y2: C=1, k=2, m=0'])

# 在图形中添加第四个子图，绘制y3
plt.subplot(2, 2, 4)
plt.plot(x, y3)
plt.legend(['y3: C=1, k=1, m=1'])

# 显示图形
plt.show()

# 在这段代码中，定义了一个逻辑函数 `logistic`，并使用不同的参数计算其值。然后，代码创建一个图形对象，绘制了四个不同参数下的逻辑函数曲线。通过这些图形，可以观察参数对逻辑函数形状的影响。