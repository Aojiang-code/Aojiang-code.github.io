import numpy as np
import matplotlib.pyplot as plt


def logistic(x, c, k, m):
    return c / (1 + np.exp(-k * (x - m)))


x = np.linspace(-6, 6, 500)
y0 = logistic(x, c=1, k=1, m=0)
y1 = logistic(x, c=2, k=1, m=0)
y2 = logistic(x, c=1, k=2, m=0)
y3 = logistic(x, c=1, k=1, m=1)

plt.figure(figsize=(9, 5))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.2, wspace=0.2)

plt.subplot(2, 2, 1)
plt.plot(x, y0)
plt.legend(['y0: C=1, k=1, m=0'])

plt.subplot(2, 2, 2)
plt.plot(x, y1)
plt.legend(['y1: C=2, k=1, m=0'])

plt.subplot(2, 2, 3)
plt.plot(x, y2)
plt.legend(['y2: C=1, k=2, m=0'])

plt.subplot(2, 2, 4)
plt.plot(x, y3)
plt.legend(['y3: C=1, k=1, m=1'])

plt.show()