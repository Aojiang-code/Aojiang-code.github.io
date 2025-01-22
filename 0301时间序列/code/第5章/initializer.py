import numpy as np


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# glorot(xavier)初始化
x0 = np.random.normal(0, 1, [256, 1])
x1 = x0

for i in range(100):
    # uniform 矩阵相乘
    w0 = np.random.uniform(-1, 1, [256, 256])
    x0 = np.matmul(w0, x0)
    x0 = tanh(x0)

    # glorot 矩阵相乘
    w1 = w0 * np.sqrt(6 / (256 + 256))
    x1 = np.matmul(w1, x1)
    x1 = tanh(x1)

print('glorot:')
print('uniform mean: {0}, std: {1}'.format(np.mean(x0), np.std(x0)))
print('glorot mean: {0}, std: {1}'.format(np.mean(x1), np.std(x1)))

# he(kaiming)初始化
x0 = np.random.normal(0, 1, [256, 1])
x1 = x0

for i in range(100):
    # uniform 矩阵相乘
    w0 = np.random.uniform(-1, 1, [256, 256])
    x0 = np.matmul(w0, x0)
    x0 = relu(x0)

    # he 矩阵相乘
    w1 = w0 * np.sqrt(6 / (256))
    x1 = np.matmul(w1, x1)
    x1 = relu(x1)

print('he:')
print('uniform mean: {0}, std: {1}'.format(np.mean(x0), np.std(x0)))
print('he mean: {0}, std: {1}'.format(np.mean(x1), np.std(x1)))

# orthogonal初始化
x0 = np.random.normal(0, 1, [256, 1])
x1 = x0

for i in range(100):
    # normal 矩阵相乘
    w0 = np.random.normal(0, 1, [256, 256])
    x0 = np.matmul(w0, x0)

    # orthogonal 矩阵相乘
    u, _, v = np.linalg.svd(w0, full_matrices=False)
    w1 = u
    x1 = np.matmul(w1, x1)

print('orthogonal:')
print('normal mean: {0}, std: {1}'.format(np.mean(x0), np.std(x0)))
print('orthogonal mean: {0}, std: {1}'.format(np.mean(x1), np.std(x1)))
