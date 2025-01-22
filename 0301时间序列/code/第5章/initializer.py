import numpy as np

# 定义tanh激活函数
def tanh(x):
    # 返回tanh函数的计算结果
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# 定义ReLU激活函数
def relu(x):
    # 返回ReLU函数的计算结果
    return np.maximum(0, x)

# glorot(xavier)初始化
# 初始化输入向量x0
x0 = np.random.normal(0, 1, [256, 1])
x1 = x0

# 进行100次迭代
for i in range(100):
    # uniform 矩阵相乘
    w0 = np.random.uniform(-1, 1, [256, 256])
    x0 = np.matmul(w0, x0)
    x0 = tanh(x0)

    # glorot 矩阵相乘
    w1 = w0 * np.sqrt(6 / (256 + 256))
    x1 = np.matmul(w1, x1)
    x1 = tanh(x1)

# 打印glorot初始化的结果
print('glorot:')
print('uniform mean: {0}, std: {1}'.format(np.mean(x0), np.std(x0)))
print('glorot mean: {0}, std: {1}'.format(np.mean(x1), np.std(x1)))

# he(kaiming)初始化
# 重新初始化输入向量x0
x0 = np.random.normal(0, 1, [256, 1])
x1 = x0

# 进行100次迭代
for i in range(100):
    # uniform 矩阵相乘
    w0 = np.random.uniform(-1, 1, [256, 256])
    x0 = np.matmul(w0, x0)
    x0 = relu(x0)

    # he 矩阵相乘
    w1 = w0 * np.sqrt(6 / (256))
    x1 = np.matmul(w1, x1)
    x1 = relu(x1)

# 打印he初始化的结果
print('he:')
print('uniform mean: {0}, std: {1}'.format(np.mean(x0), np.std(x0)))
print('he mean: {0}, std: {1}'.format(np.mean(x1), np.std(x1)))

# orthogonal初始化
# 重新初始化输入向量x0
x0 = np.random.normal(0, 1, [256, 1])
x1 = x0

# 进行100次迭代
for i in range(100):
    # normal 矩阵相乘
    w0 = np.random.normal(0, 1, [256, 256])
    x0 = np.matmul(w0, x0)

    # orthogonal 矩阵相乘
    u, _, v = np.linalg.svd(w0, full_matrices=False)
    w1 = u
    x1 = np.matmul(w1, x1)

# 打印orthogonal初始化的结果
print('orthogonal:')
print('normal mean: {0}, std: {1}'.format(np.mean(x0), np.std(x0)))
print('orthogonal mean: {0}, std: {1}'.format(np.mean(x1), np.std(x1)))
