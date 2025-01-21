try:
    import tensorflow.python.keras as keras

except:
    import tensorflow.keras as keras

import cv2
import random
import numpy as np

# 数据下载，区分训练集和测试集
# x_train, y_train 分别表示训练集的手写数字与对应标签编号
# x_test, y_test 分别表示测试集的手写数字与对应标签编号
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 打印数据的维度
print('x_train:', x_train.shape, '  y_train:', y_train.shape)
print('x_test:', x_test.shape, '  y_test:', y_test.shape)

# 采样部分训练集的手写数字图片显示
row, col = 10, 16
sample_num = row * col
sample_list = [i for i in range(x_train.shape[0])]
sample_list = random.sample(sample_list, sample_num)
samples = x_train[sample_list, :]

BigIm = np.zeros((28 * row, 28 * col))
for i in range(row):
    for j in range(col):
        BigIm[28 * i:28 * (i + 1), 28 * j:28 * (j + 1)] = samples[i * col + j]

cv2.imshow("mnist samples", BigIm)
cv2.waitKey(0)