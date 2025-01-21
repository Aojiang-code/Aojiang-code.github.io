try:
    # 尝试导入tensorflow的keras模块
    import tensorflow.python.keras as keras
except:
    # 如果失败，导入tensorflow的另一个keras模块
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
row, col = 10, 16  # 定义显示图片的行数和列数
sample_num = row * col  # 计算需要采样的图片总数
sample_list = [i for i in range(x_train.shape[0])]  # 创建一个包含所有训练集索引的列表
sample_list = random.sample(sample_list, sample_num)  # 随机采样指定数量的索引
samples = x_train[sample_list, :]  # 根据采样的索引获取对应的图片

# 创建一个大图像用于显示所有采样的图片
BigIm = np.zeros((28 * row, 28 * col))
for i in range(row):
    for j in range(col):
        # 将每张采样的图片放置在大图像的对应位置
        BigIm[28 * i:28 * (i + 1), 28 * j:28 * (j + 1)] = samples[i * col + j]

# 使用OpenCV显示大图像
cv2.imshow("mnist samples", BigIm)
cv2.waitKey(0)  # 等待按键以关闭窗口



# 在这段代码中，首先加载MNIST数据集并打印其维度信息。然后，随机采样部分训练集的手写数字图片，并将这些图片拼接成一个大图像进行显示。通过使用OpenCV库，代码实现了对手写数字图片的可视化展示。