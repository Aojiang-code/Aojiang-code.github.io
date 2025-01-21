try:
    import tensorflow.python.keras as keras

except:
    import tensorflow.keras as keras


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # 图像形状 (60000, 28, 28, 1), 图像数量:60000, 图像尺寸:28 * 28  通道数为 1
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')

    # 归一化
    x_train = x_train / 255

    # 图像形状 (10000, 28, 28, 1), 图像数量:10000, 图像尺寸:28 * 28  通道数为 1
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

    # 归一化
    x_test = x_test / 255

    # 将标签转化为 one-hot 编码
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__(name='Model')

        # 第一层卷积层
        # filters: 卷积核数量
        # kernel_size: 卷积核尺寸
        # strides: 卷积运算步长
        # padding: valid表示不对原始图像补"0", 直接对原始图像进行卷积运算
        self.conv2d1 = keras.layers.Conv2D(name='conv2d1', filters=64,  kernel_size=(3, 3), strides=1, padding='valid', activation='relu', input_shape=(28, 28, 1))
        # 第一层池化层
        # MaxPool2D 表示 Max_Pooling 池化层
        # pool_size: 池化窗口尺寸
        # strides: 池化运算步长
        # padding: valid表示不对特征图像补"0", 直接对特征图像进行池化处理
        self.max_pooling1 = keras.layers.MaxPool2D(name='max_pooling1', pool_size=(2, 2), strides=2, padding='valid')

        # 第二层卷积层
        self.conv2d2 = keras.layers.Conv2D(name='conv2d2', filters=32, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')
        # 第二层池化层
        self.max_pooling2 = keras.layers.MaxPool2D(name='max_pooling2', pool_size=(2, 2), strides=2, padding='valid')

        # 第三层特征图像数据展平
        self.flatten = keras.layers.Flatten(name='flatten')

        # 第四层全连接层
        self.dense1 = keras.layers.Dense(name='dense1', units=128, activation='relu')

        # 第五层 SoftMax 输出分类结果
        self.dense2 = keras.layers.Dense(name='dense2', units=10, activation='softmax')

    def call(self, inupts):
        layer1 = self.conv2d1(inupts)
        layer1 = self.max_pooling1(layer1)

        layer2 = self.conv2d2(layer1)
        layer2 = self.max_pooling2(layer2)

        layer3 = self.flatten(layer2)

        layer4 = self.dense1(layer3)

        output = self.dense2(layer4)

        return output


x_train, y_train, x_test, y_test = load_data()

model = Model()

# 打印网络结构信息
model.build((None, 28, 28, 1))
model.summary()

# callbacks 表示回调函数
# EarlyStopping 表示训练提前终止, 防止模型过拟合
# monitor: 监控值（默认验证集的损失值）
# min_delta: 监控值的最小变化
# patience: 连续多少个epoch, 监控值的绝对变化小于 min_delta, 将停止训练

# TensorBoard 是模型训练的可视化组件
# log_dir: 训练过程中日志记录地址
# histogram_freq: 模型中各层权重参数记录的频率
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20),
keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)]

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.2, callbacks=callbacks)
test_scores = model.evaluate(x_test, y_test, verbose=0)

print('test loss:', test_scores[0])
print('test acc:', test_scores[1])