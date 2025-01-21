try:
    # 尝试导入tensorflow的keras模块
    import tensorflow.python.keras as keras
except:
    # 如果失败，导入tensorflow的另一个keras模块
    import tensorflow.keras as keras

# 定义数据加载函数
def load_data():
    # 加载Fashion MNIST数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # 训练集图像形状调整为 (60000, 28, 28, 1)，并转换为float32类型
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')

    # 归一化处理，将像素值缩放到0到1之间
    x_train = x_train / 255

    # 测试集图像形状调整为 (10000, 28, 28, 1)，并转换为float32类型
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

    # 归一化处理
    x_test = x_test / 255

    # 将标签转化为one-hot编码
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

# 定义卷积神经网络模型类
class Model(keras.Model):
    def __init__(self):
        # 初始化父类
        super(Model, self).__init__(name='Model')

        # 第一层卷积层
        self.conv2d1 = keras.layers.Conv2D(name='conv2d1', filters=64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu', input_shape=(28, 28, 1))
        # 第一层池化层
        self.max_pooling1 = keras.layers.MaxPool2D(name='max_pooling1', pool_size=(2, 2), strides=2, padding='valid')

        # 第二层卷积层
        self.conv2d2 = keras.layers.Conv2D(name='conv2d2', filters=32, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')
        # 第二层池化层
        self.max_pooling2 = keras.layers.MaxPool2D(name='max_pooling2', pool_size=(2, 2), strides=2, padding='valid')

        # 第三层特征图像数据展平
        self.flatten = keras.layers.Flatten(name='flatten')

        # 第四层全连接层
        self.dense1 = keras.layers.Dense(name='dense1', units=128, activation='relu')

        # 第五层SoftMax输出分类结果
        self.dense2 = keras.layers.Dense(name='dense2', units=10, activation='softmax')

    # 定义前向传播函数
    def call(self, inupts):
        # 通过第一层卷积和池化
        layer1 = self.conv2d1(inupts)
        layer1 = self.max_pooling1(layer1)

        # 通过第二层卷积和池化
        layer2 = self.conv2d2(layer1)
        layer2 = self.max_pooling2(layer2)

        # 展平特征图像
        layer3 = self.flatten(layer2)

        # 通过全连接层
        layer4 = self.dense1(layer3)

        # 输出分类结果
        output = self.dense2(layer4)

        return output

# 加载数据
x_train, y_train, x_test, y_test = load_data()

# 创建模型实例
model = Model()

# 打印网络结构信息
model.build((None, 28, 28, 1))
model.summary()

# 定义回调函数
callbacks = [
    # 提前停止训练以防止过拟合
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20),
    # 使用TensorBoard记录训练日志
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
]

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.2, callbacks=callbacks)

# 评估模型在测试集上的表现
test_scores = model.evaluate(x_test, y_test, verbose=0)

# 打印测试集上的损失和准确率
print('test loss:', test_scores[0])
print('test acc:', test_scores[1])





# 在这段代码中，定义了一个卷积神经网络模型，用于对Fashion MNIST数据集进行分类。代码包括数据加载、模型定义、训练和评估的完整流程。通过使用Keras的高级API，代码实现了一个简单的卷积神经网络，并使用回调函数来监控训练过程。