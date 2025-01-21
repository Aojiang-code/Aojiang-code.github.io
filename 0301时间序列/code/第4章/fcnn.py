import tensorflow as tf

try:
    # 尝试导入tensorflow的keras模块
    import tensorflow.python.keras as keras
except:
    # 如果失败，导入tensorflow的另一个keras模块
    import tensorflow.keras as keras

# 定义数据加载函数
def load_data():
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 转换训练集图像形状为 (60000, 784) 并归一化
    x_train = x_train.reshape(60000, 784) / 255

    # 转换测试集图像形状为 (10000, 784) 并归一化
    x_test = x_test.reshape(10000, 784) / 255

    # 将标签转化为one-hot编码
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

# 准备网络结构和优化器
model = keras.Sequential([
    # 第一层全连接层，512个神经元，激活函数为ReLU
    keras.layers.Dense(units=512, activation='relu'),
    # 第二层全连接层，128个神经元，激活函数为ReLU
    keras.layers.Dense(units=128, activation='relu'),
    # 第三层全连接层，32个神经元，激活函数为ReLU
    keras.layers.Dense(units=32, activation='relu'),
    # 输出层，10个神经元，激活函数为SoftMax
    keras.layers.Dense(units=10, activation='softmax')
])

# 定义优化器
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# 加载数据
x_train, y_train, x_test, y_test = load_data()

# 创建训练数据集，批次大小为64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)

# 定义模型训练函数
def train():
    # 完成执行一次全量数据训练，称为一轮epoch
    for epoch in range(10):
        for step, (x, y) in enumerate(train_dataset):
            # 采用小批量梯度下降，将全量数据划分为不同的数据块，一块数据称为一个批次，这里设置一个数据块的大小为64个样本

            with tf.GradientTape() as g:
                # 步骤1 前向传播输出
                y_pred = model(x)

                # 步骤2 计算损失函数
                loss = keras.losses.categorical_crossentropy(y, y_pred)
                loss = tf.reduce_mean(loss)

            # 步骤3：通过损失函数计算梯度
            grad = g.gradient(loss, model.trainable_variables)

            # 步骤4：通过梯度更新网络参数
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            if step % 100 == 0:
                # 评价准确率
                acc = keras.metrics.categorical_accuracy(y, y_pred)
                acc = tf.reduce_mean(acc)

                # 打印当前epoch、step、损失和准确率
                print('epoch:%s, step:%s, loss:%s, acc:%s' % (epoch, step, loss.numpy(), acc.numpy()))

# 定义模型测试函数
def test():
    # 进行前向传播，计算预测值
    y_pred = model(x_test)
    # 计算测试集的准确率
    acc = keras.metrics.categorical_accuracy(y_test, y_pred)
    acc = tf.reduce_mean(acc)

    # 打印测试集的准确率
    print('-------------------------------------------------')
    print('test acc:%s' % (acc.numpy()))

# 训练模型
train()
# 测试模型
test()



# 在这段代码中，定义了一个全连接神经网络模型，用于对MNIST数据集进行分类。代码包括数据加载、模型定义、训练和测试的完整流程。通过使用Keras的高级API，代码实现了一个简单的全连接神经网络，并使用自定义的训练和测试函数来评估模型的性能。