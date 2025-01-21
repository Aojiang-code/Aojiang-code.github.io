import tensorflow as tf

try:
    import tensorflow.python.keras as keras

except:
    import tensorflow.keras as keras


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 转换图像形状 (60000, 28, 28) 为 (60000, 784)
    x_train = x_train.reshape(60000, 784) / 255

    # 转换图像形状 (10000, 28, 28) 为 (10000, 784)
    x_test = x_test.reshape(10000, 784) / 255

    # 将标签转化为 one-hot 编码
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# 准备网络结构和优化器
model = keras.Sequential([keras.layers.Dense(units=512, activation='relu'),
                          keras.layers.Dense(units=128, activation='relu'),
                          keras.layers.Dense(units=32, activation='relu'),
                          keras.layers.Dense(units=10, activation='softmax')
                          ])

optimizer = keras.optimizers.Adam(learning_rate=0.001)

x_train, y_train, x_test, y_test = load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)


# 模型训练
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

                print('epoch:%s, step:%s, loss:%s, acc:%s' % (epoch, step, loss.numpy(), acc.numpy()))


# 模型测试
def test():
    y_pred = model(x_test)
    acc = keras.metrics.categorical_accuracy(y_test, y_pred)
    acc = tf.reduce_mean(acc)

    print('-------------------------------------------------')
    print('test acc:%s' % (acc.numpy()))


train()
test()