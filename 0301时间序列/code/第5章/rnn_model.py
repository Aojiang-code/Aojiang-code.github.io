import tensorflow as tf


# 定义LSTM模型类
class LSTM(tf.keras.Model):
    def __init__(self, hidden_units, dropout_ratio, input_size, output_size):
        # 初始化父类
        super(LSTM, self).__init__(name='LSTM')

        # 设置隐藏单元数
        self.hidden_units = hidden_units

        # 定义输入层
        self.input_layer = tf.keras.layers.Dense(name='input', units=input_size, activation='tanh')

        # 定义第一个LSTM层
        self.lstm_layer_0 = tf.keras.layers.LSTMCell(name='lstm0', units=hidden_units,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal', bias_initializer='ones',
                                                     dropout=dropout_ratio)
        # 定义第二个LSTM层
        self.lstm_layer_1 = tf.keras.layers.LSTMCell(name='lstm1', units=hidden_units,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal', bias_initializer='ones',
                                                     dropout=dropout_ratio)

        # 定义输出层
        self.output_layer = tf.keras.layers.Dense(name='output', units=output_size, activation='tanh')

    # 定义前向传播函数
    def call(self, inputs, training=None):
        # 获取批次大小
        batch_size = len(inputs)

        # 初始化LSTM层的状态
        state_0 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]
        state_1 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]

        # 遍历输入序列
        for input in tf.unstack(inputs, axis=1):
            # 通过输入层
            input = self.input_layer(input)

            # 通过第一个LSTM层
            out_0, state_0 = self.lstm_layer_0(input, state_0, training)
            # 通过第二个LSTM层
            out_1, state_1 = self.lstm_layer_1(out_0, state_1, training)

        # 通过输出层
        pred = self.output_layer(out_1)

        return pred


# 定义GRU模型类
class GRU(tf.keras.Model):
    def __init__(self, hidden_units, dropout_ratio, input_size, output_size):
        # 初始化父类
        super(GRU, self).__init__(name='GRU')

        # 设置隐藏单元数
        self.hidden_units = hidden_units

        # 定义输入层
        self.input_layer = tf.keras.layers.Dense(name='input', units=input_size, activation='tanh')

        # 定义第一个GRU层
        self.gru_layer_0 = tf.keras.layers.GRUCell(name='gru0', units=hidden_units, kernel_initializer='glorot_uniform',
                                                   recurrent_initializer='orthogonal', bias_initializer='ones',
                                                   dropout=dropout_ratio)
        # 定义第二个GRU层
        self.gru_layer_1 = tf.keras.layers.GRUCell(name='gru1', units=hidden_units, kernel_initializer='glorot_uniform',
                                                   recurrent_initializer='orthogonal', bias_initializer='ones',
                                                   dropout=dropout_ratio)

        # 定义输出层
        self.output_layer = tf.keras.layers.Dense(name='output', units=output_size, activation='tanh')

    # 定义前向传播函数
    def call(self, inputs, training=None):
        # 获取批次大小
        batch_size = len(inputs)

        # 初始化GRU层的状态
        state_0 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]
        state_1 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]

        # 遍历输入序列
        for input in tf.unstack(inputs, axis=1):
            # 通过输入层
            input = self.input_layer(input)

            # 通过第一个GRU层
            out_0, state_0 = self.gru_layer_0(input, state_0, training)
            # 通过第二个GRU层
            out_1, state_1 = self.gru_layer_1(out_0, state_1, training)

        # 通过输出层
        pred = self.output_layer(out_1)

        return pred
