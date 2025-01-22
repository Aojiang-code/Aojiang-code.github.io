import tensorflow as tf


# LSTM 模型
class LSTM(tf.keras.Model):
    def __init__(self, hidden_units, dropout_ratio, input_size, output_size):
        super(LSTM, self).__init__(name='LSTM')

        self.hidden_units = hidden_units

        self.input_layer = tf.keras.layers.Dense(name='input', units=input_size, activation='tanh')

        self.lstm_layer_0 = tf.keras.layers.LSTMCell(name='lstm0', units=hidden_units,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal', bias_initializer='ones',
                                                     dropout=dropout_ratio)
        self.lstm_layer_1 = tf.keras.layers.LSTMCell(name='lstm1', units=hidden_units,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal', bias_initializer='ones',
                                                     dropout=dropout_ratio)

        self.output_layer = tf.keras.layers.Dense(name='output', units=output_size, activation='tanh')

    def call(self, inputs, training=None):
        batch_size = len(inputs)

        state_0 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]
        state_1 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]

        for input in tf.unstack(inputs, axis=1):
            input = self.input_layer(input)

            out_0, state_0 = self.lstm_layer_0(input, state_0, training)
            out_1, state_1 = self.lstm_layer_1(out_0, state_1, training)

        pred = self.output_layer(out_1)

        return pred


# GRU 模型
class GRU(tf.keras.Model):
    def __init__(self, hidden_units, dropout_ratio, input_size, output_size):
        super(GRU, self).__init__(name='GRU')

        self.hidden_units = hidden_units

        self.input_layer = tf.keras.layers.Dense(name='input', units=input_size, activation='tanh')

        self.gru_layer_0 = tf.keras.layers.GRUCell(name='gru0', units=hidden_units, kernel_initializer='glorot_uniform',
                                                   recurrent_initializer='orthogonal', bias_initializer='ones',
                                                   dropout=dropout_ratio)
        self.gru_layer_1 = tf.keras.layers.GRUCell(name='gru1', units=hidden_units, kernel_initializer='glorot_uniform',
                                                   recurrent_initializer='orthogonal', bias_initializer='ones',
                                                   dropout=dropout_ratio)

        self.output_layer = tf.keras.layers.Dense(name='output', units=output_size, activation='tanh')

    def call(self, inputs, training=None):
        batch_size = len(inputs)

        state_0 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]
        state_1 = [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]

        for input in tf.unstack(inputs, axis=1):
            input = self.input_layer(input)

            out_0, state_0 = self.gru_layer_0(input, state_0, training)
            out_1, state_1 = self.gru_layer_1(out_0, state_1, training)

        pred = self.output_layer(out_1)

        return pred
