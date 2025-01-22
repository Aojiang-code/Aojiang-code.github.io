import tensorflow as tf
import tensorflow_addons as tf_addon


# TCN 残差模块
class TemporalLayer(tf.keras.layers.Layer):
    def __init__(self, input_channel, output_channel, padding,
                 kernel_size, strides, dilation_rate, dropout_ratio):
        super(TemporalLayer, self).__init__()

        # input_channel 输入通道数（序列数量）
        # output_channel 输出通道数（序列数量）
        # padding 需要补0的序列长度（TCN每层计算都会损失序列长度，需要补齐）
        # kernel_size 卷积核大小
        # strides 步长（TCN中卷积步长默认1）
        # dilation_rate 空洞大小
        # dropout_ratio dropout比例
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.padding = padding
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.dropout_ratio = dropout_ratio

        self.conv_layer1 = tf_addon.layers.WeightNormalization(
            tf.keras.layers.Conv1D(filters=self.output_channel,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   dilation_rate=self.dilation_rate,
                                   kernel_initializer='he_uniform',
                                   bias_initializer='zeros')
        )

        self.conv_layer2 = tf_addon.layers.WeightNormalization(
            tf.keras.layers.Conv1D(filters=self.output_channel,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   dilation_rate=self.dilation_rate,
                                   kernel_initializer='he_uniform',
                                   bias_initializer='zeros')
        )

        self.short_cut = tf.keras.layers.Conv1D(filters=self.output_channel,
                                                kernel_size=1,
                                                strides=1,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros')


def call(self, inputs):
    inputs_padding = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])

    h1_outputs = self.conv_layer1(inputs_padding)
    h1_outputs = tf.keras.activations.relu(h1_outputs)
    h1_outputs = tf.keras.layers.Dropout(self.dropout_ratio)(h1_outputs)

    h1_padding = tf.pad(h1_outputs, [[0, 0], [self.padding, 0], [0, 0]])

    h2_outputs = self.conv_layer2(h1_padding)
    h2_outputs = tf.keras.activations.relu(h2_outputs)
    h2_outputs = tf.keras.layers.Dropout(self.dropout_ratio)(h2_outputs)

    # short_cut连接方式, 前面经过padding保证输入与输出time_step相同, 这里检查channel是否相同
    if self.input_channel != self.output_channel:
        res_x = self.short_cut(inputs)

    else:
        res_x = inputs

    return tf.keras.layers.add([res_x, h2_outputs])


# TCN 网络
class TemporalConvNet(tf.keras.Model):
    def __init__(self, channels, kernel_size, strides, dropout_ratio):
        super(TemporalConvNet, self).__init__(name='TemporalConvNet')
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_ratio = dropout_ratio
        self.temporal_layers = []

        num_layers = len(self.channels)
        for i in range(num_layers - 1):
            dilation_rate = 2 ** i
            tuple_padding = (self.kernel_size - 1) * dilation_rate,
            padding = tuple_padding[0]
            input_channel = self.channels[i]
            output_channel = self.channels[i + 1]
            temporal_layer = TemporalLayer(input_channel, output_channel,
                                           padding, self.kernel_size,
                                           self.strides, dilation_rate,
                                           self.dropout_ratio)

            self.temporal_layers.append(temporal_layer)

    def call(self, inputs):
        outputs = inputs
        for temporal_layer in self.temporal_layers:
            outputs = temporal_layer(outputs)

        return outputs
