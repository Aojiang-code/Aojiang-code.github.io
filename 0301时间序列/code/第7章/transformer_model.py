import math
import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    # atten_dim: attention q\k\v的向量维度
    # head_num: multi-head attention 多头的数量
    # atten_output_dim: 经过attention层每个元素输出的维度
    def __init__(self, atten_dim, head_num, atten_output_dim):
        super(MultiHeadAttention, self).__init__()

        self.atten_dim = atten_dim
        self.head_num = head_num
        self.multi_output_dim = atten_dim * head_num

        # 矩阵乘法, 将输入转换为q、k、v向量, 不使用偏置项与激活函数
        self.W_q = tf.keras.layers.Dense(units=self.multi_output_dim, activation=None, use_bias=False)
        self.W_k = tf.keras.layers.Dense(units=self.multi_output_dim, activation=None, use_bias=False)
        self.W_v = tf.keras.layers.Dense(units=self.multi_output_dim, activation=None, use_bias=False)

        # multi-head attention 多头合并输出
        self.W_combine = tf.keras.layers.Dense(units=atten_output_dim, activation=None, use_bias=False)

    def call(self, q, k, v):
        batch_size_q, seq_len_q = int(tf.shape(q)[0]), int(tf.shape(q)[1])
        batch_size_k, seq_len_k = int(tf.shape(k)[0]), int(tf.shape(k)[1])
        batch_size_v, seq_len_v = int(tf.shape(v)[0]), int(tf.shape(v)[1])

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        # 维度：batch_size \ head_num \ seq_len \ attention_dim
        q = tf.reshape(q, [batch_size_q, seq_len_q, self.head_num, self.atten_dim])
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        k = tf.reshape(k, [batch_size_k, seq_len_k, self.head_num, self.atten_dim])
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = tf.reshape(v, [batch_size_v, seq_len_v, self.head_num, self.atten_dim])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # 计算attention向量
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.math.sqrt(tf.cast(self.atten_dim, dtype='float32'))
        attention_vec = matmul_qk / dk

        # 经过SoftMax计算attention权重
        attention_weights = tf.nn.softmax(attention_vec, axis=-1)

        # 计算多头注意力
        multi_outputs = tf.matmul(attention_weights, v)

        # 维度：batch_size\ seq_len\ multi_output_dim
        multi_outputs = tf.transpose(multi_outputs, perm=[0, 2, 1, 3])
        multi_outputs = tf.reshape(multi_outputs, (batch_size_q, seq_len_q, self.multi_output_dim))

        # 合并多头注意力
        outputs = self.W_combine(multi_outputs)

        return outputs


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, base_val):
        super(PositionEncoding, self).__init__()

        self.base_val = float(base_val)

    def call(self, inputs):
        seq_len = int(tf.shape(inputs)[1])
        dimension = int(tf.shape(inputs)[2])

        pos_vec = np.arange(seq_len)
        i_vec = np.arange(dimension / 2)

        pos_embedding = []
        for pos in pos_vec:
            p = []
            for i in i_vec:
                # PE_2i(p) = sin(p/10000^(2i/d_pos))
                # PE_2i+1(p) = cos(p/10000^(2i/d_pos))
                p.append(math.sin(pos / self.base_val ** (2 * i / dimension)))
                p.append(math.cos(pos / self.base_val ** (2 * i / dimension)))

            # 经过上面运算, p向量的维度是偶数, 考虑dimension为奇数的情况, 保证位置编码向量与输入向量维度一致
            p = p[0:int(dimension)]
            pos_embedding.append(p)

        pos_embedding = tf.cast(pos_embedding, dtype='float32')

        outputs = inputs + pos_embedding

        return outputs


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dff, atten_output_dim):
        super(FeedForward, self).__init__()

        self.layer1 = tf.keras.layers.Dense(units=dff, activation='relu')
        self.layer2 = tf.keras.layers.Dense(units=atten_output_dim, activation=None)

    def call(self, inputs):
        val1 = self.layer1(inputs)
        outputs = self.layer2(val1)

        return outputs


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, atten_dim, head_num, atten_output_dim, dff, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(atten_dim, head_num, atten_output_dim)

        self.feed_forward = FeedForward(dff, atten_output_dim)

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        # multi-head attention
        attention_outputs = self.multi_head_attention(q=inputs, k=inputs, v=inputs)
        attention_outputs = self.dropout1(attention_outputs, training=training)
        output1 = self.layer_norm1(inputs + attention_outputs)

        # feed forward network
        feed_forward_outputs = self.feed_forward(output1)
        feed_forward_outputs = self.dropout2(feed_forward_outputs, training=training)
        output2 = self.layer_norm2(output1 + feed_forward_outputs)

        return output2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, encoder_num_layers, atten_dim, head_num, atten_output_dim, dff, dropout_rate, base_val=10.):
        super(Encoder, self).__init__()

        self.inputs_embedding = tf.keras.layers.Dense(units=atten_output_dim, activation='relu')

        self.position_encoding = PositionEncoding(base_val)

        self.encoder_layers = [EncoderLayer(atten_dim, head_num, atten_output_dim, dff, dropout_rate) for _ in
                               range(encoder_num_layers)]

    def call(self, inputs, training):
        inputs = self.inputs_embedding(inputs)
        inputs = self.position_encoding(inputs)

        outputs = inputs
        for encoder_layer in self.encoder_layers:
            outputs = encoder_layer(outputs, training)

        return outputs


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, atten_dim, head_num, atten_output_dim, dff, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(atten_dim, head_num, atten_output_dim)
        self.multi_head_attention2 = MultiHeadAttention(atten_dim, head_num, atten_output_dim)

        self.feed_forward = FeedForward(dff, atten_output_dim)

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.layer_norm3 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, training):
        # multi-head attention
        attention_outputs1 = self.multi_head_attention1(q=inputs, k=inputs, v=inputs)
        attention_outputs1 = self.dropout1(attention_outputs1, training=training)
        output1 = self.layer_norm1(inputs + attention_outputs1)

        # multi-head cross-attention
        attention_outputs2 = self.multi_head_attention2(q=output1, k=encoder_outputs, v=encoder_outputs)
        attention_outputs2 = self.dropout2(attention_outputs2, training=training)
        output2 = self.layer_norm2(output1 + attention_outputs2)

        # feed forward network
        feed_forward_outputs = self.feed_forward(output2)
        feed_forward_outputs = self.dropout3(feed_forward_outputs, training=training)
        output3 = self.layer_norm3(output2 + feed_forward_outputs)

        return output3


class Decoder(tf.keras.layers.Layer):
    def __init__(self, decoder_num_layers, atten_dim, head_num, atten_output_dim, dff, dropout_rate, base_val=10.):
        super(Decoder, self).__init__()

        self.inputs_embedding = tf.keras.layers.Dense(units=atten_output_dim, activation='relu')

        self.position_encoding = PositionEncoding(base_val)

        self.decoder_layers = [DecoderLayer(atten_dim, head_num, atten_output_dim, dff, dropout_rate) for _ in
                               range(decoder_num_layers)]

    def call(self, decoder_inputs, encoder_outputs, training):
        decoder_inputs = self.inputs_embedding(decoder_inputs)
        decoder_inputs = self.position_encoding(decoder_inputs)

        outputs = decoder_inputs
        for decoder_layer in self.decoder_layers:
            outputs = decoder_layer(outputs, encoder_outputs, training)

        return outputs


class Transformer(tf.keras.Model):
    def __init__(self, encoder_num_layers, decoder_num_layers, atten_dim, head_num, atten_output_dim, dff, dropout_rate,
                 output_len, output_dim):
        super(Transformer, self).__init__()

        self.output_len = output_len
        self.output_dim = output_dim

        self.encoder = Encoder(encoder_num_layers, atten_dim, head_num, atten_output_dim, dff, dropout_rate)

        self.decoder = Decoder(decoder_num_layers, atten_dim, head_num, atten_output_dim, dff, dropout_rate)

        self.output_heads = [tf.keras.layers.Dense(output_dim) for _ in range(output_len)]

    def call(self, encoder_inputs, decoder_inputs, training=None):
        batch_size = int(tf.shape(decoder_inputs)[0])

        encoder_outputs = self.encoder(encoder_inputs, training)

        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, training)
        decoder_outputs = tf.reshape(decoder_outputs, [batch_size, -1])

        # Decoder输出向量经过全连接神经网络, 每一步输出都是一个全连接神经网络计算结果
        outputs = None
        for output_head in self.output_heads:
            if outputs is None:
                outputs = output_head(decoder_outputs)
            else:
                outputs = tf.concat([outputs, output_head(decoder_outputs)], axis=-1)

        outputs = tf.reshape(outputs, shape=[batch_size, self.output_len, self.output_dim])

        return outputs