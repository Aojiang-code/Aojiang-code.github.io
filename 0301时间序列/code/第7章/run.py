import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from transformer_data import build_data
from transformer_model import Transformer


# encoder_num_layers Encoder的层数
# decoder_num_layers Decoder的层数
# atten_dim Attention的q、k、v向量的维度
# head_num Attention的多头数量
# atten_output_dim Attention层输出的维度
# dff FeedForward 第一层的维度
# output_len 输出序列的长度
# output_dim 输出序列每个元素的维度
def train_model(encoder_num_layers, decoder_num_layers, atten_dim,
                head_num, atten_output_dim, dff, dropout_rate,
                output_len, output_dim, lr, batch_size,
                x_encoder_train, x_decoder_train, y_decoder_train,
                x_encoder_valid, x_decoder_valid, y_decoder_valid):
    model = Transformer(encoder_num_layers, decoder_num_layers, atten_dim, head_num, atten_output_dim, dff,
                        dropout_rate, output_len, output_dim)

    # 定义模型优化器
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    # 定义损失函数
    loss_fn = tf.keras.losses.MeanSquaredError()

    # 定义模型训练的统计指标
    train_metric = tf.keras.metrics.MeanSquaredError()
    valid_metric = tf.keras.metrics.MeanSquaredError()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_encoder_train, x_decoder_train, y_decoder_train))
    train_dataset = train_dataset.batch(batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_encoder_valid, x_decoder_valid, y_decoder_valid))
    valid_dataset = valid_dataset.batch(batch_size)

    min_valid_mse, min_epoch = 100, None
    epochs = 1000
    for epoch in range(epochs):
        # 模型训练
        for step, (x_batch_encoder_train, x_batch_decoder_train, y_batch_decoder_train) in enumerate(train_dataset):
            with tf.GradientTape() as g:
                logits = model(x_batch_encoder_train, x_batch_decoder_train, True)
                loss_val = loss_fn(y_batch_decoder_train, logits)

            grads = g.gradient(loss_val, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 模型训练集效果统计
        for step, (x_batch_encoder_train, x_batch_decoder_train, y_batch_decoder_train) in enumerate(train_dataset):
            train_logits = model(x_batch_encoder_train, x_batch_decoder_train, False)
            train_metric(y_batch_decoder_train, train_logits)

        train_mse = float(train_metric.result())
        train_metric.reset_states()

        # 模型验证集效果统计
        for step, (x_batch_encoder_valid, x_batch_decoder_valid, y_batch_decoder_valid) in enumerate(valid_dataset):
            val_logits = model(x_batch_encoder_valid, x_batch_decoder_valid, False)
            valid_metric(y_batch_decoder_valid, val_logits)

        valid_mse = float(valid_metric.result())
        valid_metric.reset_states()

        print("epoch: {0},  train_mse: {1},  valid_mse: {2}".format(epoch, train_mse, valid_mse))

        if min_valid_mse >= valid_mse:
            min_valid_mse = valid_mse
            min_epoch = epoch

        os.makedirs('/root/data/model/transformer{0}'.format(epoch), exist_ok=True)
        model.save_weights('/root/data/model/transformer{0}'.format(epoch) + '/')

    print("min_valid_mse: {0},  min_epoch: {1}".format(min_valid_mse, min_epoch))


def model_pred(scaler_y, y_cols, encoder_x_data, decoder_x_data, decoder_y_data, index, model, decoder_input_time_step,
               encoder_decoder_intersection, pred_step):
    y_value = None
    for elem in decoder_y_data:

        inv_value = scaler_y.inverse_transform(np.array([elem[-1, :]]))
        if y_value is None:
            y_value = inv_value
        else:
            y_value = np.append(y_value, inv_value, axis=0)

    y_pred = None
    gap = decoder_input_time_step - encoder_decoder_intersection
    for i in range(0, len(encoder_x_data), pred_step):
        encoder_x_seq = encoder_x_data[i:i + pred_step]
        decoder_x_seq = decoder_x_data[i:i + pred_step]

        preds = None
        for j in range(len(encoder_x_seq)):
            encoder_elem = np.array([encoder_x_seq[j]])
            decoder_elem = np.array([decoder_x_seq[j]])

            if preds is None:
                pred_value = model(encoder_elem, decoder_elem, False)[:, -1, :]
                preds = pred_value
                inv_value = scaler_y.inverse_transform(pred_value)
            else:
                # Encoder序列加入预测结果
                if j > gap:
                    encoder_elem[:, - (j - gap):, -len(y_cols):] = preds[:(j - gap)]

                # Decoder序列加入预测结果
                decoder_elem[:, - j:, -len(y_cols):] = preds

                pred_value = model(encoder_elem, decoder_elem, False)[:, -1, :]
                preds = np.append(preds, pred_value, axis=0)
                inv_value = scaler_y.inverse_transform(pred_value)

            if y_pred is None:
                y_pred = inv_value
            else:
                y_pred = np.append(y_pred, inv_value, axis=0)

    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.2, wspace=0.05)
    for i in range(len(y_cols)):
        y_col = y_cols[i]
        y_col_value, y_col_pred = [], []

        for elem in y_value:
            y_col_value.append(elem[i])

        for elem in y_pred:
            y_col_pred.append(elem[i])

        mae = mean_absolute_error(y_col_value, y_col_pred)
        print(y_col + ' mae:', mae)

        plt.subplot(len(y_cols), 1, i + 1)
        plt.plot(index, y_col_value)
        plt.plot(index, y_col_pred)
        plt.legend([y_col + '(value)', y_col + '(pred)'])

    plt.show()


# 特征列与目标列
x_cols = ['financing_balance', 'financing_balance_ratio', 'financing_buy', 'financing_net_buy', 'O/N', '1W',
          'hs300_highest_price', 'hs300_lowest_price', 'ust_closing_price', 'usdx_closing_price', 'ust_extent',
          'usdx_extent', 'CPI_YoY', 'PPI_YoY', 'PMI_MI_YoY', 'PMI_NMI_YoY', 'M1_YoY', 'M2_YoY', 'credit_mon_val',
          'credit_mon_YoY', 'credit_acc_val', 'credit_acc_YoY']

y_cols = ['hs300_closing_price', 'hs300_yield_rate']

# 数据准备参数
train_ratio = 0.75
valid_ratio = 0.9
encoder_input_time_step = 5
decoder_input_time_step = 2
decoder_output_time_step = 2
encoder_decoder_intersection = 1
decoder_intersection = 1
batch_size = 32

# 模型参数
encoder_num_layers = 2
decoder_num_layers = 2
atten_dim = 16
head_num = 4
atten_output_dim = 16
dff = 4 * atten_dim
dropout_rate = 0.5
lr = 0.001
output_len = decoder_output_time_step
output_dim = len(y_cols)

# 加载数据
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('./data/informations.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

df = df[x_cols + y_cols]
df.dropna(axis=0, how='any', inplace=True)

# 准备数据集
(scaler_x, scaler_y), \
    (x_encoder_train, x_decoder_train, y_decoder_train, train_index), \
    (x_encoder_valid, x_decoder_valid, y_decoder_valid, valid_index), \
    (x_encoder_test, x_decoder_test, y_decoder_test, test_index) \
    = build_data(df, x_cols, y_cols, train_ratio, valid_ratio,
                 encoder_input_time_step, decoder_input_time_step,
                 decoder_output_time_step, encoder_decoder_intersection,
                 decoder_intersection)

# 模型训练
train_model(encoder_num_layers, decoder_num_layers, atten_dim,
            head_num, atten_output_dim, dff, dropout_rate,
            output_len, output_dim, lr, batch_size,
            x_encoder_train, x_decoder_train, y_decoder_train,
            x_encoder_valid, x_decoder_valid, y_decoder_valid)

model = Transformer(encoder_num_layers, decoder_num_layers, atten_dim, head_num, atten_output_dim, dff, dropout_rate,
                    output_len, output_dim)
model.load_weights('/root/data/model/transformer635/')

print('train data pred 1 step:')
model_pred(scaler_y, y_cols, x_encoder_train, x_decoder_train, y_decoder_train, train_index, model,
           decoder_input_time_step, encoder_decoder_intersection, pred_step=1)

print('valid data pred 1 step:')
model_pred(scaler_y, y_cols, x_encoder_valid, x_decoder_valid, y_decoder_valid, valid_index, model,
           decoder_input_time_step, encoder_decoder_intersection, pred_step=1)

print('test data pred 1 step:')
model_pred(scaler_y, y_cols, x_encoder_test, x_decoder_test, y_decoder_test, test_index, model, decoder_input_time_step,
           encoder_decoder_intersection, pred_step=1)

print('train data pred 3 step:')
model_pred(scaler_y, y_cols, x_encoder_train, x_decoder_train, y_decoder_train, train_index, model,
           decoder_input_time_step, encoder_decoder_intersection, pred_step=3)

print('valid data pred 3 step:')
model_pred(scaler_y, y_cols, x_encoder_valid, x_decoder_valid, y_decoder_valid, valid_index, model,
           decoder_input_time_step, encoder_decoder_intersection, pred_step=3)

print('test data pred 3 step:')
model_pred(scaler_y, y_cols, x_encoder_test, x_decoder_test, y_decoder_test, test_index, model, decoder_input_time_step,
           encoder_decoder_intersection, pred_step=3)
