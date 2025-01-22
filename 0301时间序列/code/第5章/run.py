import time
import numpy as np
import pandas as pd
import tensorflow as tf
from rnn_data import build_data
from rnn_model import LSTM, GRU
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# 模型预测效果
def model_pred(scaler_y, y_cols, x_data, y_data, index, model, pred_step):
    y_value = None
    for elem in y_data:
        inv_value = scaler_y.inverse_transform(np.array([elem]))

        if y_value is None:
            y_value = inv_value
        else:
            y_value = np.append(y_value, inv_value, axis=0)

    y_pred = None
    for i in range(0, len(x_data), pred_step):
        step_data = x_data[i:i + pred_step]
        step_pred = None
        for j in range(len(step_data)):

            if step_pred is None:
                elem = np.array([step_data[j]])
                pred_value = model.predict(elem)
                step_pred = pred_value
                inv_value = scaler_y.inverse_transform(pred_value)
            else:
                elem = np.array([step_data[j]])
                elem[:, - j:, -len(y_cols):] = step_pred
                pred_value = model.predict(elem)
                step_pred = np.append(step_pred, pred_value, axis=0)
                inv_value = scaler_y.inverse_transform(pred_value)

            if y_pred is None:
                y_pred = inv_value
            else:
                y_pred = np.append(y_pred, inv_value, axis=0)

    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.05)
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


# 特征列、标签列
x_cols = ['financing_balance', 'financing_balance_ratio', 'financing_buy', 'financing_net_buy', 'O/N', '1W',
          'hs300_highest_price', 'hs300_lowest_price', 'ust_closing_price', 'usdx_closing_price', 'ust_extent',
          'usdx_extent', 'CPI_YoY', 'PPI_YoY', 'PMI_MI_YoY', 'PMI_NMI_YoY', 'M1_YoY', 'M2_YoY', 'credit_mon_val',
          'credit_mon_YoY', 'credit_acc_val', 'credit_acc_YoY']

y_cols = ['hs300_closing_price', 'hs300_yield_rate']

# 数据集划分
# train_ratio=0.75, valid_ratio=0.9
# 前75%作为训练集, 75%～90%作为验证集, 90%～100%作为测试集
# time_step 表示模型输入序列的长度
# batch_size 表示批数据序列的数量
train_ratio = 0.75
valid_ratio = 0.9
time_step = 7
batch_size = 32

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('../data/informations.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

df = df[x_cols + y_cols]
df.dropna(axis=0, how='any', inplace=True)

(scaler_x, scaler_y), (x_train, y_train, train_index), \
    (x_valid, y_valid, valid_index), (x_test, y_test, test_index) = \
    build_data(df, x_cols, y_cols, train_ratio, valid_ratio, time_step)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.batch(batch_size, drop_remainder=True)
valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_data = valid_data.batch(batch_size, drop_remainder=True)

# 模型参数
hidden_units = 32
dropout_ratio = 0.5
input_size = 32
output_size = len(y_cols)

# 默认采用LSTM, 也可以换成GRU
model = LSTM(hidden_units, dropout_ratio, input_size, output_size)
# model = GRU(hidden_units, dropout_ratio, input_size, output_size)

# 训练参数
min_delta = 1e-5
patience = 100
lr = 1e-4
clipnorm = 1
epochs = 10000

start_time = time.time()

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, mode='min'),
             tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)]

model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr, clipnorm=clipnorm), loss='mse', metrics=['mse'],
              experimental_run_tf_function=False)

model.fit(train_data, epochs=epochs, validation_data=valid_data, callbacks=callbacks)

end_time = time.time()

print('train model time: ', end_time - start_time)

# 训练集拟合效果（向前预测1步）
print('train data pred 1 step:')
model_pred(scaler_y, y_cols, x_train, y_train, train_index, model, pred_step=1)

# 验证集拟合效果（向前预测1步）
print('valid data pred 1 step:')
model_pred(scaler_y, y_cols, x_valid, y_valid, valid_index, model, pred_step=1)

# 测试集拟合效果（向前预测1步）
print('test data pred 1 step:')
model_pred(scaler_y, y_cols, x_test, y_test, test_index, model, pred_step=1)

# 训练集拟合效果 （向前预测3步）
print('train data pred 3 step:')
model_pred(scaler_y, y_cols, x_train, y_train, train_index, model, pred_step=3)

# 验证集拟合效果  （向前预测3步）
print('valid data pred 3 step:')
model_pred(scaler_y, y_cols, x_valid, y_valid, valid_index, model, pred_step=3)

# 测试集拟合效果 （向前预测3步）
print('test data pred 3 step:')
model_pred(scaler_y, y_cols, x_test, y_test, test_index, model, pred_step=3)
