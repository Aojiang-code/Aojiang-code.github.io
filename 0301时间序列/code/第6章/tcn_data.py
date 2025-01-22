import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 数据归一化处理, 构造训练集、验证集、测试集
def build_data(df, x_cols, y_cols, train_ratio, valid_ratio, time_step):
    indexs = df.index.tolist()
    length = len(df)

    # 数据归一化
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_x_data = scaler_x.fit_transform(df[x_cols])
    scaled_y_data = scaler_y.fit_transform(df[y_cols])

    # 构建训练集
    scaled_x_train_data = scaled_x_data[0:int(length * train_ratio)]
    scaled_y_train_data = scaled_y_data[0:int(length * train_ratio)]
    train_indexs = indexs[0:int(length * train_ratio)]

    x_train, y_train, train_index = [], [], []
    for i in range(len(scaled_x_train_data) - time_step - 1):
        x_value = np.append(scaled_x_train_data[i: i + time_step], scaled_y_train_data[i: i + time_step], axis=1)
        x_train.append(x_value)

        y_value = scaled_y_train_data[i + 1:i + time_step + 1]
        y_train.append(y_value)

        index = train_indexs[i + time_step]
        train_index.append(index)

    # 构建验证集
    scaled_x_valid_data = scaled_x_data[int(length * train_ratio):int(length * valid_ratio)]
    scaled_y_valid_data = scaled_y_data[int(length * train_ratio):int(length * valid_ratio)]
    valid_indexs = indexs[int(length * train_ratio):int(length * valid_ratio)]

    x_valid, y_valid, valid_index = [], [], []
    for i in range(len(scaled_x_valid_data) - time_step - 1):
        x_value = np.append(scaled_x_valid_data[i: i + time_step], scaled_y_valid_data[i: i + time_step], axis=1)
        x_valid.append(x_value)

        y_value = scaled_y_valid_data[i + 1:i + time_step + 1]
        y_valid.append(y_value)

        index = valid_indexs[i + time_step]
        valid_index.append(index)

    # 构建测试集
    scaled_x_test_data = scaled_x_data[int(length * valid_ratio):]
    scaled_y_test_data = scaled_y_data[int(length * valid_ratio):]
    test_indexs = indexs[int(length * valid_ratio):]

    x_test, y_test, test_index = [], [], []
    for i in range(len(scaled_x_test_data) - time_step - 1):
        x_value = np.append(scaled_x_test_data[i: i + time_step], scaled_y_test_data[i: i + time_step], axis=1)
        x_test.append(x_value)

        y_value = scaled_y_test_data[i + 1:i + time_step + 1]
        y_test.append(y_value)

        index = test_indexs[i + time_step]
        test_index.append(index)

    return (scaler_x, scaler_y), (x_train, y_train, train_index), \
        (x_valid, y_valid, valid_index), (x_test, y_test, test_index)
