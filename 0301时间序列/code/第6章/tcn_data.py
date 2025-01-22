import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 数据归一化处理, 构造训练集、验证集、测试集
def build_data(df, x_cols, y_cols, train_ratio, valid_ratio, time_step):
    # 获取数据的索引列表
    indexs = df.index.tolist()
    # 获取数据的长度
    length = len(df)

    # 数据归一化
    # 创建用于x数据的归一化器
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    # 创建用于y数据的归一化器
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # 对x和y数据进行归一化
    scaled_x_data = scaler_x.fit_transform(df[x_cols])
    scaled_y_data = scaler_y.fit_transform(df[y_cols])

    # 构建训练集
    # 获取训练集的x和y数据
    scaled_x_train_data = scaled_x_data[0:int(length * train_ratio)]
    scaled_y_train_data = scaled_y_data[0:int(length * train_ratio)]
    # 获取训练集的索引
    train_indexs = indexs[0:int(length * train_ratio)]

    # 初始化训练集的x、y和索引列表
    x_train, y_train, train_index = [], [], []
    # 遍历训练数据，构建时间步长的数据
    for i in range(len(scaled_x_train_data) - time_step - 1):
        # 将x和y数据按时间步长拼接
        x_value = np.append(scaled_x_train_data[i: i + time_step], scaled_y_train_data[i: i + time_step], axis=1)
        x_train.append(x_value)

        # 获取当前时间步长的y值
        y_value = scaled_y_train_data[i + 1:i + time_step + 1]
        y_train.append(y_value)

        # 获取当前时间步长的索引
        index = train_indexs[i + time_step]
        train_index.append(index)

    # 构建验证集
    # 获取验证集的x和y数据
    scaled_x_valid_data = scaled_x_data[int(length * train_ratio):int(length * valid_ratio)]
    scaled_y_valid_data = scaled_y_data[int(length * train_ratio):int(length * valid_ratio)]
    # 获取验证集的索引
    valid_indexs = indexs[int(length * train_ratio):int(length * valid_ratio)]

    # 初始化验证集的x、y和索引列表
    x_valid, y_valid, valid_index = [], [], []
    # 遍历验证数据，构建时间步长的数据
    for i in range(len(scaled_x_valid_data) - time_step - 1):
        # 将x和y数据按时间步长拼接
        x_value = np.append(scaled_x_valid_data[i: i + time_step], scaled_y_valid_data[i: i + time_step], axis=1)
        x_valid.append(x_value)

        # 获取当前时间步长的y值
        y_value = scaled_y_valid_data[i + 1:i + time_step + 1]
        y_valid.append(y_value)

        # 获取当前时间步长的索引
        index = valid_indexs[i + time_step]
        valid_index.append(index)

    # 构建测试集
    # 获取测试集的x和y数据
    scaled_x_test_data = scaled_x_data[int(length * valid_ratio):]
    scaled_y_test_data = scaled_y_data[int(length * valid_ratio):]
    # 获取测试集的索引
    test_indexs = indexs[int(length * valid_ratio):]

    # 初始化测试集的x、y和索引列表
    x_test, y_test, test_index = [], [], []
    # 遍历测试数据，构建时间步长的数据
    for i in range(len(scaled_x_test_data) - time_step - 1):
        # 将x和y数据按时间步长拼接
        x_value = np.append(scaled_x_test_data[i: i + time_step], scaled_y_test_data[i: i + time_step], axis=1)
        x_test.append(x_value)

        # 获取当前时间步长的y值
        y_value = scaled_y_test_data[i + 1:i + time_step + 1]
        y_test.append(y_value)

        # 获取当前时间步长的索引
        index = test_indexs[i + time_step]
        test_index.append(index)

    # 返回归一化器和构建的训练集、验证集、测试集
    return (scaler_x, scaler_y), (x_train, y_train, train_index), \
        (x_valid, y_valid, valid_index), (x_test, y_test, test_index)
