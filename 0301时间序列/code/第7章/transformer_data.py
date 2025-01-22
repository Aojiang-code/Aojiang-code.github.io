import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 数据归一化处理, 构造训练集、验证集、测试集
# encoder_input_time_step Encoder端 Inputs的序列长度
# decoder_input_time_step Decoder端 Inputs的序列长度
# decoder_output_time_step Decoder端 Outputs的序列长度
# encoder_decoder_intersection Encoder与Decoder Inputs交集长度, 图6.46中交集为T14, 因此交集长度为1
# decoder_intersection Decoder端 Inputs与Outputs交集长度, 图6.46中交集为T15, 因此交集长度为1
def build_data(df, x_cols, y_cols, train_ratio, valid_ratio,
               encoder_input_time_step, decoder_input_time_step,
               decoder_output_time_step, encoder_decoder_intersection,
               decoder_intersection):
    indexs = df.index.tolist()
    length = len(df)

    # 数据归一化
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_x_data = scaler_x.fit_transform(df[x_cols])
    scaled_y_data = scaler_y.fit_transform(df[y_cols])

    # 计算一个Encoder-Deconder序列长度
    time_step = encoder_input_time_step + decoder_input_time_step + decoder_output_time_step - encoder_decoder_intersection - decoder_intersection

    # 计算序列起止位置
    x_encoder_start_index = 0
    x_encoder_end_index = encoder_input_time_step

    x_decoder_start_index = encoder_input_time_step - encoder_decoder_intersection
    x_decoder_end_index = encoder_input_time_step + decoder_input_time_step - encoder_decoder_intersection

    y_decoder_start_index = time_step - decoder_output_time_step
    y_decoder_end_index = time_step

    # 构建训练集
    scaled_x_train_data = scaled_x_data[0:int(length * train_ratio)]
    scaled_y_train_data = scaled_y_data[0:int(length * train_ratio)]
    train_indexs = indexs[0:int(length * train_ratio)]

    x_encoder_train, x_decoder_train, y_decoder_train, train_index = [], [], [], []
    for i in range(len(scaled_x_train_data) - time_step):
        x_time_step_seq = np.append(scaled_x_train_data[i: i + time_step],
                                    scaled_y_train_data[i: i + time_step], axis=1)
        y_time_step_seq = scaled_y_train_data[i: i + time_step]

        x_encoder_seq = x_time_step_seq[x_encoder_start_index:x_encoder_end_index]
        x_decoder_seq = x_time_step_seq[x_decoder_start_index:x_decoder_end_index]
        y_decoder_seq = y_time_step_seq[y_decoder_start_index:y_decoder_end_index]

        x_encoder_train.append(x_encoder_seq)
        x_decoder_train.append(x_decoder_seq)
        y_decoder_train.append(y_decoder_seq)

        index = train_indexs[i + time_step]
        train_index.append(index)

    # 构建验证集
    scaled_x_valid_data = scaled_x_data[int(length * train_ratio):int(length * valid_ratio)]
    scaled_y_valid_data = scaled_y_data[int(length * train_ratio):int(length * valid_ratio)]
    valid_indexs = indexs[int(length * train_ratio):int(length * valid_ratio)]

    x_encoder_valid, x_decoder_valid, y_decoder_valid, valid_index = [], [], [], []
    for i in range(len(scaled_x_valid_data) - time_step):
        x_time_step_seq = np.append(scaled_x_valid_data[i: i + time_step],
                                    scaled_y_valid_data[i: i + time_step], axis=1)
        y_time_step_seq = scaled_y_valid_data[i: i + time_step]

        x_encoder_seq = x_time_step_seq[x_encoder_start_index:x_encoder_end_index]
        x_decoder_seq = x_time_step_seq[x_decoder_start_index:x_decoder_end_index]
        y_decoder_seq = y_time_step_seq[y_decoder_start_index:y_decoder_end_index]

        x_encoder_valid.append(x_encoder_seq)
        x_decoder_valid.append(x_decoder_seq)
        y_decoder_valid.append(y_decoder_seq)

        index = valid_indexs[i + time_step]
        valid_index.append(index)

    # 构建测试集
    scaled_x_test_data = scaled_x_data[int(length * valid_ratio):]
    scaled_y_test_data = scaled_y_data[int(length * valid_ratio):]
    test_indexs = indexs[int(length * valid_ratio):]

    x_encoder_test, x_decoder_test, y_decoder_test, test_index = [], [], [], []
    for i in range(len(scaled_x_test_data) - time_step):
        x_time_step_seq = np.append(scaled_x_test_data[i: i + time_step],
                                    scaled_y_test_data[i: i + time_step], axis=1)
        y_time_step_seq = scaled_y_test_data[i: i + time_step]

        x_encoder_seq = x_time_step_seq[x_encoder_start_index:x_encoder_end_index]
        x_decoder_seq = x_time_step_seq[x_decoder_start_index:x_decoder_end_index]
        y_decoder_seq = y_time_step_seq[y_decoder_start_index:y_decoder_end_index]

        x_encoder_test.append(x_encoder_seq)
        x_decoder_test.append(x_decoder_seq)
        y_decoder_test.append(y_decoder_seq)

        index = test_indexs[i + time_step]
        test_index.append(index)

    return (scaler_x, scaler_y), \
        (x_encoder_train, x_decoder_train, y_decoder_train, train_index), \
        (x_encoder_valid, x_decoder_valid, y_decoder_valid, valid_index), \
        (x_encoder_test, x_decoder_test, y_decoder_test, test_index)
