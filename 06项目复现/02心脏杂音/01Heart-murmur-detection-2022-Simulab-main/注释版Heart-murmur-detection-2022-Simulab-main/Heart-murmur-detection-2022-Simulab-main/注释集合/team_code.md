


```python
#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.
```


```python  
################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import os
import tqdm
import numpy as np
import tensorflow as tf
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
```



```python  
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

 
# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    PRE_TRAIN = False
    NEW_FREQUENCY = 100 # longest signal, while resampling to 500Hz = 32256 samples
    EPOCHS_1 = 30
    EPOCHS_2 = 20
    BATCH_SIZE_1 = 20
    BATCH_SIZE_2 = 20

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    #TODO: remove this:
    #classes = ['Present', 'Unknown', 'Absent']
    #num_classes = len(classes)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    data = []
    murmurs = list()
    outcomes = list()
    

    for i in tqdm.tqdm(range(num_patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings, freq = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        for j in range(len(current_recordings)):
            data.append(signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * NEW_FREQUENCY)))
            current_auscultation_location = current_patient_data.split('\n')[1:len(current_recordings)+1][j].split(" ")[0]
            all_murmur_locations = get_murmur_locations(current_patient_data).split("+")
            current_murmur = np.zeros(num_murmur_classes, dtype=int)
            if get_murmur(current_patient_data) == "Present":
                if current_auscultation_location in all_murmur_locations:
                    current_murmur[0] = 1
                else:
                    pass
            elif get_murmur(current_patient_data) == "Unknown":
                current_murmur[1] = 1
            elif get_murmur(current_patient_data) == "Absent":
                current_murmur[2] = 1
            murmurs.append(current_murmur)

            current_outcome = np.zeros(num_outcome_classes, dtype=int)
            outcome = get_outcome(current_patient_data)
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[j] = 1
            outcomes.append(current_outcome)

    data_padded = pad_array(data)
    data_padded = np.expand_dims(data_padded,2)

    murmurs = np.vstack(murmurs)
    outcomes = np.argmax(np.vstack(outcomes),axis=1)
    print(f"Number of signals = {data_padded.shape[0]}")

    # The prevalence of the 3 different labels
    print("Murmurs prevalence:")
    print(f"Present = {np.where(np.argmax(murmurs,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(murmurs,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(murmurs,axis=1)==2)[0].shape[0]}")

    print("Outcomes prevalence:")
    print(f"Abnormal = {len(np.where(outcomes==0)[0])}, Normal = {len(np.where(outcomes==1)[0])}")

    new_weights_murmur=calculating_class_weights(murmurs)
    keys = np.arange(0,len(murmur_classes),1)
    murmur_weight_dictionary = dict(zip(keys, new_weights_murmur.T[1]))
  
    weight_outcome = np.unique(outcomes, return_counts=True)[1][0]/np.unique(outcomes, return_counts=True)[1][1]
    outcome_weight_dictionary = {0: 1.0, 1:weight_outcome}

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_2, verbose=0)

    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        if PRE_TRAIN == False:
            # Initiate the model.
            clinical_model = build_clinical_model(data_padded.shape[1],data_padded.shape[2])
            murmur_model = build_murmur_model(data_padded.shape[1],data_padded.shape[2])
        elif PRE_TRAIN == True:
            model = base_model(data_padded.shape[1],data_padded.shape[2])
            model.load_weights("./pretrained_model.h5")
            
            outcome_layer = tf.keras.layers.Dense(1, "sigmoid",  name="clinical_output")(model.layers[-2].output)
            clinical_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[outcome_layer])
            clinical_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(curve='ROC')])

            murmur_layer = tf.keras.layers.Dense(3, "softmax",  name="murmur_output")(model.layers[-2].output)
            murmur_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[murmur_layer])
            murmur_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(curve='ROC')])
        


        murmur_model.fit(x=data_padded, y=murmurs, epochs=EPOCHS_1, batch_size=BATCH_SIZE_1,   
                    verbose=1, shuffle = True,
                    class_weight=murmur_weight_dictionary
                    #,callbacks=[lr_schedule]
                    )

        clinical_model.fit(x=data_padded, y=outcomes, epochs=EPOCHS_2, batch_size=BATCH_SIZE_2,   
                    verbose=1, shuffle = True,
                    class_weight=outcome_weight_dictionary
                    #,callbacks=[lr_schedule]
                    )
    
    murmur_model.save(os.path.join(model_folder, 'murmur_model.h5'))

    clinical_model.save(os.path.join(model_folder, 'clinical_model.h5'))

    # Save the model.
    #save_challenge_model(model_folder, classes, imputer, classifier)
```
**注释如下：**


```python
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    PRE_TRAIN = False  # 是否预训练
    NEW_FREQUENCY = 100  # 重采样后的频率
    EPOCHS_1 = 30  # 第一次训练的迭代次数
    EPOCHS_2 = 20  # 第二次训练的迭代次数
    BATCH_SIZE_1 = 20  # 第一次训练的批量大小
    BATCH_SIZE_2 = 20  # 第二次训练的批量大小

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)  # 查找患者数据文件
    num_patient_files = len(patient_files)  # 患者数据文件数量

    if num_patient_files == 0:
        raise Exception('No data was provided.')  # 如果没有提供数据文件，则引发异常

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)  # 如果模型保存文件夹不存在，则创建它

    murmur_classes = ['Present', 'Unknown', 'Absent']  # 心脏杂音的类别
    num_murmur_classes = len(murmur_classes)  # 心脏杂音的类别数量
    outcome_classes = ['Abnormal', 'Normal']  # 结果的类别
    num_outcome_classes = len(outcome_classes)  # 结果的类别数量

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    data = []  # 数据列表
    murmurs = list()  # 心脏杂音标签列表
    outcomes = list()  # 结果标签列表

    for i in tqdm.tqdm(range(num_patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])  # 加载当前患者数据
        current_recordings, freq = load_recordings(data_folder, current_patient_data, get_frequencies=True)  # 加载患者心音图记录和频率信息
        for j in range(len(current_recordings)):
            data.append(signal.resample(current_recordings[j], int((len(current_recordings[j]) / freq[j]) * NEW_FREQUENCY)))  # 重采样心音图
            current_auscultation_location = current_patient_data.split('\n')[1:len(current_recordings) + 1][j].split(" ")[0]  # 获取听诊位置
            all_murmur_locations = get_murmur_locations(current_patient_data).split("+")  # 获取所有的心脏杂音位置
            current_murmur = np.zeros(num_murmur_classes, dtype=int)  # 初始化心脏杂音类别数组
            if get_murmur(current_patient_data) == "Present":
                if current_auscultation_location in all_murmur_locations:
                    current_murmur[0] = 1  # 设置心脏杂音为“Present”
                else:
                    pass
            elif get_murmur(current_patient_data) == "Unknown":
                current_murmur[1] = 1  # 设置心脏杂音为“Unknown”
            elif get_murmur(current_patient_data) == "Absent":
                current_murmur[2] = 1  # 设置心脏杂音为“Absent”
            murmurs.append(current_murmur)

            current_outcome = np.zeros(num_outcome_classes, dtype=int)  # 初始化结果类别数组
            outcome = get_outcome(current_patient_data)  # 获取结果信息
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[j] = 1  # 设置结果标签
            outcomes.append(current_outcome)

    data_padded = pad_array(data)  # 填充数据
    data_padded = np.expand_dims(data_padded, 2)  # 添加维度

    murmurs = np.vstack(murmurs)  # 堆叠心脏杂音标签
    outcomes = np.argmax(np.vstack(outcomes), axis=1)  # 获取最大值索引作为结果标签

    print(f"Number of signals = {data_padded.shape[0]}")  # 输出信号数量

    # The prevalence of the 3 different labels
    print("Murmurs prevalence:")
    print(f"Present = {np.where(np.argmax(murmurs, axis=1) == 0)[0].shape[0]}, Unknown = {np.where(np.argmax(murmurs, axis=1) == 1)[0].shape[0]}, Absent = {np.where(np.argmax(murmurs, axis=1) == 2)[0].shape[0]}")

    print("Outcomes prevalence:")
    print(f"Abnormal = {len(np.where(outcomes == 0)[0])}, Normal = {len(np.where(outcomes == 1)[0])}")

    new_weights_murmur = calculating_class_weights(murmurs)  # 计算心脏杂音的类别权重
    keys = np.arange(0, len(murmur_classes), 1)
    murmur_weight_dictionary = dict(zip(keys, new_weights_murmur.T[1]))  # 构建心脏杂音类别权重字典

    weight_outcome = np.unique(outcomes, return_counts=True)[1][0] / np.unique(outcomes, return_counts=True)[1][1]  # 计算结果类别权重
    outcome_weight_dictionary = {0: 1.0, 1: weight_outcome}  # 构建结果类别权重字典

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_2, verbose=0)  # 学习率调度器

    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)  # 使用多GPU训练
    with strategy.scope():
        if PRE_TRAIN == False:
            # Initiate the model.
            clinical_model = build_clinical_model(data_padded.shape[1], data_padded.shape[2])  # 构建临床数据模型
            murmur_model = build_murmur_model(data_padded.shape[1], data_padded.shape[2])  # 构建心脏杂音模型
        elif PRE_TRAIN == True:
            model = base_model(data_padded.shape[1], data_padded.shape[2])
            model.load_weights("./pretrained_model.h5")  # 加载预训练模型

            outcome_layer = tf.keras.layers.Dense(1, "sigmoid", name="clinical_output")(model.layers[-2].output)
            clinical_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[outcome_layer])
            clinical_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                   metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(curve='ROC')])  # 编译临床数据模型

            murmur_layer = tf.keras.layers.Dense(3, "softmax", name="murmur_output")(model.layers[-2].output)
            murmur_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[murmur_layer])
            murmur_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                 metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(curve='ROC')])  # 编译心脏杂音模型

        murmurd_model.fit(x=data_padded, y=murmurs, epochs=EPOCHS_1, batch_size=BATCH_SIZE_1,
                          verbose=1, shuffle=True,
                          class_weight=murmur_weight_dictionary
                          # ,callbacks=[lr_schedule]
                          )  # 使用心脏杂音模型进行训练

        clinical_model.fit(x=data_padded, y=outcomes, epochs=EPOCHS_2, batch_size=BATCH_SIZE_2,
                           verbose=1, shuffle=True,
                           class_weight=outcome_weight_dictionary
                           # ,callbacks=[lr_schedule]
                           )  # 使用临床数据模型进行训练

    murmur_model.save(os.path.join(model_folder, 'murmur_model.h5'))  # 保存心脏杂音模型

    clinical_model.save(os.path.join(model_folder, 'clinical_model.h5'))  # 保存临床数据模型

```


```python

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    model_dict = {}
    for i in os.listdir(model_folder):
        model = tf.keras.models.load_model(os.path.join(model_folder, i))
        model_dict[i.split(".")[0]] = model    
    return model_dict

```

```python

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    NEW_FREQUENCY = 100

    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    # Load the data.
    #indx = get_lead_index(data)
    #extracted_recordings = np.asarray(recordings)[indx]
    new_sig_len = model["murmur_model"].get_config()['layers'][0]['config']['batch_input_shape'][1]
    data_padded = np.zeros((len(recordings),int(new_sig_len),1))
    freq = get_frequency(data)
    murmur_probabilities_temp = np.zeros((len(recordings),3))
    outcome_probabilities_temp = np.zeros((len(recordings),1))

    for i in range(len(recordings)):
        data = np.zeros((1,new_sig_len,1))
        rec = np.asarray(recordings[i])
        resamp_sig = signal.resample(rec, int((len(rec)/freq) * NEW_FREQUENCY))
        data[0,:len(resamp_sig),0] = resamp_sig
        
        murmur_probabilities_temp[i,:] = model["murmur_model"].predict(data)
        outcome_probabilities_temp[i,:] = model["clinical_model"].predict(data)

    avg_outcome_probabilities = np.sum(outcome_probabilities_temp)/len(recordings)
    avg_murmur_probabilities = np.sum(murmur_probabilities_temp,axis = 0)/len(recordings)

    binarized_murmur_probabilities = np.argmax(murmur_probabilities_temp, axis = 1)
    binarized_outcome_probabilities = (outcome_probabilities_temp > 0.5) * 1

    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    #murmur_indx = np.bincount(binarized_murmur_probabilities).argmax()
    #murmur_labels[murmur_indx] = 1
    if 0 in binarized_murmur_probabilities:
        murmur_labels[0] = 1
    elif 2 in binarized_murmur_probabilities:
        murmur_labels[2] = 1
    elif 1 in binarized_murmur_probabilities:
        murmur_labels[1] = 1

    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    # 0 = abnormal outcome
    if 0 in binarized_outcome_probabilities:
        outcome_labels[0] = 1
    else:
        outcome_labels[1] = 1

                                                        
    outcome_probabilities = np.array([avg_outcome_probabilities,1-avg_outcome_probabilities])
    murmur_probabilities = avg_murmur_probabilities


    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities.ravel(), outcome_probabilities.ravel()))
    
    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


```

```python

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)


```

```python


def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x



```

```python



def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x



```

```python



def base_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    return model




```

```python



def build_murmur_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    murmur_output = tf.keras.layers.Dense(3, activation='softmax', name="murmur_output")(gap_layer)
    #clinical_output = tf.keras.layers.Dense(1, activation='sigmoid', name="clinical_output")(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=murmur_output)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = [tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.AUC(curve='ROC')])
    return model



```

```python


def build_clinical_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    clinical_output = tf.keras.layers.Dense(1, activation='sigmoid', name="clinical_output")(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=clinical_output)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = [tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.AUC(curve='ROC')])
    
    return model




```

```python


def get_lead_index(patient_metadata):    
    lead_name = []
    lead_num = []
    cnt = 0
    for i in patient_metadata.splitlines(): 
        if i.split(" ")[0] == "AV" or i.split(" ")[0] == "PV" or i.split(" ")[0] == "TV" or i.split(" ")[0] == "MV":
            if not i.split(" ")[0] in lead_name:
                lead_name.append(i.split(" ")[0])
                lead_num.append(cnt)
            cnt += 1
    return np.asarray(lead_num)



```

```python


def scheduler(epoch, lr):
    if epoch == 10:
        return lr * 0.1
    elif epoch == 15:
        return lr * 0.1
    elif epoch == 20:
        return lr * 0.1
    else:
        return lr

```

```python



def scheduler_2(epoch, lr):
    return lr - (lr * 0.1)

```

```python


def get_murmur_locations(data):
    murmur_location = None
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            try:
                murmur_location = l.split(': ')[1]
            except:
                pass
    if murmur_location is None:
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return murmur_location


```

```python



def pad_array(data, signal_length = None):
    max_len = 0
    for i in data:
        if len(i) > max_len:
            max_len = len(i)
    if not signal_length == None:
        max_len = signal_length
    new_arr = np.zeros((len(data),max_len))
    for j in range(len(data)):
        new_arr[j,:len(data[j])] = data[j]
    return new_arr

```

```python

def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight(class_weight='balanced', classes=[0.,1.], y=y_true[:, i])
    return weights
```


# 以下内容应删除










































```python
################################################################################
#
# 必需的函数。编辑这些函数以添加您的代码，但请不要更改参数。
#
################################################################################


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class ECGDataset(Dataset):
    def __init__(self, data, murmurs, outcomes, transform=None):
        self.data = data
        self.murmurs = murmurs
        self.outcomes = outcomes
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ecg = self.data[idx]
        murmur = self.murmurs[idx]
        outcome = self.outcomes[idx]
        
        if self.transform:
            ecg = self.transform(ecg)
        
        return ecg, murmur, outcome

    
def train_challenge_model(data_folder, model_folder, verbose):
    # 查找数据文件。
    if verbose >= 1:
        print('查找数据文件...')

    PRE_TRAIN = False  # pre_train # 是否预训练的标志
    NEW_FREQUENCY = 100  # new_frequency# 最长信号，在重新采样为500Hz时是32256个样本
    EPOCHS_1 = 30  # 第一阶段的训练epoch数
    EPOCHS_2 = 20  # 第二阶段的训练epoch数
    BATCH_SIZE_1 = 20  # 第一阶段的批大小
    BATCH_SIZE_2 = 20  # 第二阶段的批大小

    # 查找患者数据文件。
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception('未提供任何数据。')

    # 如果模型文件夹不存在，则创建一个模型文件夹。
    os.makedirs(model_folder, exist_ok=True)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    # 从挑战数据中提取特征和标签。
    if verbose >= 1:
        print('从挑战数据中提取特征和标签...')

    data = []
    murmurs = list()
    outcomes = list()

    for i in tqdm.tqdm(range(num_patient_files)):
        # 加载当前患者的数据和录音。
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings, freq = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        
        for j in range(len(current_recordings)):
            data.append(signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * NEW_FREQUENCY)))
            current_auscultation_location = current_patient_data.split('\n')[1:len(current_recordings)+1][j].split(" ")[0]
            all_murmur_locations = get_murmur_locations(current_patient_data).split("+")
            current_murmur = np.zeros(num_murmur_classes, dtype=int)
            
            if get_murmur(current_patient_data) == "Present":
                if current_auscultation_location in all_murmur_locations:
                    current_murmur[0] = 1
                else:
                    pass
            elif get_murmur(current_patient_data) == "Unknown":
                current_murmur[1] = 1
            elif get_murmur(current_patient_data) == "Absent":
                current_murmur[2] = 1
            
            murmurs.append(current_murmur)

            current_outcome = np.zeros(num_outcome_classes, dtype=int)
            outcome = get_outcome(current_patient_data)
            
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[j] = 1
                
            outcomes.append(current_outcome)

    data_padded = pad_array(data)
    data_padded = np.expand_dims(data_padded, 2)

    murmurs = np.vstack(murmurs)
    outcomes = np.argmax(np.vstack(outcomes), axis=1)
    print(f"信号数量 = {data_padded.shape[0]}")

    # 3个不同标签的患者比例
    print("心脏杂音比例:")
    print(f"存在 = {np.where(np.argmax(murmurs,axis=1)==0)[0].shape[0]}, 未知 = {np.where(np.argmax(murmurs,axis=1)==1)[0].shape[0]}, 不存在 = {np.where(np.argmax(murmurs,axis=1)==2)[0].shape[0]}")

    print("何种结果占多数:")
    print(f"异常 = {len(np.where(outcomes==0)[0])}, 正常 = {len(np.where(outcomes==1)[0])}")

    new_weights_murmur=calculating_class_weights(murmurs)
    keys = np.arange(0,len(murmur_classes),1)
    murmur_weight_dictionary = dict(zip(keys, new_weights_murmur.T[1]))

    weight_outcome = np.unique(outcomes, return_counts=True)[1][0]/np.unique(outcomes, return_counts=True)[1][1]
    outcome_weight_dictionary = {0: 1.0, 1:weight_outcome}

    # Initialize the clinical model and the murmur model
    if PRE_TRAIN == False:
        clinical_model = build_clinical_model(data_padded.shape[1], data_padded.shape[2])
        murmur_model = build_murmur_model(data_padded.shape

```




```python
import os
import numpy as np
import tensorflow as tf
import tqdm
from scipy import signal
from sklearn.utils.class_weight import compute_class_weight

def find_patient_files(data_folder):
    """
    查找指定文件夹中的患者数据文件。

    参数:
        data_folder (str): 数据文件夹的路径。

    返回:
        List[str]: 患者数据文件的路径列表。
    """
    patient_files = []
    # 遍历数据文件夹中的文件
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".txt"):
            patient_files.append(os.path.join(data_folder, file_name))
    return patient_files


def load_patient_data(file_path):
    """
    从文本文件中加载患者数据。

    参数:
        file_path (str): 患者数据文件的路径。

    返回:
        str: 患者数据内容。
    """
    with open(file_path, "r") as f:
        data = f.read()
    return data


def load_recordings(data_folder, patient_data, get_frequencies=False):
    """
    加载患者的心听图记录以及每个记录的频率（如果需要）。

    参数:
        data_folder (str): 数据文件夹的路径。
        patient_data (str): 从文本文件中读取的患者数据。
        get_frequencies (bool): 是否返回记录的频率。

    返回:
        Tuple[np.ndarray, Optional[List[float]]]: 包含心听图记录的NumPy数组和可选的频率列表的元组。
    """
    recordings = []  # 心音记录数组
    frequencies = []  # 记录的频率数组

    lines = patient_data.strip().split("\n")
    for i in range(1, len(lines)):
        file_name, freq = lines[i].strip().split(" ")
        freq = float(freq)

        # 加载记录
        file_path = os.path.join(data_folder, file_name)
        recording = np.loadtxt(file_path, delimiter=",")
        recordings.append(recording)
        
        if get_frequencies:
            frequencies.append(freq)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings


def get_murmur_locations(patient_data):
    """
    获取患者心脏杂音的位置。

    参数:
        patient_data (str): 从文本文件中读取的患者数据。

    返回:
        str: 逗号分隔的杂音位置字符串。
    """
    lines = patient_data.strip().split("\n")
    return lines[0].strip()


def get_murmur(patient_data):
    """
    获取患者是否存在心脏杂音。

    参数:
        patient_data (str): 从文本文件中读取的患者数据。

    返回:
        str: 表示杂音状态的字符串，可能的取值有"Present"（存在）、"Unknown"（不确定）或"Absent"（不存在）。
    """
    lines = patient_data.strip().split("\n")
    return lines[-1].strip()


def get_outcome(patient_data):
    """
    获取患者的诊断结果。

    参数:
        patient_data (str): 从文本文件中读取的患者数据。

    返回:
        str: 表示患者诊断结果的字符串，可能的取值有"Abnormal"（异常）或"Normal"（正常）。
    """
    lines = patient_data.strip().split("\n")
    return lines[0].strip().split(",")[-1].strip()


def pad_array(data):
    """
    将心听图记录填充为相同长度。采用使用零进行填充。

    参数:
        data (List[np.ndarray]): 心听图记录的列表。

    返回:
        np.ndarray: 填充后的心听图记录的NumPy数组。
    """
    max_length = max(len(record) for record in data)
    padded_data = []
    for record in data:
        pad_length = max_length - len(record)
        padded_record = np.pad(record, (0, pad_length), mode="constant")
        padded_data.append(padded_record)
    return np.array(padded_data)


def calculating_class_weights(labels):
    """
    计算不平衡数据集的类别权重。

    参数:
        labels (np.ndarray): 类别标签的NumPy数组。

    返回:
        np.ndarray: 类别权重的NumPy数组。
    """
    class_weights = compute_class_weight("balanced", np.unique(labels), labels)
    return class_weights


def build_clinical_model(input_shape, num_classes):
    """
    构建临床数据模型。

    参数:
        input_shape (tuple): 输入数据的形状。
        num_classes (int): 类别数。

    返回:
        tf.keras.models.Model: 构建好的临床数据模型。
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))
    return model


def build_murmur_model(input_shape, num_classes):
    """
    构建心听图模型。

    参数:
        input_shape (tuple): 输入数据的形状。
        num_classes (int): 类别数。

    返回:
        tf.keras.models.Model: 构建好的心听图模型。
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return model


def scheduler_2(epoch, lr):
    """
    学习率调度器。

    参数:
        epoch (int): 当前训练轮数。
        lr (float): 当前学习率。

    返回:
        float: 更新后的学习率。
    """
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_challenge_model(data_folder, model_folder, verbose):
    """
    使用提供的数据训练挑战模型。

    参数:
        data_folder (str): 包含患者数据文件的文件夹路径。
        model_folder (str): 训练好的模型将保存在指定的文件夹路径。
        verbose (int): 控制输出信息的详细程度。设置为0以禁用打印，设置为1以打印重要信息。

    返回:
        None
    """

    # 在这里添加根据问题要求训练挑战模型的代码。
    # 确保按要求打印相关信息，并将训练好的模型保存在指定的模型文件夹中。








