## 食物声音分类Top1(无注释)
链接：[食物声音分类Top1](https://tianchi.aliyun.com/notebook/237212)
### 引言
在这次的声音语音识别中，本人很有幸获得了第一名的成绩，但本次赛题之前，我已经正在准备考研，所以我跑的次数不多，可能就几次的样子，emmm。。。能分数这么高的原因，可能是运气比较好，发现了一个trick，另外就是其余几场学习赛积累的经验，因为最近的学习赛都是有跟着datawhale的训练营，也很有幸遇到一些志趣相同的队友，作为一个新人，学会了很多。因为本次赛题我的尝试并不多，在复赛结束后，和一个初赛在第一页但复赛没实名的队友（没仔细看规则的后果，我心跳分类也是同样），深入交流了一波，所以介绍到的很多模型与代码将以他的为主。
### 赛题理解
#### 依赖包导入
```python
# 基本库
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import minmax_scale
# 搭建分类模型所需要的库

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
import librosa
import librosa.display
import glob 
from tqdm import tqdm
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from keras import regularizers

from sklearn.utils import class_weight
```

#### 数据分析与处理

```python
feature = []
label = []
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2,'candied_fruits':3, 'carrots': 4, 'chips':5,
                  'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream':11,
                  'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon':17,
                  'soup': 18, 'wings': 19}
label_dict_inv = {v:k for k,v in label_dict.items()}
```

```python
new_parent_dir = './soxed_train/'
old_parent_dir = './train/'
save_dir = "./"
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])
def sox_wavs(new_parent_dir, old_parent_dir, sub_dirs, max_file=1, file_ext="*.wav"):

    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(new_parent_dir, sub_dir)):
            os.makedirs(os.path.join(new_parent_dir, sub_dir))
        for fn in tqdm(glob.glob(os.path.join(old_parent_dir, sub_dir, file_ext))[:max_file]): # 遍历数据集的所有文件
            new_fn = fn.replace(old_parent_dir, new_parent_dir)
#             print(f'sox -b 16 -e signed-integer {fn} {new_fn}')
            os.system(f'sox -b 16 -e signed-integer {fn} {new_fn}')
sox_wavs(new_parent_dir, old_parent_dir, sub_dirs, max_file=10000, file_ext="*.wav")
```

```python
#抽取单样特征
def extract_mfcc(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]): # 遍历数据集的所有文件
            label_name = fn.split('\\')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn,res_type='kaiser_fast')
            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T 
            feature.append(mfcc)
            
    return [feature, label]
# 获取特征feature以及类别的label

# 自己更改目录
parent_dir = './clips_rd_sox/'
save_dir = "./"
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

mfcc_128_all,label = extract_mfcc(parent_dir,sub_dirs,max_file=1000)
```

不论是选用梅尔频谱还是梅尔倒谱，对于整体来讲，分数都是有所提升的，而这里选用的128过滤器，相比于baseline中的20，效果更好了，而在这之前的sox是linux中用于规整频率的一种工具，可能对于音频研究者来讲，会比较熟悉。

如果说按原始数据来走，上述应该是正常流程，但LSTM的baseline中给出了规整好的数据，我对比之下发现好像那份数据做了一定的增广，于是我就直接沿用了数据，这应该算trick？emmm，所以这是我提前退休的原因。

### 模型构建
#### CNN
```python
nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
prediction1 = np.zeros((2000,20 ))
# print(prediction1.shape)
i = 0
for train_index, valid_index in kf.split(X, Y):
    print("\nFold {}".format(i + 1))
    train_x, val_x = X[train_index],X[valid_index]
    train_y, val_y = Y[train_index],Y[valid_index]
    train_x = train_x.reshape(-1, 16, 8, 1)
    val_x = val_x.reshape(-1, 16, 8, 1)
    # print(train_x.shape)
    # print(val_x.shape)
    train_y = to_categorical(train_y)
    val_y = to_categorical(val_y)
    # print(train_y.shape)
    # print(val_y.shape)
    model = Sequential()
    model.add(Convolution2D(64, (5, 5),padding = "same", input_shape=input_dim, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3),padding = "same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    history = model.fit(train_x, train_y, epochs = 120, batch_size = 128, validation_data = (val_x, val_y))
    # X_test = np.vstack(X_test)
    predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
    print(predictions.shape)
    prediction1 += ((predictions)) / nfold
    i += 1
```
如果说前面数据处理用的规整好的数据，那么到这里以五折就能达到我初赛的分数，后面我就没再打了，隔壁心跳分类写的模型也没用了，备战复习数学去了。下面介绍一些其它模型。
#### DNN
```python
# 定义模型
def built_model():
    model_dense = Sequential()

    model_dense.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.2))
    model_dense.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.3))
    model_dense.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.005)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.3))
    model_dense.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.2))
    model_dense.add(Dense(20, activation='softmax')) # 输出层：20个units输出20个类的概率
    # 编译模型，设置损失函数，优化方法以及评价标准
    optimizer = optimizers.Adam(learning_rate=0.001)
    model_dense.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'], )
    return model_dense

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=20, mode='auto', factor=0.8 )
EarlyStop = EarlyStopping(monitor='val_accuracy', patience=200, verbose=1, mode='max')
```
这里的早停等策略感觉还能设置得更极端点，因为我感觉这数据其实泛化性挺好的。
#### lightgbm
```python
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 10
    seed = 2020
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed )

    train = np.zeros(train_x.shape[0])
    test = np.zeros((700,20))
    
    cv_scores = []
    
    print(train_y.shape)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x[train_index], train_y[train_index], train_x[valid_index], train_y[valid_index]
        
        if clf_name == "lgb":
            
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)
            
            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'metric': 'multi_error',
                'min_child_weight': 5,
                'num_leaves': 2 ** 4,
                'lambda_l2': 13,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 2,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 24,
                'n_jobs':24,
                'silent': True,
                'verbose': -1,
                'num_class':20,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                    
#         train[valid_index] += val_pred
#         print(test_pred.shape)
        test += test_pred / kf.n_splits
#         print(val_pred.shape)
        cv_scores.append(accuracy_score(val_y, np.argmax(val_pred, axis=1)))
        
#         print(cv_scores)
        
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test
```
### 模型融合
根据反馈，lightgbm调调参数能到94左右，而DNN和CNN都有97+的成绩，LSTM的效果看起来不行，可能这数据还是不太具有时序性，以原数据sox后128滤波，两个DNN和一CNN进行vote，能到99+，具体多少分队友没跟我说。整体来看，这题除了语音识别方向领域研究人员比较少，数据层面上区别挺大挺好预测的，这也是我职业生涯中第一次满分，emmm，太意外了。
```python
# 投票
def voting(preds_conv, preds_dense, preds_lstm):
    prob_max = np.tile(np.max(preds_conv, axis=1).reshape(-1, 1), preds_conv.shape[1])
    preds_c = preds_conv // prob_max
    prob_max = np.tile(np.max(preds_dense, axis=1).reshape(-1, 1), preds_dense.shape[1])
    preds_d = preds_dense // prob_max
    prob_max = np.tile(np.max(preds_lstm, axis=1).reshape(-1, 1), preds_lstm.shape[1])
    preds_l = preds_lstm // prob_max
    result_voting = preds_c + preds_d + preds_l
    preds_voting = np.argmax(result_voting, axis=1)
    return preds_voting
preds_voting = voting(preds_conv, preds_dense, preds_dense)
```
### 参赛感受
很感谢datawhale举办的这些学习赛，让我一个小白能迅速成长，参加过挺多期学习，从最早的几期开始断断续续，每次都能收获良多。大学电气工程毕业后，到现在转行编程干码农也有接近3年了，遇到了很多坎坷，当然也相遇了很多很有意思的人。今年我也准备踏出比较坚定的一步，接受系统性的学习，希望结果成真。

### 比赛建议
我觉得不管是啥比赛，只要不是完全没有头绪以及相关资料，其实都是可以做的，记得今年建筑物识别那个比赛，当时语义分割领域其实我理解实战比较少，但是有搜到很多的资料，我就包括模型都一个个去验证，最终单模到了89+拿了11名给了我很大鼓舞，于是就相继参加了后面的比赛。

这个比赛其实当时我都根本没想参加，当时今年规划还没定，还比较闲，然后记得当时同期的我想玩的两个早就满了，我才被迫来了。。。不过不管是任何比赛，只要坚持打下去，总能收获很多，虽然这比赛提前退休了，但不妨碍我的收获，emmm。。。

人生有梦，各自精彩。很荣幸能和一群大佬一起打比赛，明年再次相遇，希望还能来一场精彩的对决。


## 食物声音分类Top1(有注释)
链接：[食物声音分类Top1](https://tianchi.aliyun.com/notebook/237212)

### 引言
在这次的声音语音识别中，本人很有幸获得了第一名的成绩，但本次赛题之前，我已经正在准备考研，所以我跑的次数不多，可能就几次的样子，emmm。。。能分数这么高的原因，可能是运气比较好，发现了一个trick，另外就是其余几场学习赛积累的经验，因为最近的学习赛都是有跟着datawhale的训练营，也很有幸遇到一些志趣相同的队友，作为一个新人，学会了很多。因为本次赛题我的尝试并不多，在复赛结束后，和一个初赛在第一页但复赛没实名的队友（没仔细看规则的后果，我心跳分类也是同样），深入交流了一波，所以介绍到的很多模型与代码将以他的为主。
### 赛题理解
#### 依赖包导入
```python
# 基本库
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import minmax_scale
# 搭建分类模型所需要的库

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
import librosa
import librosa.display
import glob 
from tqdm import tqdm
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from keras import regularizers

from sklearn.utils import class_weight
```

#### 数据分析与处理

```python
feature = []
label = []
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2,'candied_fruits':3, 'carrots': 4, 'chips':5,
                  'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream':11,
                  'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon':17,
                  'soup': 18, 'wings': 19}
label_dict_inv = {v:k for k,v in label_dict.items()}
```

```python
new_parent_dir = './soxed_train/'
old_parent_dir = './train/'
save_dir = "./"
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])
def sox_wavs(new_parent_dir, old_parent_dir, sub_dirs, max_file=1, file_ext="*.wav"):

    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(new_parent_dir, sub_dir)):
            os.makedirs(os.path.join(new_parent_dir, sub_dir))
        for fn in tqdm(glob.glob(os.path.join(old_parent_dir, sub_dir, file_ext))[:max_file]): # 遍历数据集的所有文件
            new_fn = fn.replace(old_parent_dir, new_parent_dir)
#             print(f'sox -b 16 -e signed-integer {fn} {new_fn}')
            os.system(f'sox -b 16 -e signed-integer {fn} {new_fn}')
sox_wavs(new_parent_dir, old_parent_dir, sub_dirs, max_file=10000, file_ext="*.wav")
```

```python
#抽取单样特征
def extract_mfcc(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]): # 遍历数据集的所有文件
            label_name = fn.split('\\')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn,res_type='kaiser_fast')
            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T 
            feature.append(mfcc)
            
    return [feature, label]
# 获取特征feature以及类别的label

# 自己更改目录
parent_dir = './clips_rd_sox/'
save_dir = "./"
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

mfcc_128_all,label = extract_mfcc(parent_dir,sub_dirs,max_file=1000)
```

不论是选用梅尔频谱还是梅尔倒谱，对于整体来讲，分数都是有所提升的，而这里选用的128过滤器，相比于baseline中的20，效果更好了，而在这之前的sox是linux中用于规整频率的一种工具，可能对于音频研究者来讲，会比较熟悉。

如果说按原始数据来走，上述应该是正常流程，但LSTM的baseline中给出了规整好的数据，我对比之下发现好像那份数据做了一定的增广，于是我就直接沿用了数据，这应该算trick？emmm，所以这是我提前退休的原因。

### 模型构建
#### CNN
```python
nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
prediction1 = np.zeros((2000,20 ))
# print(prediction1.shape)
i = 0
for train_index, valid_index in kf.split(X, Y):
    print("\nFold {}".format(i + 1))
    train_x, val_x = X[train_index],X[valid_index]
    train_y, val_y = Y[train_index],Y[valid_index]
    train_x = train_x.reshape(-1, 16, 8, 1)
    val_x = val_x.reshape(-1, 16, 8, 1)
    # print(train_x.shape)
    # print(val_x.shape)
    train_y = to_categorical(train_y)
    val_y = to_categorical(val_y)
    # print(train_y.shape)
    # print(val_y.shape)
    model = Sequential()
    model.add(Convolution2D(64, (5, 5),padding = "same", input_shape=input_dim, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3),padding = "same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    history = model.fit(train_x, train_y, epochs = 120, batch_size = 128, validation_data = (val_x, val_y))
    # X_test = np.vstack(X_test)
    predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
    print(predictions.shape)
    prediction1 += ((predictions)) / nfold
    i += 1
```
如果说前面数据处理用的规整好的数据，那么到这里以五折就能达到我初赛的分数，后面我就没再打了，隔壁心跳分类写的模型也没用了，备战复习数学去了。下面介绍一些其它模型。
#### DNN
```python
# 定义模型
def built_model():
    model_dense = Sequential()

    model_dense.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.2))
    model_dense.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.3))
    model_dense.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.005)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.3))
    model_dense.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model_dense.add(BatchNormalization())
    model_dense.add(Dropout(0.2))
    model_dense.add(Dense(20, activation='softmax')) # 输出层：20个units输出20个类的概率
    # 编译模型，设置损失函数，优化方法以及评价标准
    optimizer = optimizers.Adam(learning_rate=0.001)
    model_dense.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'], )
    return model_dense

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=20, mode='auto', factor=0.8 )
EarlyStop = EarlyStopping(monitor='val_accuracy', patience=200, verbose=1, mode='max')
```
这里的早停等策略感觉还能设置得更极端点，因为我感觉这数据其实泛化性挺好的。
#### lightgbm
```python
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 10
    seed = 2020
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed )

    train = np.zeros(train_x.shape[0])
    test = np.zeros((700,20))
    
    cv_scores = []
    
    print(train_y.shape)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x[train_index], train_y[train_index], train_x[valid_index], train_y[valid_index]
        
        if clf_name == "lgb":
            
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)
            
            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'metric': 'multi_error',
                'min_child_weight': 5,
                'num_leaves': 2 ** 4,
                'lambda_l2': 13,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 2,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 24,
                'n_jobs':24,
                'silent': True,
                'verbose': -1,
                'num_class':20,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                    
#         train[valid_index] += val_pred
#         print(test_pred.shape)
        test += test_pred / kf.n_splits
#         print(val_pred.shape)
        cv_scores.append(accuracy_score(val_y, np.argmax(val_pred, axis=1)))
        
#         print(cv_scores)
        
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test
```
### 模型融合
根据反馈，lightgbm调调参数能到94左右，而DNN和CNN都有97+的成绩，LSTM的效果看起来不行，可能这数据还是不太具有时序性，以原数据sox后128滤波，两个DNN和一CNN进行vote，能到99+，具体多少分队友没跟我说。整体来看，这题除了语音识别方向领域研究人员比较少，数据层面上区别挺大挺好预测的，这也是我职业生涯中第一次满分，emmm，太意外了。
```python
# 投票
def voting(preds_conv, preds_dense, preds_lstm):
    prob_max = np.tile(np.max(preds_conv, axis=1).reshape(-1, 1), preds_conv.shape[1])
    preds_c = preds_conv // prob_max
    prob_max = np.tile(np.max(preds_dense, axis=1).reshape(-1, 1), preds_dense.shape[1])
    preds_d = preds_dense // prob_max
    prob_max = np.tile(np.max(preds_lstm, axis=1).reshape(-1, 1), preds_lstm.shape[1])
    preds_l = preds_lstm // prob_max
    result_voting = preds_c + preds_d + preds_l
    preds_voting = np.argmax(result_voting, axis=1)
    return preds_voting
preds_voting = voting(preds_conv, preds_dense, preds_dense)
```
### 参赛感受
很感谢datawhale举办的这些学习赛，让我一个小白能迅速成长，参加过挺多期学习，从最早的几期开始断断续续，每次都能收获良多。大学电气工程毕业后，到现在转行编程干码农也有接近3年了，遇到了很多坎坷，当然也相遇了很多很有意思的人。今年我也准备踏出比较坚定的一步，接受系统性的学习，希望结果成真。

### 比赛建议
我觉得不管是啥比赛，只要不是完全没有头绪以及相关资料，其实都是可以做的，记得今年建筑物识别那个比赛，当时语义分割领域其实我理解实战比较少，但是有搜到很多的资料，我就包括模型都一个个去验证，最终单模到了89+拿了11名给了我很大鼓舞，于是就相继参加了后面的比赛。

这个比赛其实当时我都根本没想参加，当时今年规划还没定，还比较闲，然后记得当时同期的我想玩的两个早就满了，我才被迫来了。。。不过不管是任何比赛，只要坚持打下去，总能收获很多，虽然这比赛提前退休了，但不妨碍我的收获，emmm。。。

人生有梦，各自精彩。很荣幸能和一群大佬一起打比赛，明年再次相遇，希望还能来一场精彩的对决。






