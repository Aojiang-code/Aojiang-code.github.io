## 食物声音分类(无注释)
### 链接：[零基础入门语音识别-食物声音识别](https://tianchi.aliyun.com/competition/entrance/531887/introduction?spm=a2c22.28136470.0.0.201a4a0aLe7Mr7&from=search-list)

### Task4 食物声音识别-深度学习模型搭建与训练

#### 1 前情摘要

前面的task2与task3讲解了音频数据的分析以及特征提取等内容，本次任务主要是讲解CNN模型的搭建与训练，由于模型训练需要用到之前的特侦提取等得让，于是在此再贴一下相关代码。

##### 1.1提取特征

###### 1.1.1导入包

```python
#基本库
import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#深度学习框架
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow.keras

#音频处理库
import os
import librosa
import librosa.display
import glob 
```
###### 1.2特征提取以及数据集的建立

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

**建立提取音频特征的函数**

```python
from tqdm import tqdm
def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    c = 0
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]): # 遍历数据集的所有文件
            
           # segment_log_specgrams, segment_labels = [], []
            #sound_clip,sr = librosa.load(fn)
            #print(fn)
            label_name = fn.split('/')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn,res_type='kaiser_fast')
            mels = np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate).T,axis=0) # 计算梅尔频谱(mel spectrogram),并把它作为特征
            feature.extend([mels])
            
    return [feature, label]
```

```python
# 自己更改目录
parent_dir = './train_sample/'
save_dir = "./"
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

# 获取特征feature以及类别的label
temp = extract_features(parent_dir,sub_dirs,max_file=100)
```

```python
temp = np.array(temp)
data = temp.transpose()
```

```python
# 获取特征
X = np.vstack(data[:, 0])

# 获取标签
Y = np.array(data[:, 1])
print('X的特征尺寸是：',X.shape)
print('Y的特征尺寸是：',Y.shape)
```

```python
# 在Keras库中：to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y = to_categorical(Y)
```

```python
'''最终数据'''
print(X.shape)
print(Y.shape)
```

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1, stratify=Y)
print('训练集的大小',len(X_train))
print('测试集的大小',len(X_test))
```

```python
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)
```
#### 2建立模型
##### 2.1 深度学习框架

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。现在Keras已经和TensorFlow合并，可以通过TensorFlow来调用。
###### 2.1.1 网络结构搭建
Keras 的核心数据结构是 model，一种组织网络层的方式。最简单的模型是 Sequential 顺序模型，它由多个网络层线性堆叠。对于更复杂的结构，你应该使用 Keras 函数式 API，它允许构建任意的神经网络图。

Sequential模型可以直接通过如下方式搭建：

from keras.models import Sequential

model = Sequential()
```python
model = Sequential()
```
###### 2.1.2 搭建CNN网络
```python
# 输入的大小
input_dim = (16, 8, 1)
```

```python
model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))# 卷积层
model.add(MaxPool2D(pool_size=(2, 2)))# 最大池化
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh")) #卷积层
model.add(MaxPool2D(pool_size=(2, 2))) # 最大池化层
model.add(Dropout(0.1))
model.add(Flatten()) # 展开
model.add(Dense(1024, activation = "tanh"))
model.add(Dense(20, activation = "softmax")) # 输出层：20个units输出20个类的概率
```

```python
# 编译模型，设置损失函数，优化方法以及评价标准
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
```
#### CNN模型训练与测试
##### 3.1 模型训练

批量的在之前搭建的模型上训练：
```python
# 训练模型
model.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))
```
查看网络的统计信息
```python
model.summary()
```
##### 3.2预测测试集
新的数据生成预测
```python
def extract_features(test_dir, file_ext="*.wav"):
    feature = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]): # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn,res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate).T,axis=0) # 计算梅尔频谱(mel spectrogram),并把它作为特征
        feature.extend([mels])
    return feature
```
保存预测的结果
```python
X_test = extract_features('./test_a/')
```

```python
X_test = np.vstack(X_test)
predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
```

```python
preds = np.argmax(predictions, axis = 1)
preds = [label_dict_inv[x] for x in preds]

path = glob.glob('./test_a/*.wav')
result = pd.DataFrame({'name':path, 'label': preds})

result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
result.to_csv('submit.csv',index=None)
```

```python
!ls ./test_a/*.wav | wc -l
```

```python
!wc -l submit.csv
```
以上就是深度学习模型搭建与训练的全部内容。请尽情享受科技之光吧，少年！

## 食物声音分类(有注释)
### 链接：[零基础入门语音识别-食物声音识别](https://tianchi.aliyun.com/competition/entrance/531887/introduction?spm=a2c22.28136470.0.0.201a4a0aLe7Mr7&from=search-list)

### Task4 食物声音识别-深度学习模型搭建与训练

#### 1 前情摘要

前面的task2与task3讲解了音频数据的分析以及特征提取等内容，本次任务主要是讲解CNN模型的搭建与训练，由于模型训练需要用到之前的特侦提取等得让，于是在此再贴一下相关代码。

##### 1.1提取特征

###### 1.1.1导入包

```python
#基本库
import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#深度学习框架
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow.keras

#音频处理库
import os
import librosa
import librosa.display
import glob 
```
###### 1.2特征提取以及数据集的建立

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

这段代码的目的是创建空的特征和标签列表，并建立一个字典将类别标签映射到数字。以下进行逐行解读：

```python
feature = []
label = []
```
这两行代码创建了两个空列表`feature`和`label`，用于存储特征数据和对应的标签数据。

```python
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2,'candied_fruits':3, 'carrots': 4, 'chips':5,
                  'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream':11,
                  'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon':17,
                  'soup': 18, 'wings': 19}
```
这段代码定义了一个字典`label_dict`，其中每一个键值对都表示一种类别及其对应的数字标签。例如，'aloe'对应标签0，'burger'对应标签1，以此类推。这个字典可以用于将类别名称转换为数字标签。

```python
label_dict_inv = {v:k for k,v in label_dict.items()}
```
这行代码通过将`label_dict`的键值对颠倒，创建了字典`label_dict_inv`。现在，它可以用于将数字标签转换回类别名称。

因此，这段代码旨在为特征和标签数据创建空列表，并提供了一种从类别名称到数字标签的映射方法，以及从数字标签到类别名称的反向映射。

**建立提取音频特征的函数**

```python
from tqdm import tqdm
def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    c = 0
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]): # 遍历数据集的所有文件
            
           # segment_log_specgrams, segment_labels = [], []
            #sound_clip,sr = librosa.load(fn)
            #print(fn)
            label_name = fn.split('/')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn,res_type='kaiser_fast')
            mels = np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate).T,axis=0) # 计算梅尔频谱(mel spectrogram),并把它作为特征
            feature.extend([mels])
            
    return [feature, label]
```
```python
#上面的代码块添加注释后码如下所示：
from tqdm import tqdm

# 导入所需库

def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    c = 0
    label, feature = [], []

    # 遍历每个子目录
    for sub_dir in sub_dirs:
        # 使用glob.glob()获取子目录下符合file_ext的文件，并限制最大处理文件数量为max_file
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):

            # 提取音频文件的类别标签
            label_name = fn.split('/')[-2]
            label.extend([label_dict[label_name]])

            # 加载音频文件并获取音频数据和采样率
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')

            # 计算梅尔频谱特征
            mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

            # 将特征和标签添加到对应的列表中
            feature.extend([mels])

    return [feature, label]

```
这段代码定义了一个名为`extract_features`的函数，用于提取音频文件的特征。以下进行逐行解读：

```python
from tqdm import tqdm
```
该行导入了`tqdm`库，它用于在循环中显示进度条。

```python
def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    c = 0
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):
            ...
```
这部分是函数定义的开始。`extract_features`函数接受四个参数：`parent_dir`（父目录路径），`sub_dirs`（子目录列表），`max_file`（最大提取文件数，默认为10），`file_ext`（文件扩展名，默认为"*.wav"）。

在函数内部，初始化了两个空列表`label`和`feature`，用于存储标签和特征数据。然后，通过对每个子目录进行循环，使用`tqdm`遍历每个子目录下指定文件扩展名的文件。

```python
label_name = fn.split('/')[-2]
label.extend([label_dict[label_name]])
```
此代码将文件路径中的子目录名称作为类别名称，并使用预先定义的`label_dict`字典将其转换为数字标签。然后，将该数字标签添加到`label`列表中。

```python
# 使用librosa.load()函数加载音频文件，并将返回的音频数据存储在变量X中，采样率存储在变量sample_rate中
X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
feature.extend([mels])
```
这段代码使用`librosa`库加载音频文件，并使用梅尔频谱（Mel Spectrogram）作为特征之一。首先，通过`librosa.load()`函数加载音频文件，然后计算梅尔频谱，并取平均值以得到一个特征向量 `mels`。最后将该特征向量添加到 `feature` 列表中。

```python
return [feature, label]
```
函数的返回语句返回包含特征列表和标签列表的列表。

因此，这个函数的主要功能是从指定的父目录和子目录中提取音频文件的特征，并返回特征列表和对应的标签列表。



```python
# 自己更改目录
parent_dir = './train_sample/'
save_dir = "./"
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

# 获取特征feature以及类别的label
temp = extract_features(parent_dir,sub_dirs,max_file=100)
```

这段代码包含了一些变量定义和函数调用。

```python
# 自己更改目录
parent_dir = './train_sample/'  # 指定包含训练样本的父目录路径
save_dir = "./"  # 指定保存提取特征的目录路径
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])  # 定义包含子目录名称的数组

# 调用函数提取特征和标签
temp = extract_features(parent_dir, sub_dirs, max_file=100)
```

在这段代码中，`parent_dir`变量指定了包含训练样本的父目录路径，`save_dir`变量指定了保存提取特征的目录路径。`folds`和`sub_dirs`都是包含子目录名称的数组，用于表示不同类别的分类标签。

然后，通过调用`extract_features`函数来提取特征和标签。函数接受三个参数：父目录路径、子目录列表和最大文件数（默认为100）。调用结果被存储在名为`temp`的变量中。

请注意，您需要根据自己的目录结构和需求修改这些路径和参数，以正确加载数据并将特征提取到您希望保存的位置。


```python
temp = np.array(temp)
data = temp.transpose()
```

这段代码进行了两个操作：首先将`temp`转换为NumPy数组，然后对该数组进行转置。

```python
temp = np.array(temp)
data = temp.transpose()
```

以下是对该代码的注释：

```python
# 将列表temp转换为NumPy数组
temp = np.array(temp)

# 对数组temp进行转置
data = temp.transpose()
```

解释：
- `np.array(temp)`: 这行代码使用NumPy的`array()`函数将列表`temp`转换为NumPy数组。由于NumPy数组具有更广泛的功能和更高效的计算能力，转换为NumPy数组可以方便地进行各种操作和分析。
- `temp.transpose()`: 这行代码使用NumPy数组的`transpose()`方法对数组进行转置操作。转置操作将数组的行与列互换位置，如果初始数组是二维数组，则每一列会变成新数组的一行，每一行会变成新数组的一列。转置操作常用于矩阵操作或者改变数组的形状。
- `data = temp.transpose()`: 转置后的数组存储在变量`data`中，以便后续使用。











