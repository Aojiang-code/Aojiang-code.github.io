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
##### 1.2特征提取以及数据集的建立

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

`from keras.models import Sequential``

`model = Sequential()`

```python
model = Sequential()
```
###### 2.1.2 搭建CNN网络
```python
# 输入的大小
input_dim = (16, 8, 1)
```
卷积神经网络CNN的结构一般包含这几个层：

1)输入层：用于数据的输入

2)卷积层：使用卷积核进行特征提取和特征映射------>可以多次重复使用

3)激励层：由于卷积也是一种线性运算，因此需要增加非线性映射(也就是激活函数)

4)池化层：进行下采样，对特征图稀疏处理，减少数据运算量----->可以多次重复使用

5）Flatten操作：将二维的向量，拉直为一维的向量，从而可以放入下一层的神经网络中

6)全连接层：通常在CNN的尾部进行重新拟合，减少特征信息的损失----->DNN网络

对于Keras操作中，可以简单地使用 .add() ，将需要搭建的神经网络的layer堆砌起来，像搭积木一样：

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
如果需要，你还可以进一步地配置你的优化器.complies())。Keras 的核心原则是使事情变得相当简单，同时又允许用户在需要的时候能够进行完全的控制（终极的控制是源代码的易扩展性）。

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
##### 1.2特征提取以及数据集的建立

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

在未转置之前，`temp`的数据结构是一个二维列表。该列表包含两个子列表，第一个子列表存储特征数据，第二个子列表存储标签数据。每个子列表中的元素按照样本的顺序排列。

例如，如果 `temp` 数据如下所示：

```
temp = [[feature_1, feature_2, ..., feature_n],
        [label_1, label_2, ..., label_n]]
```

其中，`feature_i`表示第i个样本的特征，而`label_i`表示第i个样本的标签。这里的`n`代表样本数量。换句话说，`temp`的第一行代表特征数据，第二行代表标签数据。

在转置之后，`temp`的数据结构变为一个二维NumPy数组。转置操作将原来的行变为列，并且数组的形状也发生了变化。

如果在转置之前 `temp` 是一个二维数组，它可能看起来像这样：

```
temp = [[feature_1, feature_2, ..., feature_n],
        [label_1, label_2, ..., label_n]]
```

进行转置操作后，`temp` 变成了：

```
temp_transposed = [[feature_1, label_1],
                  [feature_2, label_2],
                  ...,
                  [feature_n, label_n]]
```

因此，转置后的 `temp` 具有 `n` 行和 2 列，其中每一行代表一个样本。第一列包含特征数据，第二列包含对应的标签数据。

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




```python
# 获取特征
X = np.vstack(data[:, 0])

# 获取标签
Y = np.array(data[:, 1])
print('X的特征尺寸是：',X.shape)
print('Y的特征尺寸是：',Y.shape)
```

这段代码用于从`data`数组中获取特征和标签，并打印它们的维度信息。

```python
# 获取特征
X = np.vstack(data[:, 0])

# 获取标签
Y = np.array(data[:, 1])
print('X的特征尺寸是：',X.shape)
print('Y的特征尺寸是：',Y.shape)
```



以下是对该代码的注释：

```python
# 从数组data中获取特征，即获取第一列的数据
X = np.vstack(data[:, 0])

# 从数组data中获取标签，即获取第二列的数据
Y = np.array(data[:, 1])

# 打印特征X的尺寸信息
print('X的特征尺寸是：', X.shape)

# 打印标签Y的尺寸信息
print('Y的特征尺寸是：', Y.shape)
```

解释：
- `data[:, 0]`: 这行代码使用切片操作`[:, 0]`从数组`data`中获取第一列的数据，表示特征。
- `np.vstack(data[:, 0])`: 这行代码使用NumPy的`vstack()`函数将所得的特征数组堆叠起来，使其成为一个垂直方向上排列的二维数组。`vstack()`函数会根据传入参数的维度自动决定如何进行堆叠操作。
- `data[:, 1]`: 这行代码使用切片操作`[:, 1]`从数组`data`中获取第二列的数据，表示标签。
- `np.array(data[:, 1])`: 这行代码将标签数据转换为NumPy数组。由于获取到的标签是一维的，使用`np.array()`函数将其转换为NumPy数组。
- `print('X的特征尺寸是：', X.shape)`: 这行代码打印特征`X`的尺寸信息，即它的形状（行数和列数）。
- `print('Y的特征尺寸是：', Y.shape)`: 这行代码打印标签`Y`的尺寸信息，即它的形状（行数和列数）。

请注意，在上述代码中，`data`数组必须是一个二维数组，并且第一列包含特征数据，第二列包含标签数据。




```python
# 在Keras库中：to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y = to_categorical(Y)
```
这段代码使用Keras库中的`to_categorical`函数将类别向量转换为二进制的矩阵类型表示。

转换后的Y矩阵将具有 N 行和 C 列的形状，其中 N 是样本数量，C 是类别数量（在这个例子中，C 等于20,因为食物的声音有二十种）。每一行代表一个样本，而每一列代表一个类别。元素的值要么为0要么为1，表示样本是否属于对应的类别。为0则表示不属于，为1则表示属于。

如果假设原始的类别标签向量`Y`如下所示：

```
Y = [1, 0, 2, 1, 2]
```

执行`to_categorical`函数后，将生成一个二进制矩阵类型表示的`Y`。该矩阵的形状将根据类别数量确定，然后每个样本的类别将以二进制形式表示。

对于上述的`Y`向量，在执行`to_categorical`函数后，可以得到以下数据结构：

```
Y = [[0, 1, 0],
     [1, 0, 0],
     [0, 0, 1],
     [0, 1, 0],
     [0, 0, 1]]
```

矩阵`Y`具有5行和3列，其中每一行代表一个样本。每一列都是一个二进制位，用于表示样本是否属于相应的类别。在这个例子中，共有3个不同的类别（column），分别用0和1表示。每一行中只有一个位置为1，其余位置为0，代表了样本所属的类别。

注意：上述代码使用了keras库的to_categorical函数。如果你在运行代码时遇到找不到模块名为keras的错误，请确认你已经正确安装并导入了keras库，并且库的版本符合要求。

```python
# 在Keras库中：to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y = to_categorical(Y)
```

以下是对该代码的注释：

```python
# 使用Keras库的to_categorical函数将类别向量转换为二进制矩阵表示
Y = to_categorical(Y)
```

解释：
- `to_categorical`函数是Keras库中的一个函数，用于将类别向量转换为二进制的矩阵类型表示。它可以将整数类别编码转换为独热编码(one-hot encoding)的形式。
- 在这段代码中，输入的参数`Y`是一个包含整数类别标签的向量。经过`to_categorical`函数处理后，被转换为一个二维的矩阵，其中每一行对应一个类别标签的独热编码表示。矩阵中只有一个元素为1，其余元素均为0。
- 转换后的结果存储在变量`Y`中，以便后续使用。
- 通过将类别向量转换为独热编码表示，可以将分类问题转化为多类别的逻辑回归问题，并更方便地用于神经网络的训练和预测。




```python
'''最终数据'''
print(X.shape)
print(Y.shape)
```

这段代码打印出处理后的最终数据的维度信息。

```python
'''最终数据'''
print(X.shape)
print(Y.shape)
```

以下是对该代码的注释：

```python
'''最终数据'''
# 打印特征X的维度信息
print(X.shape)

# 打印标签Y的维度信息
print(Y.shape)
```

解释：
- `print(X.shape)`: 这行代码打印处理后特征`X`的维度信息，即它的形状（行数和列数）。
- `print(Y.shape)`: 这行代码打印处理后标签`Y`的维度信息，即它的形状（行数和列数）。

其中，`X`和`Y`分别表示进行了一系列处理后得到的特征和标签数据。通过打印维度信息，可以了解最终数据的形状以及是否符合预期。

请注意，代码中使用了三引号字符串来注释`'''最终数据'''`，这是一种多行注释的方式，通常用于提供文件或函数的整体描述等情况。




```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1, stratify=Y)
print('训练集的大小',len(X_train))
print('测试集的大小',len(X_test))
```
上述代码使用了 `train_test_split` 函数来将数据集分割为训练集和测试集。训练集用于模型的训练，而测试集用于评估模型的性能。

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, stratify=Y)
print('训练集的大小:', len(X_train))
print('测试集的大小:', len(X_test))
```

下面是对上述代码的解释：

- `X`: 特征矩阵，包含了所有样本的特征数据。
- `Y`：目标向量，包含了所有样本的类别标签数据。

通过调用 `train_test_split(X, Y, random_state=1, stratify=Y)` 函数，我们将数据集按指定的比例（默认是 75% 训练集和 25% 测试集）进行划分，并且保持了原始数据集中样本类别的分布比例。

- `random_state` 参数用于设置随机数生成器的种子，使得每次运行结果都是一致的。
- `stratify` 参数表示按照类别标签进行分层抽样，确保训练集和测试集中的样本类别分布与原始数据集中的类别分布相同。

当使用 `stratify` 参数进行分层抽样时，意味着训练集和测试集中的样本在类别标签上的分布将与原始数据集中的类别标签分布相同。

具体来说，假设原始数据集中有两个类别：A 和 B。使用 `train_test_split` 进行分割时，默认情况下，会以一定比例（例如 75%:25%）随机选择样本构成训练集和测试集。然而，在这种默认情况下，由于随机性，可能存在训练集或测试集中某个类别的样本数量较少的情况。

但是，当设置 `stratify=Y` 时，函数会根据类别标签（`Y`）来确保在进行分割时，训练集和测试集中每个类别的样本数量与原始数据集中的每个类别的样本数量占比相同。这可以避免类别不平衡问题，从而更好地代表整个数据集的特征和性质。

例如，如果在原始数据集中，类别 A 占总样本的 60%，而类别 B 占 40%。使用 `stratify=Y` 分层抽样后，训练集和测试集中的样本类别比例也会保持近似为 60%:40%。

通过此分层抽样策略，可以提高模型在测试集上的可靠性，从而更好地评估和推广模型的性能。

当使用 `stratify` 参数进行分层抽样时，确保训练集和测试集中的样本类别分布与原始数据集中的类别分布相同。

让我们通过一个具体的例子来说明这个概念：

假设我们有一个二分类问题，原始数据集中有100个样本，其中70个属于类别A，30个属于类别B。我们想要将数据集划分为训练集和测试集。

在考虑到类别不平衡问题的情况下，我们可以使用 `stratify` 参数，确保每个类别在训练集和测试集中的比例保持一致。例如，如果我们将数据集按照 80%:20% 的比例进行划分，并使用 `stratify=Y` 进行分层抽样，那么在训练集和测试集中，类别A和类别B的样本数量将保持一致：

原始数据集：
- 类别A：70个样本
- 类别B：30个样本

划分后的训练集：
- 类别A：56个样本（80% * 70）
- 类别B：24个样本（80% * 30）

划分后的测试集：
- 类别A：14个样本（20% * 70）
- 类别B：6个样本（20% * 30）

通过这种分层抽样方式，我们可以确保训练集和测试集中的类别比例与原始数据集中的类别比例一致。这在处理类别不平衡问题时特别有用，能够更好地反映整个数据集的特征和性质。

划分后，将训练集的特征、测试集的特征、训练集的类别标签和测试集的类别标签分别保存在 `X_train`、`X_test`、`Y_train`、`Y_test` 变量中。

最后，通过打印输出 `len(X_train)` 和 `len(X_test)`，可以获取训练集和测试集的样本数量。

通过划分训练集和测试集，可以用训练集数据来训练模型参数，然后使用测试集数据来评估模型性能。




```python
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)
```
上述代码对 `X_train` 和 `X_test` 进行了形状重塑操作，调用了 `reshape` 方法。

```python
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)
```

下面是对上述代码的解释：

- `X_train` 是训练集的特征矩阵。
- `X_test` 是测试集的特征矩阵。

通过调用 `reshape` 方法，对特征矩阵进行了形状重塑操作。具体来说，使用参数 `-1` 表示根据其他维度的大小自动计算当前维度的大小。

这里的 `reshape` 操作将 `X_train` 和 `X_test` 从原始形状重新组织为新的形状。

具体解释如下：

- `X_train` 是训练集的特征矩阵。
- `X_test` 是测试集的特征矩阵。

通过调用 `reshape` 方法进行形状重塑，我们可以改变特征矩阵的维度和大小。在这个例子中，我们将每个样本重新组织为一个四维数组，以适应某些模型（例如卷积神经网络）的输入要求。

新的形状 `(样本数量, 高度, 宽度, 通道数)` 表示：

- 样本数量：根据原始数据集确定，可能会改变。
- 高度：代表数据的行数或分割长度。
- 宽度：代表数据的列数或每个分割时间步的特征维度。
- 通道数：表示特征的数量或在卷积操作中的通道数。

举个例子来说明：

假设原始数据集中有5个样本，每个样本是一个一维向量，大小为128。现在我们希望将这些样本重新组织成一个四维数组，其中每个样本按照16行和8列的形状进行分割，并将通道数设置为1。

- `X_train` 在形状重塑后变为 4 维数组：`(样本数量, 16, 8, 1)`
    - 样本数量：根据原始数据集确定，可能会改变。
    - 16：代表数据被分割成16行。
    - 8：代表每个分割时间步的特征维度（每行有8个特征）。
    - 1：表示特征在卷积操作中的通道数量。

- `X_test` 经过相同的形状重塑操作。

非常抱歉前面的回答仍存在错误。对于给定的例子，我将重新进行解释。

假设原始数据集中有1000个样本，每个样本是一个一维向量，大小为128。现在我们希望将这些样本重新组织成一个四维数组，其中每个样本按照16行和8列的形状进行分割，并将通道数设置为1。

根据题目描述，通过重塑形状操作后，`X_train` 变为一个四维数组：`(样本数量, 16, 8, 1)`。

让我们通过具体的数值示例来解释：

假设 `X_train` 中有5个样本，每个样本是一个一维向量，大小为128。那么 `X_train` 的初始形状将是 `(5, 128)`.

我们要将每个样本重新组织为16行和8列的形状，并将通道数设置为1。这意味着每个样本将被分割成16行和8列的结构，其中每个时间步（行）包含8个特征维度。最后一个维度1表示特征在卷积操作中的通道数量。

所以，从 `(5, 128)` 到 `(5, 16, 8, 1)` 的形状重塑过程如下所示：

- 样本1:

原始一维向量: `(128,)`

重塑后的数组: `(16, 8, 1)`

- 样本2:

原始一维向量: `(128,)`

重塑后的数组: `(16, 8, 1)`

- 样本3:

原始一维向量: `(128,)`

重塑后的数组: `(16, 8, 1)`

- 样本4:

原始一维向量: `(128,)`

重塑后的数组: `(16, 8, 1)`

- 样本5:

原始一维向量: `(128,)`

重塑后的数组: `(16, 8, 1)`

这样，原始数据集中的每个样本都被重新组织为一个形状为 `(5, 16, 8, 1)` 的四维数组。注意，具体的数值会根据数据集的大小和内容而变化，我在此处提供的示例是简化的表示方式。

通过这种形状重塑，我们可以将原始数据集重新组织成适应于需要固定输入形状的模型的结构，例如卷积神经网络。每个样本都按照一定行和列的形状重新排列，并通过最后一个维度的通道数表示该位置上的特征信息。

#### 2建立模型
##### 2.1 深度学习框架
Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。现在Keras已经和TensorFlow合并，可以通过TensorFlow来调用。
###### 2.1.1 网络结构搭建
Keras 的核心数据结构是 model，一种组织网络层的方式。最简单的模型是 Sequential 顺序模型，它由多个网络层线性堆叠。对于更复杂的结构，你应该使用 Keras 函数式 API，它允许构建任意的神经网络图。

Sequential模型可以直接通过如下方式搭建：

`from keras.models import Sequential`

`model = Sequential()`

```python
model = Sequential()
```
上述代码是用来创建一个Sequential模型。Sequential模型是Keras中一种常用的模型类型，用于按照顺序将各个层组合在一起构建深度神经网络。

通过创建Sequential对象并赋值给变量model后，我们可以使用该对象来添加各个层和配置模型的参数。

下面是对该代码的解释：

```python
model = Sequential()
```

这行代码创建了一个名为model的Sequential对象。

Sequential模型是一个线性堆叠的层，适用于大多数情况下的前馈神经网络。它允许我们按照顺序将各个层添加到模型中。

创建Sequential模型后，我们可以使用model对象来添加层（例如Dense、Conv2D等）和配置模型的各个参数（例如优化器、损失函数等）。

在实际构建深度神经网络时，我们会进一步添加各个层来定义模型的结构，并设置模型的超参数，如激活函数、输入形状、优化器和损失函数等，以便进行训练和预测任务。

以下是一个伪代码示例，展示如何使用Sequential模型添加两个全连接层，并对模型进行编译和训练：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 创建Sequential模型
model = Sequential()

# 添加第一个全连接层
model.add(Dense(units=64, activation='relu', input_shape=(input_dim,)))

# 添加第二个全连接层
model.add(Dense(units=10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 进行预测
y_pred = model.predict(X_test)
```

上述代码使用Sequential模型创建了一个具有两个全连接层的神经网络，并使用adam优化器和交叉熵损失函数进行模型编译。然后，通过调用fit方法对模型进行训练，并使用predict方法进行预测。

请注意，以上仅为示例代码，并不完整或可运行的代码。在实际使用时，需要根据具体的问题和数据来设置模型的参数和配置。


###### 2.1.2 搭建CNN网络

```python
# 输入的大小
input_dim = (16, 8, 1)
```
上述代码定义了一个变量 `input_dim`，它的值为 `(16, 8, 1)`。

`input_dim` 表示输入数据的大小或形状。在深度学习模型中，通常需要指定输入层的形状以确保模型能够处理正确的输入数据格式。

具体解释如下：

```python
input_dim = (16, 8, 1)
```

这行代码将一个元组 `(16, 8, 1)` 赋值给变量 `input_dim`。

上面的元组表示输入数据的大小或形状，其中：

- `16` 表示输入数据被分割成16个时间步（行）。
- `8` 表示每个时间步（行）有8个特征维度（每行有8个特征）。
- `1` 表示特征在卷积操作中的通道数量，即每个时间步（行）中只有1个通道。

这个 `(16, 8, 1)` 的形状描述适用于四维数组，用于表示一批样本，其中每个样本被重塑为 `(16, 8, 1)` 的形状。

在构建深度学习模型时，我们通常需要根据输入数据的实际情况来设置合适的形状。这可以通过观察原始数据集的样本形状来确定，同时也要考虑模型的设计需求和任务类型。在这种情况下，输入数据的大小被定义为 `(16, 8, 1)`，适用于处理数据分割成16行、每行有8个特征、1个通道的情况。




在卷积神经网络（CNN）的结构中，通常包含以下几个层：

1) 输入层：用于接收输入数据。输入层的形状需要与输入数据的维度相匹配。

2) 卷积层：卷积层使用一个或多个卷积核对输入数据进行特征提取和特征映射。每个卷积核通过滑动窗口在输入数据上移动，并计算出局部区域的卷积操作结果。卷积层可以多次重复使用，以增加模型的深度和表达能力。

3) 激活层：由于卷积操作本身是一种线性运算，而深度学习模型需要具备非线性建模能力，因此在卷积层之后添加激活函数层来引入非线性映射。常见的激活函数有ReLU、sigmoid、tanh等，在激活层中对卷积层输出进行非线性变换。

4) 池化层：池化层用于对特征图进行下采样，通过减少特征图的尺寸和参数数量来降低计算量。常用的池化层操作有最大池化和平均池化。池化层可以多次堆叠使用，以进一步减小特征图的尺寸。

5) Flatten层：Flatten操作用于将特征图展平为一维向量，将二维的空间结构转换为一维的特征向量。这样可以将池化层输出的特征图输入到后续的全连接层或其他类型的层中。

6) 全连接层：通常在CNN的尾部添加全连接层，将展平后的特征向量作为输入，利用全连接层来建立DNN（深度神经网络）结构进行重新拟合。全连接层用于对特征信息进行整合和处理，从而将特征映射到最终的输出类别或回归值。

在Keras中，我们可以使用 `.add()` 方法来搭建神经网络模型。通过调用 `.add()` 方法，并传递需要的层对象，可以将各个层堆叠在一起构建神经网络模型。例如：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

# 创建Sequential模型
model = Sequential()

# 添加卷积层
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))

# 添加激活层
model.add(Activation('relu'))

# 添加池化层
model.add(MaxPooling2D(pool_size=(2, 2)))

# 添加Flatten层
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=64, activation='relu'))

# 输出层
model.add(Dense(units=num_classes, activation='softmax'))
```

上面的代码演示了使用Keras构建CNN模型的常见操作。首先创建了一个Sequential模型，然后使用`.add()`方法按顺序添加了卷积层、激活层、池化层、Flatten层、全连接层以及最后的输出层。每个层都有相应的参数，例如卷积层的`filters`（过滤器数量）、`kernel_size`（卷积核大小），激活层的激活函数类型，全连接层的神经元数量等。

上述代码使用了Keras库中的Sequential模型来构建一个卷积神经网络（CNN）模型。

首先，通过引入需要的层对象和模型类型：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
```

然后，创建一个Sequential模型对象：

```python
model = Sequential()
```

接下来，按照顺序使用`.add()`方法将各个层添加到模型中：

```python
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
```

这行代码添加了一个卷积层。`Conv2D`表示二维卷积层，`filters=32`表示使用32个卷积核进行特征提取，`kernel_size=(3, 3)`表示每个卷积核的大小为3x3，`activation='relu'`表示在卷积操作后应用ReLU激活函数，`input_shape=(img_height, img_width, img_channels)`表示输入数据的形状。

```python
model.add(Activation('relu'))
```

这行代码添加了一个ReLU激活层。

```python
model.add(MaxPooling2D(pool_size=(2, 2)))
```

这行代码添加了一个最大池化层。`MaxPooling2D`表示二维最大池化层，`pool_size=(2, 2)`表示池化窗口大小为2x2。

```python
model.add(Flatten())
```

这行代码添加了一个Flatten层，用于将二维的特征图拉平为一维向量，方便后续全连接层处理。

```python
model.add(Dense(units=64, activation='relu'))
```

这行代码添加了一个全连接层。`Dense`表示全连接层，`units=64`表示该层有64个神经元，`activation='relu'`表示在全连接操作后应用ReLU激活函数。

```python
model.add(Dense(units=num_classes, activation='softmax'))
```

最后一行代码添加了输出层。`num_classes`表示输出的类别数量，`units=num_classes`表示该层有与类别数量相同数量的神经元，`activation='softmax'`表示在输出层应用softmax激活函数，用于多类别分类问题。

需要注意，上述代码中的`img_height`、`img_width`、`img_channels`和`num_classes`是需要根据具体问题设置的变量，在实际使用时需要填写合适的值。

总结：该代码片段构建了一个简单的CNN模型，包含卷积层、激活层、池化层、Flatten层和全连接层。最后的输出层可根据具体问题进行调整。

在实际构建CNN时，需要根据具体问题和网络结构需求来选择和配置合适的层，并设置相应的超参数。




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

这段代码构建了一个卷积神经网络模型，并依次添加了多个不同的层和配置参数。

```python
model.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=input_dim))  # 卷积层
model.add(MaxPool2D(pool_size=(2, 2)))  # 最大池化
model.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))  # 卷积层
model.add(MaxPool2D(pool_size=(2, 2)))  # 最大池化层
model.add(Dropout(0.1))
model.add(Flatten())  # 展开
model.add(Dense(1024, activation="tanh"))
model.add(Dense(20, activation="softmax"))  # 输出层：20个units输出20个类的概率
```

以下是对该代码的注释：

```python
# 添加一个卷积层，64个卷积核，每个卷积核大小为(3,3)，使用“same”边缘填充方式，激活函数为tanh，输入形状为input_dim
model.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=input_dim))

# 添加一个最大池化层，池化窗口大小为(2,2)
model.add(MaxPool2D(pool_size=(2, 2)))

# 添加一个卷积层，128个卷积核，每个卷积核大小为(3,3)，使用“same”边缘填充方式，激活函数为tanh
model.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))

# 添加一个最大池化层，池化窗口大小为(2,2)
model.add(MaxPool2D(pool_size=(2, 2)))

# 添加一个Dropout层，丢弃率为0.1，用于正则化防止过拟合
model.add(Dropout(0.1))

# 添加一个展开层，将多维输入数据展平成一维向量
model.add(Flatten())

# 添加一个全连接层，输出维度为1024，激活函数为tanh
model.add(Dense(1024, activation="tanh"))

# 添加一个全连接层，输出维度为20（即20个类别），激活函数为softmax（得到每个类别的概率）
model.add(Dense(20, activation="softmax"))
```

解释：
- 代码中使用`model.add()`方法逐层添加神经网络模型中的不同类型的层。
- `Conv2D` 表示卷积层，用于提取图像特征。第一个参数是卷积核的数量，其后的`(3, 3)`表示卷积核的尺寸。`padding="same"`表示采用相同的填充方式，保持输出特征图的大小与输入相同。`activation="tanh"`表示激活函数使用双曲正切函数。`input_shape` 是输入数据的形状。
- `MaxPool2D` 表示最大池化层，用于减少特征图的大小，保留重要的特征。`pool_size=(2, 2)` 意味着将特征图按(2, 2)窗口进行下采样，每个窗口取最大值作为输出特征值。
- `Dropout` 层是一种正则化技术，在训练过程中随机丢弃部分神经元，以减轻模型对训练数据的过拟合。
- `Flatten` 层用于将多维输入数据展平成一维向量，为后面的全连接层做准备。
- `Dense` 表示全连接层，即前一层的所有神经元与当前层的所有神经元都有连接关系。第一个参数指定了输出的维度或神经元数量，`activation` 参数指定了该层使用的激活函数。
- 在这段代码中，最后一个`Dense`层指定输出维度为20，代表共有20个类别。激活函数使用`softmax`，用于输出每个类别的概率。

通过添加不同的层和配置参数，可以构建出具有特定结构和功能的卷积神经网络模型。

上述代码使用Keras库构建了一个卷积神经网络（CNN）模型，下面对每一行代码进行详细解释：

```python
model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
```

这行代码添加了一个卷积层。`Conv2D`表示二维卷积层，`64`表示使用64个卷积核进行特征提取，`(3, 3)`表示每个卷积核的大小为3x3，`padding = "same"`表示在卷积操作时添加边界填充使得输入和输出的特征图大小保持相同，`activation = "tanh"`表示在卷积操作后应用tanh激活函数，`input_shape = input_dim`表示输入数据的形状。

```python
model.add(MaxPool2D(pool_size=(2, 2)))
```

这行代码添加了一个最大池化层。`MaxPool2D`表示二维最大池化层，`pool_size=(2, 2)`表示池化窗口大小为2x2。池化操作会对特征图进行下采样，减小特征图的大小。

```python
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
```

这行代码添加了另一个卷积层。与第一个卷积层类似，这里使用了128个卷积核，并且应用了tanh激活函数。

```python
model.add(MaxPool2D(pool_size=(2, 2)))
```

这行代码再次添加了一个最大池化层。

```python
model.add(Dropout(0.1))
```

这行代码添加了一个Dropout层，其目的是为了在训练过程中随机丢弃一部分神经元，以减少模型过拟合的风险。这里设置了丢弃概率为0.1，即丢弃10%的神经元。

```python
model.add(Flatten())
```

这行代码添加了一个Flatten层，将二维的特征图展平为一维向量，方便后续全连接层处理。

```python
model.add(Dense(1024, activation = "tanh"))
```

这行代码添加了一个全连接层。`Dense`表示全连接层，`1024`表示该层有1024个神经元，`activation = "tanh"`表示在全连接操作后应用tanh激活函数。

```python
model.add(Dense(20, activation = "softmax"))
```

最后一行代码添加了输出层。该层有20个神经元，对应着20个类别，`activation = "softmax"`表示在输出层应用softmax激活函数，用于多类别分类问题，输出每个类别的概率值。

所以，上述代码构建了一个具有两个卷积层、两个最大池化层以及一个全连接层的CNN模型。通过Flatten将特征图展平为一维向量，并添加Dropout层来减小过拟合风险。最后的输出层由20个神经元组成，用于预测20个不同类别的概率值。


如果需要，你还可以进一步地配置你的优化器.complies())。Keras 的核心原则是使事情变得相当简单，同时又允许用户在需要的时候能够进行完全的控制（终极的控制是源代码的易扩展性）。





```python
# 编译模型，设置损失函数，优化方法以及评价标准
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
```

上述代码中的`model.compile()`函数用于编译模型，并设置损失函数、优化方法和评价标准。

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

下面对其中的参数进行解释：

- `optimizer='adam'`：这里选择了Adam优化器作为模型的优化方法。Adam是一种常用的优化算法，它结合了动量梯度下降和自适应学习率方法，能够有效地加快收敛速度并提高模型性能。
- `loss='categorical_crossentropy'`：这里选择了交叉熵损失函数作为模型的损失函数。在多分类问题中，交叉熵损失函数是一种常用的选择，可以用于衡量模型输出与真实标签之间的差异。
- `metrics=['accuracy']`：这里设置了模型的评价标准为准确率。准确率是常用的分类模型评价指标之一，它表示模型预测正确的样本所占的比例。

通过使用`model.compile()`函数来配置模型的优化器、损失函数和评价标准后，模型就准备好进行训练了。在训练过程中，优化器将根据损失函数的值更新模型的权重，评价标准将用于衡量模型的性能。

#### CNN模型训练与测试
##### 3.1 模型训练
批量的在之前搭建的模型上训练：

```python
# 训练模型
model.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))
```
上述代码中的`model.fit()`函数用于训练模型，并设置了训练的相关参数。

```python
model.fit(X_train, Y_train, epochs=90, batch_size=50, validation_data=(X_test, Y_test))
```

下面对其中的参数进行解释：

- `X_train` 和 `Y_train`：训练数据集的输入特征和对应的标签。`X_train`是一个形状为`(样本数量, 特征数)`的numpy数组，包含训练样本的输入特征；`Y_train`是一个形状为`(样本数量, 类别数)`的numpy数组，包含训练样本的标签。注意，这里的输入特征和标签需要预先处理为适合模型输入的格式。
- `epochs=90`：表示将整个训练数据集迭代90次，每次迭代称为一个epoch。在每个epoch中，模型将根据训练数据进行权重更新。

在机器学习中，训练模型时通常使用迭代的方式进行。每次迭代中，模型将根据输入数据进行一次权重更新，以逐步优化模型的性能。一个epoch表示将整个训练数据集完整地传递给模型进行一次更新。

在训练神经网络模型时，一个epoch包含了以下几个步骤：

1. 将训练数据集分成若干个小批量(batch)。每个batch包含一定数量的训练样本。
2. 将一个batch的训练样本喂给模型进行前向传播和反向传播。
3. 根据反向传播计算得到的梯度来更新模型的权重和偏置。
4. 重复以上步骤，对每个batch进行更新，直到整个训练数据集都被处理完毕。这称为完成了一次epoch的训练。

通过设置`epochs`参数，可以指定希望让模型迭代的次数。在每个epoch中，模型会接收到数据集的完整副本并进行更新，以便在不同的数据样本上学习和调整权重。

对于`epochs=90`，表示模型将使用整个训练数据集进行90次的迭代更新。它将按照每次迭代处理所有训练样本的顺序来更新模型的权重。一般来说，增加epochs数量会增加训练时间，但可能提高模型的性能（对于足够大且具有代表性的数据集）。选择合适的epochs数量通常需要通过验证数据的结果和训练曲线来决定。

- `batch_size=50`：表示在每个epoch中，将训练数据划分为若干个大小为50的小批量(batch)，并使用这些小批量数据进行一次权重更新。通过使用小批量随机梯度下降(SGD)来训练模型，可以减少计算资源的占用，并更快地收敛到较好的解。

在训练机器学习模型时，数据集通常很大，并且无法一次性将所有样本放入模型进行训练。因此，为了高效地训练模型，可以采用小批量(batch)随机梯度下降(SGD)的方法。

`batch_size=50`表示每个epoch中将训练数据划分为大小为50的小批量（也可以理解为子集），并使用这些小批量数据来计算损失和梯度，然后通过反向传播更新模型的权重。

使用小批量随机梯度下降的优点包括：

1. **减少计算资源的占用**：相比于批量梯度下降(Batch Gradient Descent)，小批量随机梯度下降只需要处理部分数据，从而减少计算资源的占用。这对于较大的数据集和复杂的模型非常重要。
2. **更快地收敛到较好的解**：使用小批量数据更新模型的权重时，可以更频繁地进行权重更新，从而使模型更快地向最佳解决方案收敛。此外，使用随机抽样的小批量数据还能够有助于模型跳出局部最小值。

相应地，每个小批量数据（大小为50）都会通过前向传播计算损失，并使用反向传播计算梯度。然后，这些梯度将用于更新模型的参数（权重和偏置），从而调整模型以更好地拟合训练数据。

需要注意的是，选择合适的batch size是一个需要仔细考虑的问题。较大的batch size可能会导致更稳定的梯度估计，但也需要更多的内存和计算资源。较小的batch size可以使模型更快地收敛，并且在训练过程中对于局部最优点有更好的探索，较小的batch size可以使模型跳出当前的局部最优解状态，以更好地探索全局最优解的可能性。然而，选择合适的batch size并不是一个固定的规则，它依赖于特定问题的性质和可用的计算资源等因素。因此，实际情况下通常需要尝试不同的batch size，并通过验证数据上的性能评估来找到合适的平衡点。这样能够确保模型在训练过程中既能够有效地利用计算资源，又能够更好地探索全局最优解。

- `validation_data=(X_test, Y_test)`：用于验证模型性能的数据集。`X_test`和`Y_test`是形状相同的numpy数组，分别包含验证数据集的输入特征和对应的标签。在每个epoch结束后，模型将使用这些验证数据计算验证损失和验证指标，并输出验证结果。

通过使用`model.fit()`函数来进行模型的训练，在训练过程中，模型将根据训练数据进行权重更新，从而使模型能够更好地拟合训练数据，并最小化损失函数，同时，通过验证数据，我们可以观察模型在未见过的数据上的表现，以评估模型的泛化能力和预测准确性。训练过程将迭代多个epoch，并且每个epoch由若干个小批量数据组成。在训练完成后，可以通过观察训练过程中的损失和指标变化，以及最终的验证结果来评估模型的性能。



**查看网络的统计信息**

```python
model.summary()
```

上述代码中的 `model.summary()` 用于显示模型的概要信息，提供了关于模型结构、参数数量和层之间连接方式的详细信息。

具体解释每一部分如下：

- **Model**: 表示模型的名称或类型，这里是 "sequential"。
- **Layer (type)**: 每个层的名称和类型。在这个模型中，依次有 Conv2D、MaxPooling2D、Conv2D、MaxPooling2D、Dropout、Flatten、Dense 和 Dense 层。
- **Output Shape**: 显示每一层输出的形状。例如 `(None, 16, 8, 64)` 表示输出的形状是一个四维张量，其中 `None` 表示批量大小(batch size)在训练时可以是任意的，而 `(16, 8, 64)` 则表示每个样本的输出尺寸。
- **Param #:** 显示每层的参数数量。这些参数包括权重（weights）和偏置（biases）。模型通过学习这些参数来适应训练数据，并在后续的预测过程中使用它们。
    - 总参数数目 (Total params) 是所有层参数数量的总和。
    - 训练可更新的参数数目 (Trainable params) 是需要通过训练进行优化学习的参数数量。
    - 不可训练的参数数目 (Non-trainable params) 是不需要更新的参数数量，例如在使用预训练模型时，有些层的参数可能已经固定。

通过查看模型概要信息可以了解每个层的输入和输出形状，以及需要学习的参数数量。这对于理解模型的架构、调试和优化模型都很有帮助。


这个模型使用了一个序列（sequential）结构，其中包含了几个不同的层。

1. 第一层是一个卷积层（Conv2D） named `conv2d`。它有64个过滤器(filter)（也称为卷积核），每个过滤器的大小为 $16\times 8$ pixels。输出形状为 `(None, 16, 8, 64)`，其中 `None` 表示批量大小(batch size)。
   - 参数数量：640，表示在该层中需要学习或拟合的参数的数量。

2. 第二层是一个最大池化层（MaxPooling2D）named `max_pooling2d`。它对输入进行 $2\times 2$ 的窗口上的最大池化操作，以减小特征图的空间尺寸。
   - 输出形状为 `(None, 8, 4, 64)`。

3. 第三层是另一个卷积层（Conv2D）named `conv2d_1`。它有128个 $1\times 1$ 的过滤器，用于对前一层的特征图进行处理。
   - 参数数量：73,856。

4. 第四层是另一个最大池化层（MaxPooling2D）named `max_pooling2d_1`。
   - 输出形状为 `(None, 4, 2, 128)`。

5. 第五层是一个dropout层，在训练过程中将一部分神经元以给定的概率（通常为0.5）排除在外，以减少过拟合。
   - 输出形状为 `(None, 4, 2, 128)`。

6. 接下来是一个展平层（Flatten），用于将输入数据从二维形状转换为一维形状，以便进行全连接层（Dense）的操作。
   - 输出形状为 `(None, 1024)`。

7. 第七层是一个全连接层（Dense） named `dense`。它包含1,049,600个参数，该层的输出大小为1024。
   - 参数数量：1,049,600。

8. 最后一层也是一个全连接层（Dense） named `dense_1`，它将输入映射到最终的输出类别上。输出大小为20，表示模型被训练用于进行20个不同类别的分类任务。
   - 参数数量：20,500。

总参数数目为1,144,596，其中Trainable params表示需要通过训练进行学习或优化的参数数量，而Non-trainable params表示不需要更新的参数数量（例如对于某些预训练层来说）。这个模型的架构和参数统计提供了有关每个层如何相互连接并影响数据维度和参数数量的信息。

##### 3.2预测测试集
**新的数据生成预测**

```python
def extract_features(test_dir, file_ext="*.wav"):
    feature = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]): # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn,res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate).T,axis=0) # 计算梅尔频谱(mel spectrogram),并把它作为特征
        feature.extend([mels])
    return feature
```

这段代码定义了一个函数 `extract_features()`，用于从音频文件中提取特征。

```python
def extract_features(test_dir, file_ext="*.wav"):
    feature = [] # 用于保存提取的特征
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]): # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn,res_type='kaiser_fast') # 加载音频文件，并获取音频数据和采样率
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0) # 计算梅尔频谱(mel spectrogram)，并将其作为特征
        feature.extend([mels]) # 将提取的特征添加到特征列表中
    return feature # 返回提取的特征
```

以下是对该代码的注释：

```python
# 定义一个函数，用于从音频文件中提取特征
def extract_features(test_dir, file_ext="*.wav"):
    feature = []  # 用于保存提取的特征
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]):  # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn, res_type='kaiser_fast')  # 加载音频文件，并获取音频数据和采样率

        # 使用librosa提取音频的梅尔频谱特征，并取每个时间步的平均值，得到一维特征向量
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

        feature.extend([mels])  # 将提取的特征添加到特征列表中
    
    return feature  # 返回提取的特征
```

解释：
- `extract_features()` 函数定义了一个用于从音频文件中提取特征的过程。
- 在这段代码中，
    - `test_dir`: 测试目录的路径，用于存放需要提取特征的音频文件。
    - `file_ext`: 要匹配的文件扩展名，默认为 `"*.wav"`，表示提取所有扩展名为 `.wav` 的音频文件。
    - `feature` 是一个空列表，用于存储从音频文件中提取的特征。
    - `glob.glob(os.path.join(test_dir, file_ext))[:]` 遍历指定目录下所有符合文件扩展名的文件。使用 `glob.glob()` 函数遍历 `test_dir` 目录下符合 `file_ext` 扩展名规则的文件，并且通过列表切片操作 `[:])` 只选取前面的文件（可以修改切片操作来处理更多或全部文件）。
    - 对于每个音频文件 `fn` ，使用 `librosa.load()` 函数加载音频文件，并指定 `res_type='kaiser_fast'` 参数以使用快速的采样率转换方法，并返回音频数据 `X` 和采样率 `sample_rate`。
    - `librosa.feature.melspectrogram()` 计算音频信号 `X` 的梅尔频谱(mel spectrogram)特征。通过转置 `.T` 和对时间维度取平均值 `np.mean(..., axis=0)` 可以将其转换为一维特征向量。
    - 使用 `numpy.mean()` 函数对转置后的矩阵按列求均值，得到平均梅尔频谱。
    - 将平均梅尔频谱添加到 `feature` 列表中。
    - 循环结束后，返回存储了所有音频文件特征的 `feature` 列表。
- 这个函数使用了一些音频处理库的函数，例如 `librosa` 和 `numpy`，用于加载音频、计算梅尔频谱特征。通过调用这个函数，我们可以从音频文件中提取出有用的特征，用于后续的模型训练或其他任务。

这段代码使用了Librosa库来加载音频文件并计算梅尔频谱作为特征。梅尔频谱是一种在语音和音乐处理中常用的特征表示方式，可以用于训练机器学习模型或进行其他音频分析任务。函数通过遍历指定目录下的音频文件，并将每个文件的梅尔频谱作为特征进行提取和保存。




**保存预测的结果**

```python
X_test = extract_features('./test_a/')
```

这段代码调用了之前定义的 `extract_features()` 函数来从给定目录中提取音频特征，并将结果赋值给 `X_test` 变量。

```python
X_test = extract_features('./test_a/')
```

以下是对该代码的注释：

```python
# 从指定目录中提取音频文件的特征，并将结果存储在 X_test 变量中
X_test = extract_features('./test_a/')
```

解释：
- `X_test` 是一个变量，用于存储从音频文件中提取的特征。
- 在这段代码中，
    - `./test_a/` 是包含音频文件的目录路径。
    - `extract_features()` 函数被调用，传入 `./test_a/` 目录作为参数，以从该目录下的音频文件中提取特征。
    - `extract_features()` 函数会执行音频特征的提取过程，并将提取得到的特征返回。
    - 最后，返回的特征列表被赋值给 `X_test` 变量。

通过这段代码，我们可以得到 `X_test` 变量，其中包含了从 `./test_a/` 目录中的音频文件提取的特征。这些特征可以用于进一步的音频分析、模型预测等任务。



```python
X_test = np.vstack(X_test)
predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
```
这段代码对特征数据 `X_test` 进行预测。

```python
X_test = np.vstack(X_test)
predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
```

以下是对该代码的注释：

```python
# 对特征数据 X_test 进行预测
# 使用 np.vstack() 将特征列表转换为一个二维数组，以便进行预测
X_test = np.vstack(X_test)

# 调用模型的 predict() 方法进行预测
# 将特征数据 reshape 成符合输入要求的形状 (-1, 16, 8, 1)，其中 -1 表示不确定的批次大小
# predictions 变量保存了模型的预测结果
predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
```

解释：
- `X_test` 是之前从音频文件中提取的特征数据。
- 在这段代码中，
    - `np.vstack()` 函数将特征列表 `X_test` 垂直堆叠，转换为一个二维数组。这个步骤通常是因为模型要求输入为二维或多维数组的形式。
    - `.reshape(-1, 16, 8, 1)` 将特征数据重新排列成指定的形状 (-1, 16, 8, 1)。其中 `-1` 代表不确定的批次大小，`16` 和 `8` 分别表示特征图的高度和宽度，`1` 表示特征图的通道数。
    - `model.predict()` 方法调用模型进行预测，传入重新排列后的特征数据。模型将对这些特征进行处理，并生成相应的预测结果。
    - `predictions` 变量保存了模型的预测结果，可以在后续代码中使用这些预测结果进行进一步的分析、判定或可视化等操作。

通过以上代码，我们可以根据预先训练好的模型对提取的音频文件特征进行分类或预测工作，并通过 `predictions` 变量获取预测结果。















