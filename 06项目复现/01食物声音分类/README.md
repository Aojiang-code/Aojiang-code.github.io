## 食物声音分类

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

这段代码的目的是创建空的特征和标签列表，并建立一个字典将类别标签映射到数字。

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


