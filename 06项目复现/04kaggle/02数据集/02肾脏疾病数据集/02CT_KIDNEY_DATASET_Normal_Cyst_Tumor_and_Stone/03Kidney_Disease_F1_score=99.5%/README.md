# Kidney Disease F1 score= 99.5%

## 目录
* 1. Import Needed Modules
* 2 Concept for Callback Approach
* 3. Define function to print text in rgb foreground and background colors
* 4. Read in images and create a dataframe of image paths and class labels
* 5. Trim the trainning set
* 6. Create train, test and validation generators
* 7. Create a function to show Training Image Samples
* 8 Create a function that computes the F1 score metric 
* 9. Create the Model
* 10. Create a custom Keras callback to continue or halt training
* 11. Instantiate custom callback 
* 12. Train the model
* 13. Define a function to plot the training data
* 14. Make predictions on test set, create Confusion Matrix and Classification Report
* 15 Define a function to print list of misclassified test files
* 16 Save the model



## 1. Import Needed Modules

```python
# 导入 pandas 库，并使用别名 pd。Pandas 是一个强大的数据结构和数据分析工具。
import pandas as pd

# 导入 numpy 库，并使用别名 np。NumPy 提供了多维数组对象和一系列处理数组的函数。
import numpy as np

# 导入 os 模块，用于与操作系统进行交互，如文件路径操作和环境变量设置。
import os

# 设置环境变量，用于控制 TensorFlow 产生的日志信息级别。
# '2' 表示只显示警告和错误信息，不显示调试信息。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入 time 模块，提供各种时间相关的函数。
import time

# 导入 matplotlib.pyplot 模块，并使用别名 plt。Matplotlib 是一个用于创建静态、动态和交互式图表的库。
import matplotlib.pyplot as plt

# 导入 cv2 模块，即 OpenCV，是一个开源的计算机视觉和图像处理库。
import cv2

# 导入 seaborn 库，并使用别名 sns。Seaborn 是一个基于 Matplotlib 的数据可视化库，提供了更多样化的绘图风格和接口。
import seaborn as sns

# 设置 seaborn 的绘图风格为 'darkgrid'。
sns.set_style('darkgrid')

# 导入 shutil 模块，提供文件操作功能，如复制、移动和删除文件。
import shutil

# 从 sklearn.metrics 中导入 confusion_matrix 和 classification_report 函数。
# 这些函数用于生成混淆矩阵和分类报告，常用于评估分类模型的性能。
from sklearn.metrics import confusion_matrix, classification_report

# 从 sklearn.model_selection 中导入 train_test_split 函数。
# 该函数用于将数据集分割为训练集和测试集。
from sklearn.model_selection import train_test_split

# 导入 tensorflow 库，并使用别名 tf。TensorFlow 是一个开源的机器学习框架。
import tensorflow as tf

# 从 tensorflow.keras 模块中导入相关层、优化器、损失函数、正则化器、模型和后端。
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# 再次导入 time 模块。
import time

# 导入 tqdm 模块，提供进度条功能，用于显示循环的进度。
from tqdm import tqdm

# 从 sklearn.metrics 中导入 f1_score 函数。
# F1 分数是精确率和召回率的调和平均，常用于评估二分类和多分类问题的性能。
from sklearn.metrics import f1_score

# 从 IPython.display 中导入 YouTubeVideo 类。
from IPython.display import YouTubeVideo

# 导入 sys 模块，用于访问与 Python 解释器相关的变量和函数。
import sys

# 如果 sys.warnoptions 没有设置，则导入 warnings 模块，并设置警告过滤器。
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# 设置 pandas 显示选项，以显示更多的列、行和更宽的列内容。
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

# 打印提示信息，表示所有模块已经加载完成。
print('Modules loaded')
```

这段代码导入了多个 Python 库和模块，用于数据处理、图像处理、机器学习模型构建、可视化和环境配置等任务。通过设置环境变量和模块选项，代码还配置了 TensorFlow 和 pandas 的显示行为，以及忽略了警告信息，以便在开发过程中减少不必要的干扰。最后，打印了 'Modules loaded' 信息，表示所有需要的模块已经成功加载。


```python
Modules loaded
```





## 2 Concept for Callback Approach

### Custom Callback Concept
This notebook implements a custom callback to adjust the learning rate during training.
The callback has a parameter dwell. If dwell is set to True, the callback monitors the
validation loss. It keeps track of the lowest validation loss thus far achieved as you run
through each epoch and stores this as the lowest loss and also stores the weights for that
epoch as the best weights. At the end of an epoch the validation loss for that epoch is
compared with the lowest loss. If the validation loss at the end of the current epoch is
less than the lowest loss than it becomes the lowest loss and the weights of the current
epoch become the best weights

If the validation loss at the end of the current epoch is greator than the lowest loss
this implies you have moved to a location in Nspace(N is the number of trainable parameters
on the validation cost function surface that is less favorable(higher cost) than the position
in Nspace defined by the best weights. Therefore why move the models weights to this less
favorable location? Better to reset the models weights to the best weights, then lower the
learning rate and run more epochs. The new learning rate is set to new_lr=current_lr * factor
where factor is a user specified parameter in the instantiation of the callback. By default
it is set to .04 and by default dwell is set to True.

At the end of training the callback always returns your model with the weights set to the
best weights. The callback provides a feature where it periodically queries the user to
either contine and optionally manually specify a new learning rate or halt training.
During training the calback provides useful information on the percent improvement in the
validation loss for each epoch. The is useful to decide when to halt training or manually
specifying a new learning rate.



## 3. Define function to print text in rgb foreground and background colors
Add some PZAZZ to your printed output with this function

form of the call is: print_in_color(txt_msg, fore_tupple, back_tupple where:

* txt_msg is the string to be printed out
* fore_tuple is tuple of the form (r,g,b) specifying the foreground color of the text
* back_tuple is tuple of the form (r,g,b) specifying the background color of the text


```python
# 定义一个函数 print_in_color，用于以指定的前景色和背景色打印文本消息。
# 参数说明：
# txt_msg: 需要打印的文本消息。
# fore_tupple: 一个元组，表示前景色，格式为 (r, g, b)，代表红、绿、蓝三个颜色通道的强度，默认值为 (0, 255, 255)，即青色。
# back_tupple: 一个元组，表示背景色，格式同样为 (r, g, b)，默认值为 (100, 100, 100)，即灰色。
def print_in_color(txt_msg, fore_tupple=(0, 255, 255), back_tupple=(100, 100, 100)):
    
    # 解包前景色和背景色的元组。
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    
    # 构造消息字符串，这里只是简单地在文本前加上了 '{0}'，实际上这个字符串可以更复杂。
    msg = '{0}' + txt_msg
    
    # 构造 ANSI 转义序列，用于设置前景色和背景色。
    # \33[38;2; 后面跟随的是前景色的颜色代码，';48;2;' 后面是背景色的颜色代码。
    # 这些颜色代码指定了 RGB 颜色空间中的颜色值。
    mat = '\33[38;2;' + str(rf) + ';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(bb) + 'm'
    
    # 使用 format 方法将 mat 插入到消息字符串中，并打印出来。
    # flush=True 确保立即刷新输出，确保颜色代码能够立即生效。
    print(msg.format(mat), flush=True)
    
    # 打印默认的重置代码 '\33[0m'，将终端的颜色设置恢复到默认状态（黑色背景）。
    print('\33[0m', flush=True)
    
    # 函数执行完毕，返回 None。
    return

# 示例：使用默认参数打印文本消息 'test of default colors'。
# 由于使用了默认的颜色参数，文本将以青色前景色和灰色背景色打印出来。
msg = 'test of default colors'
print_in_color(msg)
```

这段代码定义了一个函数 `print_in_color`，它允许用户以自定义的颜色打印文本消息。通过使用 ANSI 转义序列，可以在支持 ANSI 颜色代码的终端中以不同的颜色显示文本。函数的默认颜色是青色前景色和灰色背景色，但也可以通过参数自定义颜色。在函数的末尾，使用了一个重置代码来确保终端的颜色设置恢复到默认状态。

```python
test of default colors
```
## 4. Read in images and create a dataframe of image paths and class labels




```python
# 定义一个函数 make_dataframes，用于创建包含图像文件路径和标签的数据框架（DataFrames）。
# sdir 参数是图像数据集的根目录路径。
def make_dataframes(sdir): 
    filepaths = []  # 用于存储图像文件路径的列表。
    labels = []  # 用于存储标签的列表。
    classlist = sorted(os.listdir(sdir))  # 获取数据集根目录下的所有类别，并排序。
    for klass in classlist:
        classpath = os.path.join(sdir, klass)  # 构造每个类别的完整路径。
        if os.path.isdir(classpath):
            flist = sorted(os.listdir(classpath))  # 获取当前类别下所有图像文件，并排序。
            desc = f'{klass:25s}'  # 构造进度条描述信息。
            for f in tqdm(flist, ncols=130, desc=desc, unit='files', colour='blue'):
                fpath = os.path.join(classpath, f)  # 构造单个图像文件的完整路径。
                filepaths.append(fpath)  # 将文件路径添加到 filepaths 列表。
                labels.append(klass)  # 将对应的类别标签添加到 labels 列表。

    # 创建一个包含图像文件路径的 pandas Series 对象。
    Fseries = pd.Series(filepaths, name='filepaths')
    # 创建一个包含标签的 pandas Series 对象。
    Lseries = pd.Series(labels, name='labels')
    # 将两个 Series 对象合并成一个 DataFrame。
    df = pd.concat([Fseries, Lseries], axis=1)
    
    # 使用 train_test_split 函数将 DataFrame 分割为训练集和测试集。
    train_df, dummy_df = train_test_split(df, train_size=.7, shuffle=True, random_state=123, stratify=df['labels'])
    # 再次分割，将剩余的数据分为验证集和测试集。
    valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])
    
    # 获取训练集中所有唯一类别并排序。
    classes = sorted(train_df['labels'].unique())
    # 计算类别数量。
    class_count = len(classes)
    # 从训练集中随机抽取 50 个样本。
    sample_df = train_df.sample(n=50, replace=False)
    
    # 计算图像的平均高度和宽度。
    ht = 0
    wt = 0
    count = 0
    for i in range(len(sample_df)):
        fpath = sample_df['filepaths'].iloc[i]
        try:
            img = cv2.imread(fpath)
            h = img.shape[0]
            w = img.shape[1]
            wt += w
            ht += h
            count += 1
        except:
            pass
    have = int(ht / count)  # 计算平均高度。
    wave = int(wt / count)  # 计算平均宽度。
    aspect_ratio = have / wave  # 计算宽高比。
    
    # 打印一些统计信息。
    print('number of classes in processed dataset= ', class_count)
    counts = list(train_df['labels'].value_counts())
    print(counts[0], type(counts[0]))
    print('the maximum files in any class in train_df is ', max(counts), '  the minimum files in any class in train_df is ', min(counts))
    print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
    print('average image height= ', have, '  average image width= ', wave, ' aspect ratio h/w= ', aspect_ratio)
    
    # 返回创建的训练集、测试集、验证集、类别列表和类别数量。
    return train_df, test_df, valid_df, classes, class_count

# 指定图像数据集的根目录路径。
sdir = r'../input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
# 调用函数并获取结果。
train_df, test_df, valid_df, classes, class_count = make_dataframes(sdir)
```

这段代码定义了一个函数 `make_dataframes`，用于处理图像数据集并创建相关的 DataFrames。函数首先遍历数据集的根目录，收集所有图像文件的路径和对应的标签，然后使用 `train_test_split` 函数将数据集分割为训练集、验证集和测试集。此外，函数还计算了图像的平均高度和宽度，并打印了一些有用的统计信息。最后，函数返回创建的 DataFrames 和类别相关的信息。在代码的最后，调用了 `make_dataframes` 函数并传入了数据集的路径。




```python
Cyst                     : 100%|███████████████████████████████████████████████████████| 3709/3709 [00:00<00:00, 214016.89files/s]
Normal                   : 100%|███████████████████████████████████████████████████████| 5077/5077 [00:00<00:00, 544992.23files/s]
Stone                    : 100%|███████████████████████████████████████████████████████| 1377/1377 [00:00<00:00, 446092.27files/s]
Tumor                    : 100%|███████████████████████████████████████████████████████| 2283/2283 [00:00<00:00, 398650.96files/s]
number of classes in processed dataset=  4
3554 <class 'int'>
the maximum files in any class in train_df is  3554   the minimum files in any class in train_df is  964
train_df length:  8712   test_df length:  1867   valid_df length:  1867
average image height=  575   average image width=  629  aspect ratio h/w=  0.9141494435612083
```




## 5. Trim the trainning set
Since all classes have more than 500 images, the train_df dataset will be balanced and training time will be reduced


## 6. Create train, test and validation generators



## 7. Create a function to show Training Image Samples



## 8 Create a function that computes the F1 score metric 



## 9. Create the Model



## 10. Create a custom Keras callback to continue or halt training



## 11. Instantiate custom callback 



## 12. Train the model



## 13. Define a function to plot the training data



## 14. Make predictions on test set, create Confusion Matrix and Classification Report



## 15 Define a function to print list of misclassified test files



## 16 Save the model











