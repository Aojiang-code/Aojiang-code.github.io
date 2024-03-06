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
##### 第一步：建立类别标签，不同类别对应不同的数字
```python
# 初始化一个空列表来存储特征数据。
feature = []

# 初始化一个空列表来存储标签数据。
label = []

# 创建一个字典来映射类别名称到对应的数字标签。
# 这个字典用于将文本类别标签转换为模型可以处理的数值标签。
label_dict = {
    'aloe': 0,        # 芦荟对应的数字标签是0
    'burger': 1,      # 汉堡对应的数字标签是1
    'cabbage': 2,     # 卷心菜对应的数字标签是2
    'candied_fruits': 3, # 蜜饯水果对应的数字标签是3
    'carrots': 4,     # 胡萝卜对应的数字标签是4
    'chips': 5,       # 薯片对应的数字标签是5
    'chocolate': 6,  # 巧克力对应的数字标签是6
    'drinks': 7,      # 饮料对应的数字标签是7
    'fries': 8,       # 薯条对应的数字标签是8
    'grapes': 9,      # 葡萄对应的数字标签是9
    'gummies': 10,    # 软糖对应的数字标签是10
    'ice-cream': 11,  # 冰淇淋对应的数字标签是11
    'jelly': 12,       # 果冻对应的数字标签是12
    'noodles': 13,    # 面条对应的数字标签是13
    'pickles': 14,   # 泡菜对应的数字标签是14
    'pizza': 15,      # 披萨对应的数字标签是15
    'ribs': 16,       # 排骨对应的数字标签是16
    'salmon': 17,     # 三文鱼对应的数字标签是17
    'soup': 18,       # 汤对应的数字标签是18
    'wings': 19       # 鸡翅对应的数字标签是19
}

# 创建一个反向字典，用于将数字标签映射回原始的类别名称。
# 这在需要将模型的输出转换回可读的类别名称时非常有用。
label_dict_inv = {v: k for k, v in label_dict.items()}
```

这段代码定义了两个列表`feature`和`label`，用于存储特征数据和标签数据。然后，创建了一个名为`label_dict`的字典，它将食品类别名称映射到从0到19的整数。这样的映射允许模型使用数值标签进行训练，这对于大多数机器学习算法来说是必要的。最后，创建了一个反向字典`label_dict_inv`，它允许我们将模型的数值输出转换回原始的食品类别名称。
###### 详细介绍
上述代码段是一个Python脚本的一部分，它用于准备和处理数据，以便在机器学习模型中使用。这个脚本的主要目的是将食品的类别名称转换为模型可以识别的数值标签，并创建一个反向映射，以便在模型预测后将数值标签转换回原始的类别名称。下面是对这段代码的详细介绍：

1. 初始化特征和标签列表：
   ```python
   feature = []  # 这个列表将用于存储每个样本的特征数据。
   label = []    # 这个列表将用于存储每个样本的标签数据。
   ```
   这两个列表是空的，它们将在数据预处理阶段被填充。

2. 创建类别标签映射字典：
   ```python
   label_dict = {
       # ... 类别名称和对应的数字标签 ...
   }
   ```
   `label_dict`是一个字典，它将食品的类别名称作为键（key），将对应的数字标签作为值（value）。这种映射是必要的，因为大多数机器学习模型需要数值输入。在这个例子中，每个食品类别都被分配了一个唯一的整数标签，从0到19。

3. 创建反向字典：
   ```python
   label_dict_inv = {v: k for k, v in label_dict.items()}
   ```
   `label_dict_inv`是`label_dict`的反向字典。它将数字标签映射回原始的类别名称。这在模型预测阶段非常有用，因为模型通常会输出数值标签，而我们可能需要知道这些数值标签对应的具体食品类别名称。通过这个反向字典，我们可以轻松地将模型的输出转换为人类可读的形式。

这个脚本的目的是为了数据预处理，它为后续的模型训练和评估提供了必要的数据结构。在实际应用中，你需要填充`feature`和`label`列表，然后使用`label_dict`来编码你的数据集，最后在模型训练和预测时使用这些字典来转换标签。


##### 第二步：定义并调用sox_wavs函数，转换音频并将音频移动到新目录
```python
# 设置新的父目录路径，用于存放处理后的音频文件。
new_parent_dir = './soxed_train/'

# 设置原始的父目录路径，原始音频文件存放在这里。
old_parent_dir = './train/'

# 设置保存处理后音频文件的目录。
save_dir = "./"

# 创建一个包含所有子目录名称的NumPy数组，这些子目录对应不同的食品类别。
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

# 定义一个函数，用于处理音频文件，将它们从原始目录移动到新目录，并使用sox工具进行处理。
def sox_wavs(new_parent_dir, old_parent_dir, sub_dirs, max_file=1, file_ext="*.wav"):

    # 遍历所有子目录。
    for sub_dir in sub_dirs:
        # 如果新目录不存在，则创建它。
        if not os.path.exists(os.path.join(new_parent_dir, sub_dir)):
            os.makedirs(os.path.join(new_parent_dir, sub_dir))
        # 使用glob模块查找原始目录下指定子目录中的所有.wav文件。
        # tqdm是一个进度条库，用于显示进度。
        # glob.glob用于查找匹配特定模式的文件路径。
        # [:max_file]限制处理的文件数量。
        for fn in tqdm(glob.glob(os.path.join(old_parent_dir, sub_dir, file_ext))[:max_file]): 
            # 构造新的文件路径，将原始目录替换为新目录。
            new_fn = fn.replace(old_parent_dir, new_parent_dir)
            # 使用os.system调用sox命令行工具，对音频文件进行处理。
            # sox是一个音频处理工具，这里用于将音频文件转换为16位有符号整数格式。
            os.system(f'sox -b 16 -e signed-integer {fn} {new_fn}')

# 调用sox_wavs函数，处理最多10000个.wav文件。
sox_wavs(new_parent_dir, old_parent_dir, sub_dirs, max_file=10000, file_ext="*.wav")
```

这段代码的目的是对一组音频文件进行处理，将它们从原始目录移动到新目录，并使用`sox`这个音频处理工具将音频文件转换为16位有符号整数格式。这通常是为了在机器学习模型中使用音频数据时，确保数据的一致性和兼容性。代码中使用了`glob`模块来查找文件，`tqdm`库来显示进度条，以及`os.system`来执行系统命令。在实际应用中，你需要确保`sox`工具已经安装在你的系统上，并且`glob`和`tqdm`库也被安装。此外，你可能需要根据实际情况调整`max_file`参数，以控制处理的文件数量。
###### 详细介绍
上述代码段是一个Python脚本，用于处理音频文件，具体来说，是将一组.wav格式的音频文件从原始目录移动到新的目录，并使用`sox`工具对这些文件进行格式转换。这个过程通常在准备数据集以供机器学习模型训练时使用。下面是对这段代码的详细介绍：

1. 设置目录路径：
   - `new_parent_dir`：新的父目录路径，用于存放经过处理的音频文件。
   - `old_parent_dir`：原始的父目录路径，原始的.wav音频文件存放在这里。
   - `save_dir`：保存处理后音频文件的目录，这里设置为当前目录（`.`）。

2. 创建子目录列表：
   - `folds` 和 `sub_dirs` 是一个NumPy数组，包含了所有子目录的名称。这些子目录对应不同的食品类别，每个类别的音频文件将被处理并保存在相应的子目录下。

3. 定义处理音频文件的函数：
   - `sox_wavs` 函数接受新旧目录路径、子目录列表、最大文件数量和文件扩展名作为参数。
   - 函数内部，首先遍历所有子目录。
   - 对于每个子目录，如果新目录不存在，则创建它。
   - 使用`glob.glob`查找每个子目录下的所有.wav文件，并通过`tqdm`显示进度条。
   - 对于每个找到的文件，构造新的文件路径，将原始目录替换为新目录。
   - 使用`os.system`调用`sox`命令行工具，对每个音频文件进行处理。这里的`sox`命令将音频文件转换为16位有符号整数格式。

4. 调用函数处理音频文件：
   - 最后，调用`sox_wavs`函数，设置最大处理文件数量为10000个，文件扩展名为`.wav`。

在实际应用中，这段代码需要在已经安装了`sox`、`glob`和`tqdm`库的环境中运行。`sox`是一个强大的音频处理工具，它支持多种音频格式的转换和处理。`glob`模块用于文件路径模式匹配，而`tqdm`库用于显示进度条，使得长时间运行的进程更加用户友好。通过这个脚本，你可以批量处理音频文件，为后续的数据分析或机器学习任务做好准备。


##### 第三步：
```python
# 定义一个函数，用于从音频文件中抽取梅尔频率倒谱系数（MFCC）特征。
def extract_mfcc(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    # 初始化标签和特征的空列表。
    label, feature = [], []
    
    # 遍历所有子目录。
    for sub_dir in sub_dirs:
        # 使用glob模块查找指定子目录下的所有.wav文件，最多处理max_file个文件。
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]): 
            # 从文件路径中提取子目录名称作为标签名称。
            label_name = fn.split('\\')[-2]
            # 将标签名称转换为对应的数字标签，并添加到标签列表中。
            label.extend([label_dict[label_name]])
            # 使用librosa库加载音频文件。
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            # 使用librosa库计算MFCC特征。
            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T 
            # 将MFCC特征添加到特征列表中。转置操作是为了将特征向量的形状从(n_mfcc, T)转换为(T, n_mfcc)。
            feature.append(mfcc)
    
    # 返回特征列表和标签列表。
    return [feature, label]

# 这个函数的目的是为每个音频文件计算MFCC特征，并将它们与对应的标签关联起来。
# 这样，我们就可以准备数据集，用于机器学习模型的训练或评估。
```

这段代码定义了一个名为`extract_mfcc`的函数，它用于从一组.wav格式的音频文件中提取MFCC特征。MFCC是一种常用的音频特征，它在语音和音乐处理中非常有用，因为它能够捕捉到音频信号的重要特性。函数首先初始化两个空列表，用于存储特征和标签。然后，它遍历每个子目录中的音频文件，使用`librosa`库加载音频数据，并计算MFCC特征。每个文件的MFCC特征和对应的标签都被添加到列表中。最后，函数返回这两个列表，它们可以用于训练机器学习模型。在实际应用中，你需要确保`librosa`库已经安装在你的系统上。
###### 详细介绍
上述代码段定义了一个名为 `extract_mfcc` 的Python函数，该函数的目的是从一组音频文件中提取梅尔频率倒谱系数（MFCC）特征。MFCC是一种在语音和音频处理领域广泛使用的特征，它能够捕捉音频信号的频谱包络，对于音频分类、识别等任务非常有用。下面是对这段代码的详细介绍：

1. 初始化标签和特征列表：
   ```python
   label, feature = [], []
   ```
   这里创建了两个空列表，`label` 用于存储每个音频文件的标签，`feature` 用于存储每个文件的MFCC特征。

2. 遍历子目录和文件：
   ```python
   for sub_dir in sub_dirs:
       for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]): 
   ```
   这段代码遍历了所有子目录（`sub_dirs`），并在每个子目录中查找所有匹配特定文件扩展名（默认为 `.wav`）的文件。`max_file` 参数限制了处理的文件数量。`tqdm` 库用于显示进度条，使得长时间运行的进程更加用户友好。

3. 提取标签名称并转换为数字标签：
   ```python
   label_name = fn.split('\\')[-2]
   label.extend([label_dict[label_name]])
   ```
   从文件路径中提取出子目录名称作为标签名称，然后使用之前定义的 `label_dict` 字典将标签名称转换为对应的数字标签，并添加到 `label` 列表中。

4. 加载音频文件并计算MFCC特征：
   ```python
   X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
   mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T
   ```
   使用 `librosa.load` 函数加载音频文件，获取音频波形和采样率。然后，使用 `librosa.feature.mfcc` 函数计算MFCC特征。`n_mfcc=128` 参数指定了MFCC的系数数量。`.T` 转置操作用于调整MFCC特征矩阵的形状，使其更适合后续处理。

5. 添加MFCC特征到特征列表：
   ```python
   feature.append(mfcc)
   ```
   将计算出的MFCC特征添加到 `feature` 列表中。

6. 返回特征和标签列表：
   ```python
   return [feature, label]
   ```
   函数执行完毕后，返回包含所有MFCC特征的列表和对应的标签列表。

这个函数可以用于准备机器学习模型的训练数据。在实际应用中，你需要确保已经安装了 `librosa` 和 `glob` 库，并且 `label_dict` 字典已经根据你的数据集定义好。此外，你可能需要根据实际情况调整 `max_file` 参数，以控制处理的文件数量。

##### 第四步：
```python
# 设置音频文件的父目录路径，这里假设音频文件已经被sox处理过。
parent_dir = './clips_rd_sox/'

# 设置保存处理后数据的目录，这里使用当前目录。
save_dir = "./"

# 创建一个包含所有子目录名称的NumPy数组，这些子目录对应不同的食品类别。
# 这些类别将用于提取对应的MFCC特征。
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

# 调用之前定义的extract_mfcc函数，从指定的目录中提取MFCC特征。
# 这里最多处理1000个文件。
mfcc_128_all, label = extract_mfcc(parent_dir, sub_dirs, max_file=1000)

# 作者在这里讨论了使用梅尔频谱或梅尔倒谱对模型性能的影响。
# 作者提到，使用128个梅尔滤波器相比于baseline模型中的20个滤波器，效果有所提升。
# sox是一个在Linux中用于处理音频的工具，它可以用于规整音频文件的频率。
# 作者提到，尽管原始数据流程是标准的，但在LSTM的baseline模型中使用的数据似乎进行了某种增强。
# 作者选择直接使用这些已经增强的数据，这可能是一个提高模型性能的技巧。
# 最后一句是作者的幽默表达，暗示这种技巧可能是他提前退休的原因。
```

这段代码的目的是从一个指定的目录中提取音频文件的MFCC特征，并保存标签信息。作者在这里提到了一些关于音频处理和模型训练的经验和观察，包括使用梅尔滤波器的数量对模型性能的影响，以及数据增强可能带来的性能提升。最后，作者以一种幽默的方式表达了对这些技巧的看法。在实际应用中，这段代码需要在已经安装了 `librosa`、`glob` 和 `numpy` 库的环境中运行。
###### 详细介绍
上述代码段是一个Python脚本的一部分，它用于在音频处理和机器学习任务中提取梅尔频率倒谱系数（MFCC）特征。这段代码的目的是为音频文件准备数据集，以便后续的模型训练。下面是对这段代码的详细介绍：

1. 设置目录路径：
   - `parent_dir`：这是音频文件所在的父目录路径。在这个例子中，假设音频文件已经被`sox`工具处理过，这意味着它们可能已经被转换为特定的格式或采样率。
   - `save_dir`：这是保存处理后数据的目录。在这个例子中，数据将被保存在当前工作目录。

2. 定义子目录列表：
   - `folds` 和 `sub_dirs` 是一个NumPy数组，包含了所有子目录的名称。这些子目录代表了不同的类别，例如食品名称。在音频处理任务中，这可能意味着每个子目录包含了对应类别的音频文件。

3. 调用`extract_mfcc`函数：
   - 这个函数之前已经定义过，它用于从音频文件中提取MFCC特征。在这个调用中，我们指定了最多处理1000个文件（`max_file=1000`）。
   - 函数执行后，返回两个列表：`mfcc_128_all`（包含所有音频文件的MFCC特征）和`label`（包含对应的标签）。

4. 讨论梅尔滤波器的数量对模型性能的影响：
   - 作者提到，使用128个梅尔滤波器相比于baseline模型中的20个滤波器，模型性能有所提升。梅尔滤波器是MFCC特征提取过程中的一个关键步骤，它们用于模拟人耳对不同频率的响应。

5. 讨论`sox`工具：
   - `sox`是一个在Linux中广泛使用的音频处理工具。它可以用于执行各种音频操作，如格式转换、采样率调整等。在这个上下文中，`sox`可能被用来预处理音频文件，以便于后续的MFCC特征提取。

6. 数据增强的讨论：
   - 作者提到，在LSTM的baseline模型中使用的数据可能进行了某种增强。数据增强是一种常用的技术，通过在原始数据上应用各种变换（如时间拉伸、音调变化等），来增加数据集的多样性和模型的泛化能力。

7. 使用增强数据的决策：
   - 作者选择了直接使用已经增强的数据，这可能是为了提高模型性能的一个技巧。在机器学习中，数据增强通常被视为一种有效的策略，尤其是在数据量有限的情况下。

8. 幽默的结尾：
   - 最后一句是作者的幽默表达，暗示使用这些技巧可能是他提前退休的原因。这可能是在轻松地表达，通过这些技术改进，模型的性能已经足够好，以至于不再需要作者的进一步工作。

在实际应用中，这段代码需要在已经安装了`librosa`、`glob`、`numpy`和`tqdm`库的环境中运行。此外，确保`label_dict`字典已经根据你的数据集定义好，以便正确地将类别名称映射到数字标签。


### 模型构建
#### CNN
```python
# 设置交叉验证的折数为5折。
nfold = 5
# 创建KFold对象，用于进行K折交叉验证。设置随机种子为2020。
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)

# 初始化一个数组来存储所有折的预测结果的平均值。
prediction1 = np.zeros((2000, 20))

# 初始化一个计数器，用于跟踪当前的折数。
i = 0

# 使用KFold进行5折交叉验证。
for train_index, valid_index in kf.split(X, Y):
    print("\nFold {}".format(i + 1))  # 打印当前的折数。
    # 分割训练集和验证集的索引。
    train_x, val_x = X[train_index], X[valid_index]
    train_y, val_y = Y[train_index], Y[valid_index]
    
    # 调整训练集和验证集的形状以适应CNN模型的输入要求。
    train_x = train_x.reshape(-1, 16, 8, 1)
    val_x = val_x.reshape(-1, 16, 8, 1)
    
    # 将标签转换为独热编码（one-hot encoding）格式。
    train_y = to_categorical(train_y)
    val_y = to_categorical(val_y)
    
    # 创建一个Sequential模型，用于构建卷积神经网络（CNN）。
    model = Sequential()
    
    # 添加卷积层、池化层、Dropout层、全连接层和输出层。
    # 添加第一个卷积层，使用64个5x5的卷积核，padding设置为"same"以保持输出尺寸不变，激活函数使用ReLU。
    model.add(Convolution2D(64, (5, 5), padding="same", input_shape=input_dim, activation='relu'))

    # 添加第一个池化层，使用2x2的池化窗口，用于降低特征图的空间尺寸。
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 添加第二个卷积层，使用128个3x3的卷积核，同样设置padding为"same"，激活函数使用ReLU。
    model.add(Convolution2D(128, (3, 3), padding="same", activation='relu'))
    
    # 添加第二个池化层，同样使用2x2的池化窗口，进一步降低特征图的空间尺寸。
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 添加Dropout层，丢弃率为0.1，有助于防止过拟合。
    model.add(Dropout(0.1))
    
    # 添加Flatten层，将多维的特征图展平成一维，为全连接层做准备。
    model.add(Flatten())
    
    # 添加第一个全连接层，有1024个神经元，激活函数使用ReLU。
    model.add(Dense(1024, activation='relu'))
    
    # 添加第二个全连接层，有100个神经元，激活函数使用ReLU。
    model.add(Dense(100, activation='relu'))
    
    # 添加输出层，有20个神经元，对应20个类别，激活函数使用softmax，用于多分类任务。
    model.add(Dense(20, activation='softmax'))
    #上面这段代码是在Keras中构建一个卷积神经网络模型的过程。模型由多个层组成，包括卷积层、池化层、Dropout层、全连接层和输出层。每一层都有其特定的作用，例如卷积层用于提取图像特征，池化层用于降低特征的空间尺寸，Dropout层用于减少过拟合，全连接层用于学习特征之间的复杂关系，输出层用于根据学习到的特征进行分类。在实际应用中，这些层的参数（如卷积核的数量、大小、全连接层的神经元数量等）需要根据具体任务进行调整。

    
    # 编译模型，使用Adam优化器和分类交叉熵损失函数。
    model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    # 打印模型结构。
    model.summary()
    
    # 训练模型，使用训练集数据和验证集数据。
    history = model.fit(train_x, train_y, epochs=120, batch_size=128, validation_data=(val_x, val_y))
    
    # 使用训练好的模型对测试集进行预测。
    predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
    print(predictions.shape)  # 打印预测结果的形状。
    
    # 计算所有折的平均预测结果。
    prediction1 += (predictions) / nfold
    
    # 更新计数器。
    i += 1

# 作者在这里提到，使用规整好的数据进行5折交叉验证已经达到了初赛的分数。
# 因此，作者没有继续进行更多的模型训练，而是转向了其他任务，如心跳分类。
# 作者还提到，为了备战复习数学，没有继续使用之前为心跳分类编写的模型。
# 下面，作者可能会介绍其他类型的模型。
```

这段代码展示了如何使用Keras构建和训练一个卷积神经网络（CNN）模型。它使用了K折交叉验证来评估模型的性能，并计算了所有折的平均预测结果。作者提到，使用这种方法已经达到了初赛的分数，因此没有继续进行更多的模型训练。在实际应用中，这段代码需要在已经安装了Keras和相关库的环境中运行，并且确保`X`、`Y`和`X_test`等变量已经被正确定义。
##### 详细介绍
上述代码段是一个使用Python编写的脚本，它利用Keras库构建和训练一个卷积神经网络（CNN）模型，并通过K折交叉验证（K-Fold Cross-Validation）来评估模型的性能。这个过程中，数据被分成多个部分，每个部分轮流作为验证集，其余部分作为训练集。这样做可以更好地评估模型的泛化能力。下面是对这段代码的详细介绍：

1. 设置交叉验证的折数为5折，并创建一个KFold对象。`shuffle=True`表示在分割之前会随机打乱数据，`random_state=2020`设置了随机数种子，以确保结果的可重复性。

2. 初始化一个名为`prediction1`的NumPy数组，用于存储所有折的预测结果的平均值。这个数组的形状是`(2000, 20)`，假设有2000个样本和20个类别。

3. 初始化一个计数器`i`，用于跟踪当前的折数。

4. 使用`kf.split(X, Y)`进行5折交叉验证。在每次迭代中，`train_index`和`valid_index`分别代表当前折的训练集和验证集的索引。

5. 根据索引分割训练集和验证集的数据。`X`和`Y`分别是特征数据和标签数据。

6. 调整训练集和验证集的形状，以适应CNN模型的输入要求。这里假设输入数据的形状是`(16, 8, 1)`，这可能是经过预处理后的图像数据。

7. 使用`to_categorical`函数将标签转换为独热编码（one-hot encoding）格式。这是多分类问题中常用的标签格式。

8. 创建一个Sequential模型，并添加多个层，包括卷积层（`Convolution2D`）、池化层（`MaxPooling2D`）、Dropout层、全连接层（`Dense`）和输出层。这里使用了ReLU激活函数和softmax输出层，适合于多分类问题。

9. 编译模型，指定Adam优化器和分类交叉熵损失函数（`categorical_crossentropy`）。

10. 打印模型结构，以便查看模型的层和参数。

11. 使用`model.fit`训练模型，指定训练集数据、验证集数据、训练轮数（`epochs`）和批量大小（`batch_size`）。

12. 使用训练好的模型对测试集`X_test`进行预测。这里假设`X_test`已经是一个NumPy数组，且需要调整形状以匹配模型的输入要求。

13. 计算所有折的平均预测结果，并将结果存储在`prediction1`数组中。

14. 更新计数器`i`，准备进行下一折的训练和验证。

作者在代码的最后提到，使用这种方法已经达到了初赛的分数，因此没有继续进行更多的模型训练。作者转向了其他任务，如心跳分类，并提到为了备战复习数学，没有继续使用之前为心跳分类编写的模型。这表明作者可能在准备某个比赛或考试，并且已经对当前的模型性能感到满意。在实际应用中，这段代码需要在已经安装了Keras和相关库的环境中运行，并且确保所有必要的数据和变量都已经准备好。


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
##### 详细介绍







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
##### 详细介绍



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


