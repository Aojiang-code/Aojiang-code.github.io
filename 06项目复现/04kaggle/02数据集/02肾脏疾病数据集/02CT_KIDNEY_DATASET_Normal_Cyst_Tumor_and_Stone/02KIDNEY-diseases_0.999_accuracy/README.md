# KIDNEY-diseases 0.999 accuracy

## 0. 准备
下面这段代码导入了多个Python库，这些库在数据处理、图像处理和可视化方面非常有用。以下是对每行代码的详细中文注释：

```python
# 导入NumPy库，并使用别名np。NumPy是一个用于科学计算的Python库，提供了强大的多维数组对象和相关操作。
import numpy as np # 线性代数运算

# 导入Pandas库，并使用别名pd。Pandas是一个用于数据操作和分析的Python库，提供了DataFrame数据结构和CSV文件的读写功能。
import pandas as pd # 数据处理，CSV文件输入/输出（例如使用pd.read_csv）

# 导入Python的os模块，提供了与操作系统交互的功能，如文件路径操作和环境变量设置。
import os

# 导入OpenCV库，这是一个开源的计算机视觉和图像处理库。
import cv2

# 从pathlib模块导入Path类，用于对象化文件系统路径操作。
from pathlib import Path

# 导入Seaborn库，并使用别名sns。Seaborn是一个基于Matplotlib的高级数据可视化库，提供了更多样化的绘图风格和接口。
import seaborn as sns

# 导入Matplotlib的pyplot模块，并使用别名plt。Matplotlib是一个用于创建静态、动态和交互式图表的Python库。
import matplotlib.pyplot as plt

# 从skimage.io模块导入imread函数，用于读取图像文件。
from skimage.io import imread
```

执行这段代码后，Python环境中将可以使用上述库的功能。NumPy和Pandas在数据分析和预处理中非常有用；os模块在处理文件和目录时经常使用；OpenCV是进行图像处理和计算机视觉任务的常用库；Pathlib提供了面向对象的文件系统路径操作；Seaborn和Matplotlib用于创建丰富的数据可视化图表；skimage.io模块中的imread函数可以方便地读取不同格式的图像文件。这些库的组合为数据科学、机器学习和图像处理任务提供了强大的工具集。



下面这段代码使用`pathlib`模块来定义数据集的文件路径，并创建一个路径对象指向训练数据的目录。以下是对每行代码的详细中文注释：

```python
# 定义一个路径对象data_dir，指向数据集的顶层目录
# '../input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/'是相对于当前工作目录的路径
# 使用Pathlib库的Path类可以更方便地处理文件和目录路径
data_dir = Path('../input/ct-kidney-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/')

# 使用路径操作符/来创建一个新的路径对象train_dir，指向训练数据的子目录
# train_dir是基于data_dir的相对路径'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
# 这样，train_dir包含了完整的路径到训练数据集的目录
train_dir = data_dir / 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'

# 打印train_dir路径对象，展示构建的完整路径
# 这通常用于确认路径是否正确构建
train_dir
```

执行这段代码后，`train_dir`变量将包含训练数据集的完整路径。使用`pathlib`模块可以更加直观和方便地处理文件路径，特别是在进行文件操作和目录遍历时。这种对象化路径操作方式减少了对`os.path`模块的依赖，使得代码更加简洁易读。在Jupyter Notebook或其他Python环境中，打印路径对象可以直接显示路径字符串，而在脚本中，可以通过字符串格式化或其他方法来使用路径。


```python
PosixPath('../input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
```


执行上述代码后，得到的结果是`train_dir`路径对象的输出，它是一个`PosixPath`对象。以下是对结果的分析：

1. **PosixPath**:
   - `PosixPath`是`pathlib`模块中用于表示文件系统路径的类。在这个上下文中，它表示从根目录开始的完整文件路径。

2. **路径字符串**:
   - 路径字符串`'../input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'`显示了训练数据集所在的完整路径。
   - 路径中的`..`表示上一级目录，即从当前工作目录的上一级开始。
   - 接下来的部分`'input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/'`是数据集的子目录路径。
   - 最后的重复部分`'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'`是训练数据集的子目录名称。

这个结果表明`train_dir`变量已经成功地构建了指向训练数据集目录的路径。在实际应用中，这个路径对象可以用于访问和操作文件系统中的文件和目录，例如读取数据文件、保存模型结果等。需要注意的是，这里的路径是基于Unix-like系统的路径格式（使用正斜杠`/`作为路径分隔符），在Windows系统中可能需要使用反斜杠`\`。使用`pathlib`模块可以使得代码更加平台无关，因为它会自动处理不同操作系统间的路径差异。





下面这段代码使用`pathlib`和`glob`模块来处理图像数据集的文件路径，并将图像路径及其对应的标签存储在一个列表中，最后将这个列表转换为一个Pandas的DataFrame对象，并进行洗牌。以下是对每行代码的详细中文注释：

```python
# 获取正常、囊肿、结石和肿瘤子目录的路径
normal_cases_dir = train_dir / 'Normal'
Cyst_cases_dir = train_dir / 'Cyst'
Stone_cases_dir = train_dir / 'Stone'
Tumor_cases_dir = train_dir / 'Tumor'

# 使用glob方法获取各个子目录下所有.jpg格式的图像文件路径
normal_cases = normal_cases_dir.glob('*.jpg')
Cyst_cases = Cyst_cases_dir.glob('*.jpg')
Stone_cases = Stone_cases_dir.glob('*.jpg')
Tumor_cases = Tumor_cases_dir.glob('*.jpg')

# 初始化一个空列表，用于存储图像路径和标签
train_data = []

# 遍历Cyst_cases中的所有图像路径，并将它们的标签设置为0
for img in Cyst_cases:
    train_data.append((img, 0))

# 遍历normal_cases中的所有图像路径，并将它们的标签设置为1
for img in normal_cases:
    train_data.append((img, 1))

# 遍历Stone_cases中的所有图像路径，并将它们的标签设置为2
for img in Stone_cases:
    train_data.append((img, 2))

# 遍历Tumor_cases中的所有图像路径，并将它们的标签设置为3
for img in Tumor_cases:
    train_data.append((img, 3))

# 将train_data列表转换为Pandas的DataFrame对象，指定列名为'image'和'label'
train_data = pd.DataFrame(train_data, columns=['image', 'label'], index=None)

# 洗牌DataFrame，打乱数据的顺序
train_data = train_data.sample(frac=1.).reset_index(drop=True)

# 显示DataFrame的前几行，以检查其结构和内容
train_data.head()
```

执行这段代码后，将创建一个包含图像路径和对应标签的DataFrame，并对数据进行了随机洗牌。这种处理方式在机器学习中很常见，用于确保模型训练时数据的随机性，从而提高模型的泛化能力。在实际应用中，这个DataFrame可以用于进一步的数据加载和模型训练。需要注意的是，代码中的标签设置可能存在错误，因为通常正常情况的标签是0，而其他情况（如囊肿、结石、肿瘤）应该有其他不同的标签。此外，标签的设置应该根据实际的数据集和任务需求来确定。




![0.1head](01图片/0.1head.png)



下面这行代码使用Pandas库从`train_data` DataFrame中提取`'label'`列，并使用`unique()`函数找出该列中所有唯一的标签值。以下是对这行代码的详细中文注释：

```python
# 从DataFrame train_data中选择'label'列
# 'label'列包含了图像对应的标签信息
label_column = train_data['label']

# 使用unique()函数找出label_column中所有唯一的标签值
# 这通常用于了解数据集中有多少种不同的类别或标签
unique_labels = label_column.unique()
```

执行这段代码后，`unique_labels`变量将包含`train_data` DataFrame中`'label'`列的所有唯一值。这个结果有助于我们了解数据集中的类别分布，以及是否所有的类别都被正确地标记。在机器学习任务中，了解类别的唯一值对于设置分类模型的类别数或进行数据探索性分析是非常重要的。例如，如果我们在进行多类别分类任务，我们需要确保模型的输出层有与唯一标签数相匹配的神经元数量。


```python
array([0, 2, 1, 3])
```

执行上述代码后，得到的结果是`train_data` DataFrame中`'label'列的唯一标签值的数组。以下是对结果的分析：

1. **唯一标签数组**:
   - `array([0, 2, 1, 3])`显示了数据集中存在的四个唯一标签值。
   - 这些标签值可能代表了不同的类别或状况，例如正常、囊肿、结石和肿瘤等。

从这个结果可以看出，数据集中包含了四种不同的类别，每个类别都有一个唯一的标签值。在机器学习分类任务中，这意味着模型需要能够区分这四种不同的状况。标签值的顺序（0, 1, 2, 3）通常不重要，但在某些算法中，类别的顺序可能会影响结果的解释，例如在决策树或某些类型的聚类算法中。在使用这些标签进行模型训练时，需要确保模型的输出层或分类器配置正确地反映了类别的数量和顺序。此外，如果标签值不是连续的整数，可能需要考虑使用标签编码技术，如独热编码（One-Hot Encoding），以便更好地处理类别间的不平衡或确保模型正确地学习类别间的关系。



下面这行代码用于获取Pandas DataFrame `train_data`的维度信息。以下是对这行代码的详细中文注释：

```python
# 使用shape属性获取DataFrame train_data的行数和列数
# shape是一个元组，第一个元素表示DataFrame的行数（即样本数量），第二个元素表示列数
train_data.shape
```

执行这段代码后，将返回一个元组，其中包含两个整数值。第一个值表示`train_data`中的行数，即数据集中的样本数量；第二个值表示`train_data`中的列数，即特征数量加上标签列。这个信息对于了解数据集的规模和结构非常重要，特别是在进行数据分析和机器学习任务时，了解数据集的大小可以帮助我们决定适当的数据处理方法和模型选择。例如，如果样本数量很少，我们可能需要考虑使用更简单的模型或进行数据增强；如果特征数量很多，我们可能需要进行特征选择或降维处理。


```python
(12446, 2)
```


执行上述代码后，得到的结果是`train_data` DataFrame的维度信息。以下是对结果的分析：

1. **行数**:
   - 12446表示`train_data`中有12446行，即数据集中包含12446个样本。

2. **列数**:
   - 2表示`train_data`中有2列，这通常意味着DataFrame中除了一个特征列（通常是'image'列）外，还有一个'label'列，用于存储每个样本的标签。

从这个结果可以看出，数据集相对较大，包含超过一万个样本。这种规模的数据集通常足以训练一个机器学习模型，但模型的性能还取决于样本的质量和多样性。在进行模型训练之前，可能还需要进一步的数据探索和预处理，例如检查缺失值、进行数据清洗、特征工程等。此外，了解数据集的维度有助于我们选择合适的数据存储和处理工具，以及设置合适的机器学习模型参数。




下面这段代码首先使用Pandas库计算每个类别的样本数量，然后使用Seaborn和Matplotlib库创建一个条形图来可视化每个类别的样本计数。以下是对每行代码的详细中文注释：

```python
# 计算每个类别（标签）的样本数量
# train_data['label']获取DataFrame中的'label'列
# value_counts()方法返回每个唯一标签值的计数
cases_count = train_data['label'].value_counts()

# 打印每个类别的样本计数
print(cases_count)

# 创建一个新的图形对象，设置图形的大小为宽10英寸、高8英寸
plt.figure(figsize=(10,8))

# 使用Seaborn的barplot函数创建条形图
# x参数设置为cases_count.index，即类别标签
# y参数设置为cases_count.values，即对应的样本计数
sns.barplot(x=cases_count.index, y=cases_count.values)

# 设置图形的标题和坐标轴标签
# fontsize参数设置字体大小
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)

# 设置x轴的刻度标签，显示类别的描述
# range(len(cases_count.index))生成从0到类别数量减1的整数序列
# ['Cyst(0)','Normal(1)', 'Stone(2)', 'Tumor(3)']是类别标签的描述
plt.xticks(range(len(cases_count.index)), ['Cyst(0)','Normal(1)', 'Stone(2)', 'Tumor(3)'])

# 显示图形
plt.show()
```

执行这段代码后，将在控制台打印出每个类别的样本数量，并通过条形图直观地展示这些信息。条形图的x轴表示类别标签，y轴表示每个类别的样本数量。通过这个可视化，我们可以快速了解数据集中各个类别的样本分布情况，这对于评估数据集的平衡性和制定后续的数据处理策略非常重要。例如，如果某个类别的样本数量远多于其他类别，可能需要考虑数据增强或过采样/欠采样技术来平衡类别分布。此外，这种可视化也有助于我们理解数据集的特点，为后续的模型训练和评估提供有用的信息。


```python
1    5077
0    3709
3    2283
2    1377
Name: label, dtype: int64
```
执行上述代码后，得到的结果是`train_data` DataFrame中`'label'列的每个类别的样本数量计数。以下是对结果的分析：

1. **样本计数**:
   - `1    5077` 表示标签为1的类别有5077个样本。
   - `0    3709` 表示标签为0的类别有3709个样本。
   - `3    2283` 表示标签为3的类别有2283个样本。
   - `2    1377` 表示标签为2的类别有1377个样本。

2. **类别分布**:
   - 从计数结果可以看出，标签为3的类别（肿瘤）样本数量最多，有2283个样本。
   - 标签为0的类别（囊肿）样本数量最少，有1377个样本。
   - 标签为1的类别（正常）和标签为2的类别（结石）的样本数量分别位于中间，分别是5077和3709。

这个结果表明数据集中的类别分布是不均衡的，特别是标签为3的类别样本数量显著多于其他类别。在机器学习中，类别分布的不均衡可能会影响模型的性能，特别是当某些类别的样本数量远多于其他类别时。在这种情况下，模型可能会偏向于那些具有更多样本的类别。为了提高模型对较少样本类别的识别能力，可能需要采取一些策略，如过采样少数类别、欠采样多数类别或使用加权损失函数等。

此外，通过可视化工具（如上述代码中的条形图）可以直观地展示每个类别的样本数量，帮助我们更好地理解数据集的特点，并为后续的数据处理和模型训练提供指导。

![0.2](01图片/0.2.png)


下面这段代码首先从`train_data` DataFrame中提取每个类别的前5个样本的图像路径，然后将这些样本合并到一个列表中，并使用`matplotlib`和`skimage`库来显示这些图像样本。以下是对每行代码的详细中文注释：

```python
# 从DataFrame中提取标签为0（囊肿）的前5个样本的图像路径，并转换为列表
Cyst_samples = (train_data[train_data['label'] == 0]['image'].iloc[:5]).tolist()

# 从DataFrame中提取标签为1（正常）的前5个样本的图像路径，并转换为列表
Normal_samples = (train_data[train_data['label'] == 1]['image'].iloc[:5]).tolist()

# 从DataFrame中提取标签为2（结石）的前5个样本的图像路径，并转换为列表
Stone_samples = (train_data[train_data['label'] == 2]['image'].iloc[:5]).tolist()

# 从DataFrame中提取标签为3（肿瘤）的前5个样本的图像路径，并转换为列表
Tumor_samples = (train_data[train_data['label'] == 3]['image'].iloc[:5]).tolist()

# 将四个类别的样本列表合并成一个单一的样本列表
samples = Cyst_samples + Normal_samples + Stone_samples + Tumor_samples

# 删除之前创建的四个样本列表，释放内存
del Cyst_samples, Normal_samples, Stone_samples, Tumor_samples

# 创建一个4行5列的子图，设置图形的大小为宽30英寸、高30英寸
f, ax = plt.subplots(4, 5, figsize=(30, 30))

# 遍历样本列表，并在子图中显示每个图像样本
for i in range(20):
    # 使用skimage.io的imread函数读取图像
    img = imread(samples[i])
    # 在当前子图位置显示图像，使用灰度颜色映射
    ax[i//5, i%5].imshow(img, cmap='gray')
    # 根据当前子图的位置设置标题
    if i < 5:
        ax[i//5, i%5].set_title("Cyst_samples")
    elif i < 10:
        ax[i//5, i%5].set_title("Normal_samples")
    elif i < 15:
        ax[i//5, i%5].set_title("Stone_samples")
    elif i < 20:
        ax[i//5, i%5].set_title("Tumor_samples")
    
    # 关闭当前子图的坐标轴显示
    ax[i//5, i%5].axis('off')
    # 设置子图的长宽比为自动，保持图像的原始比例
    ax[i//5, i%5].set_aspect('auto')

# 显示图形
plt.show()
```

执行这段代码后，将在屏幕上显示一个包含20个子图的图形，每个子图显示一个类别的样本图像。这个可视化有助于我们快速了解数据集中不同类别的图像样本的外观和特点。通过观察样本图像，我们可以评估图像的质量、对比度、亮度等特征，这对于后续的图像预处理和特征提取工作非常重要。此外，这种可视化方法也有助于我们检查数据集中是否存在异常或错误的样本，以及类别标签是否正确分配。



![0.3](01图片/0.3.png)





## 1. loading train image using cv2

下面这段代码用于从不同的子目录中加载图像数据，对图像进行预处理，并将它们存储在列表中，最后将这些列表转换为NumPy数组。以下是对每行代码的详细中文注释：

```python
# 获取正常、囊肿、结石和肿瘤子目录的路径
normal_cases_dir = train_dir / 'Normal'
Cyst_cases_dir = train_dir / 'Cyst'
Stone_cases_dir = train_dir / 'Stone'
Tumor_cases_dir = train_dir / 'Tumor'

# 获取各个子目录下所有.jpg格式的图像文件路径
normal_cases = normal_cases_dir.glob('*.jpg')
Cyst_cases = Cyst_cases_dir.glob('*.jpg')
Stone_cases = Stone_cases_dir.glob('*.jpg')
Tumor_cases = Tumor_cases_dir.glob('*.jpg')

# 初始化两个空列表，用于存储图像数据和标签
train_data = []
train_labels = []

# 遍历囊肿案例的图像路径，读取并预处理图像
for img in Cyst_cases:
    # 使用cv2.imread读取图像
    img = cv2.imread(str(img))
    # 调整图像大小为28x28像素
    img = cv2.resize(img, (28, 28))
    # 如果图像是灰度图，则转换为三通道图像
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    # 将图像从BGR格式转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像转换为NumPy数组
    img = np.array(img)
    # 归一化图像像素值到[0, 1]范围
    img = img / 255
    # 设置标签为'Cyst'
    label = 'Cyst'
    # 将预处理后的图像和标签添加到列表中
    train_data.append(img)
    train_labels.append(label)

# 对正常案例的图像执行相同的预处理和标签操作
# ...
# 对结石案例的图像执行相同的预处理和标签操作
# ...
# 对肿瘤案例的图像执行相同的预处理和标签操作
# ...

# 将列表转换为NumPy数组
train_data1 = np.array(train_data)
train_labels1 = np.array(train_labels)

# 打印训练数据和标签的总数
print("Total number of validation examples: ", train_data1.shape)
print("Total number of labels:", train_labels1.shape)
```

执行这段代码后，`train_data1`和`train_labels1`将包含所有预处理后的图像数据和对应的标签。这里的错误在于，代码中的注释提到了“验证样本”，但实际上这些数据应该是用于训练的。此外，标签应该是数值型的，以便与神经网络的输出层相匹配。在实际应用中，可能需要使用`keras.utils.np_utils.to_categorical`函数将标签转换为独热编码形式，以便用于分类任务。此外，归一化操作通常在所有图像数据集上统一进行，而不是在循环中对每个图像单独进行。这样可以确保所有图像的预处理方式一致，并且归一化的范围相同。



```python
Total number of validation examples:  (12446, 28, 28, 3)
Total number of labels: (12446,)
```

执行上述代码后，得到的结果是关于训练数据集的维度信息。以下是对结果的分析：

1. **训练样本总数**:
   - `Total number of validation examples: (12446, 28, 28, 3)` 表示`train_data1`数组中有12446个训练样本。
   - 每个样本是一幅图像，图像的尺寸被调整为28x28像素，且具有3个颜色通道（RGB）。

2. **标签总数**:
   - `Total number of labels: (12446,)` 表示`train_labels1`数组中有12446个标签，与训练样本的数量相匹配。

从这个结果可以看出，数据集中包含了大量的图像样本，每个样本都被预处理成了统一的尺寸和格式。图像数据具有三个颜色通道，这可能是因为原始图像是彩色的，或者是将灰度图像通过`np.dstack`函数复制到三个通道以模拟RGB图像。标签数组是一个一维数组，每个元素对应一个图像样本的类别标签。

需要注意的是，代码中的注释提到了“验证样本”，但实际上这些数据应该是用于训练的。此外，标签应该是数值型的，以便与神经网络的输出层相匹配。在实际应用中，可能需要使用`keras.utils.np_utils.to_categorical`函数将标签转换为独热编码形式，以便用于分类任务。此外，归一化操作已经在这里完成，每个像素值都被除以255以得到[0, 1]范围内的值，这是准备图像数据用于深度学习模型的常见步骤。最后，这个结果表明我们已经成功地从原始图像文件路径创建了一个结构化的NumPy数组，可以用于后续的模型训练。






下面这行代码用于获取`train_data1` NumPy数组的形状（维度）。以下是对这行代码的详细中文注释：

```python
# 获取train_data1 NumPy数组的形状（行数和列数）
# shape属性返回一个元组，其中包含数组的维度信息
# 对于图像数据，通常形状为(样本数量, 高度, 宽度, 通道数)
train_data1.shape
```

执行这段代码后，将返回一个元组，包含`train_data1`数组的维度。对于图像数据集，这个形状通常表示为(样本数量, 图像高度, 图像宽度, 颜色通道数)。例如，如果返回的形状是(12446, 28, 28, 3)，这意味着数组包含12446个图像样本，每个图像是28x28像素大小，并且有3个颜色通道（例如RGB）。这个信息对于理解数据集的结构和准备数据以供机器学习模型使用非常重要。通过知道数组的形状，我们可以确保数据加载和预处理的正确性，以及为模型输入层配置适当的参数。



```python
(12446, 28, 28, 3)
```






下面






## 2. Train output file convert list to csv file










## 3. Solving image dataset imbalance using SMOTE










## 4. Augmentation












## 5. Create the model








## 6. Testing the a image with sample data
















