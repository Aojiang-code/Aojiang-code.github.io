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





下面




![0.1head](01图片/0.1head.png)



下面




## 1. loading train image using cv2











## 2. Train output file convert list to csv file










## 3. Solving image dataset imbalance using SMOTE










## 4. Augmentation












## 5. Create the model








## 6. Testing the a image with sample data
















