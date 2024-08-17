# 01Intermediate_ML_Techniques_for_Detection_of_Cancer注释


这本笔记本旨在开发一种用于检测卵巢癌的机器学习模型。本项目的主要目标是利用各种机器学习方法。代码被组织成几个部分。

## 数据整理
数据整理部分对数据集进行预处理，将原始数据转换成适合训练机器学习模型的清洁数据集。

## 初步数据分析
初步数据分析部分探索所提供的数据集并执行基本的统计分析，以更好地理解数据。

## 特征工程
特征工程部分从数据中提取相关特征，以提高模型的准确性。

## 模型训练
模型训练部分使用几种集成学习算法训练机器学习模型，并根据各种指标评估它们的性能。

## 集成方法分析
集成方法分析部分分析了对所选集成学习算法性能有贡献的因素。

此外，代码还包括一个可解释的人工智能部分，该部分使用各种方法来解释机器学习模型并提供对其决策过程的洞察。


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# 这个Python 3环境预装了许多有用的分析库

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python 
# 该环境由kaggle/python Docker镜像定义

# For example, here's several helpful packages to load
# 例如，下面是一些有用的包加载示例

import numpy as np # linear algebra
# 导入numpy库，用于进行线性代数运算

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# 导入pandas库，用于数据处理，CSV文件的读取与写入等

import seaborn as sns
# 导入seaborn库，用于数据可视化

import matplotlib.pyplot as plt
# 导入matplotlib.pyplot，用于创建图形和绘图

# Input data files are available in the read-only "../input/" directory
# 输入数据文件可在只读的“../input/”目录下找到

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# 例如，运行这段代码（通过点击运行或按Shift+Enter）将列出输入目录下的所有文件

import os
# 导入os库，用于与操作系统交互

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# 使用os.walk遍历“/kaggle/input”目录，打印出所有文件的完整路径

!pip install mrmr_selection
# 使用pip安装`mrmr_selection`库，该库用于特征选择
```

这段代码加载了多个用于数据处理和可视化的Python库，并遍历了Kaggle指定输入目录下的所有文件。同时，它还通过pip安装了一个额外的库`mrmr_selection`。



```python
# Pandas display options for easy viewing of dataframes
# 设置Pandas显示选项，以便更容易查看数据框

pd.set_option('display.width', 150)
# 设置Pandas的显示宽度选项，将数据框的显示宽度设置为150个字符
# 这样可以避免数据框内容换行，方便在终端或Jupyter Notebook中查看宽数据框
```


```python
# Reading the dataset
# 读取数据集

cancer_data = pd.read_excel('/kaggle/input/predict-ovarian-cancer/Supplementary data 1.xlsx', sheet_name='All Raw Data', dtype=str)
# 使用pandas的read_excel函数读取Excel文件中的数据集
# 文件路径为 '/kaggle/input/predict-ovarian-cancer/Supplementary data 1.xlsx'
# 读取的工作表名称为 'All Raw Data'
# 将所有数据读取为字符串类型（dtype=str）

# Remove trailing whitespace from all string columns
# 删除所有字符串列中的尾随空格

cancer_data = cancer_data.apply(lambda x: x.str.rstrip() if x.dtype == "object" else x)
# 使用apply函数遍历数据框中的每一列
# 如果列的数据类型是字符串（object），则使用rstrip方法删除字符串中的尾随空格
# 对非字符串列，保持原样不作处理
```



```python
# Printing the first 5 rows of the data
# 打印数据集的前5行

print(cancer_data.head())
# 使用print函数输出数据集的前5行数据
# head()函数默认返回数据集的前5行，便于快速查看数据的基本结构和内容
```


```python
# Printing the shape of the data
# 打印数据集的形状（即行数和列数）

print(cancer_data.shape)
# 使用print函数输出数据集的形状
# shape属性返回一个元组，表示数据集的维度，其中第一个值是行数，第二个值是列数
```

### 结果解读
输出的结果是 `(349, 51)`，这意味着数据集中有 349 行和 51 列。具体来说：
- **349 行** 表示数据集中有 349 个样本或观测值。
- **51 列** 表示每个样本有 51 个特征或变量。


```python
# Printing the summary statistics of the data
# 打印数据集的摘要统计信息

print(cancer_data.info())
# 使用print函数输出数据集的info信息
# info()方法提供有关DataFrame的基本信息，包括数据类型、非空值计数等
```

### 结果解读
输出结果是关于数据集的摘要信息。以下是解读：

- **数据集类型**：`<class 'pandas.core.frame.DataFrame'>` 表示数据集是一个Pandas DataFrame。
  
- **行数**：`RangeIndex: 349 entries, 0 to 348` 表示数据集共有349行，索引范围从0到348。

- **列数**：`Data columns (total 51 columns):` 表示数据集共有51列。

- **列的信息**：每一列都有以下信息：
  - **列索引和名称**：如 `0   SUBJECT_ID` 表示这是第0列，列名是`SUBJECT_ID`。
  - **非空值计数**：如 `349 non-null` 表示这一列有349个非空值。
  - **数据类型**：如 `object` 表示这一列的数据类型为字符串或混合类型。

- **列数据类型**：`dtypes: object(51)` 表示所有51列的类型都是`object`，通常表示文本数据。

- **内存使用**：`memory usage: 139.2+ KB` 表示整个DataFrame占用的内存大小为139.2 KB。

从这个摘要中可以看出，数据集中有51列，每列的数据类型都是`object`，而且某些列存在缺失值。



## 数据整理
本部分代码负责通过清理、转换和重构数据，将数据集准备好以供分析使用。本部分涉及处理缺失数据、处理异常值以及转换变量，以确保它们符合分析方法的假设。目标是创建一个可靠的数据集，在使用机器学习算法时最大限度地提高准确性。数据整理是数据分析过程中的关键步骤，因为结果的准确性在很大程度上依赖于所使用数据集的质量。


```python
# 在转换列数据类型之前，首先处理数据中的不一致值

# 如果AFP列的值为'>1210.00'，将其替换为'1210.00'
cancer_data.loc[cancer_data['AFP'] == '>1210.00', 'AFP'] = '1210.00'

# 如果AFP列的值为'>1210'，将其替换为'1210.00'
cancer_data.loc[cancer_data['AFP'] == '>1210', 'AFP'] = '1210.00'

# 如果CA125列的值为'>5000.00'，将其替换为'5000.00'
cancer_data.loc[cancer_data['CA125'] == '>5000.00', 'CA125'] = '5000.00'

# 如果CA19-9列的值为'>1000.00'，将其替换为'1000.00'
cancer_data.loc[cancer_data['CA19-9'] == '>1000.00', 'CA19-9'] = '1000.00'

# 如果CA19-9列的值为'>1000'，将其替换为'1000.00'
cancer_data.loc[cancer_data['CA19-9'] == '>1000', 'CA19-9'] = '1000.00'

# 如果CA19-9列的值为'<0.600'，将其替换为'0.5'
cancer_data.loc[cancer_data['CA19-9'] == '<0.600', 'CA19-9'] = '0.5'
```



```python
# 将对象类型的列转换为浮点型列
# 遍历所有列（除了'TYPE'列），如果列的数据类型为object，则将其转换为float类型
for col in cancer_data.drop('TYPE', axis=1).select_dtypes(include=['object']).columns:
    cancer_data[col] = cancer_data[col].astype('float')

# 将目标列'TYPE'转换为整数类型
# 将'TYPE'列的数据类型转换为64位整数（int64）
cancer_data['TYPE'] = cancer_data['TYPE'].astype('int64')
```


### 逐行代码注释
```python
# 计算每一列中缺失数据的比例
# 使用isnull()方法获取布尔值表示的数据框，mean()方法计算布尔值的平均值，即为缺失值比例
missing_ratio = cancer_data.isnull().mean()

# 显示每一列中缺失数据的比例
print(missing_ratio)
```

### 结果解读
结果显示了每个列中缺失数据的比例，其中比例的范围在 `0.000000` 到 `0.687679` 之间。

- **SUBJECT_ID** 列没有缺失值（`0.000000`）。
- **AFP** 列有约 `6.30%` 的数据缺失。
- **CA72-4** 列缺失数据比例最高，约为 `68.77%`，这表明大部分数据都缺失。
- **CA19-9** 列缺失数据比例为 `6.88%`。
- **NEU** 列的缺失比例为 `26.07%`，相对较高。

总体来看，大多数列的缺失数据比例都在 `0.00%` 到 `6.88%` 之间，只有少数几列缺失比例较高，例如 **CA72-4** 和 **NEU**。在分析过程中可能需要对这些高比例缺失的数据列进行特殊处理，例如删除或填充缺失值，以确保分析的准确性。


```python
# 在处理缺失数据之前，先将原始数据复制一份并存储到另一个变量中，以便稍后分析
cancer_data_missing = cancer_data.copy()

# 删除缺失数据比例大于 0.5 的列
# cols_to_drop = ['CA72-4', 'NEU']
cols_to_drop = ['CA72-4']  # 选择需要删除的列
cancer_data = cancer_data.drop(cols_to_drop, axis=1)  # 删除这些列

# 获取存在缺失数据的列名
cols_with_missing = [col for col in cancer_data.columns if cancer_data[col].isnull().any()]

# 用中位数填充缺失数据
for col in cols_with_missing:
    median_val = cancer_data[col].median()  # 计算列的中位数
    cancer_data[col].fillna(median_val, inplace=True)  # 用中位数填充缺失值
    
# 显示更新后的缺失数据比例
print(cancer_data.isnull().mean())
```

### 结果解读
输出结果显示了在处理缺失数据后的每一列的缺失比例，所有列的缺失比例都为 `0.0`，这意味着在执行了删除高缺失比例的列（`CA72-4`）和用中位数填充缺失值的操作后，数据集中的所有列都已经没有缺失值了。数据集已经被清理并准备好进行进一步的分析。


```python
# also drop ID column
# 同时删除 ID 列

# 删除 'SUBJECT_ID' 列，inplace=True 表示直接在原数据上进行操作，axis=1 表示按列删除
cancer_data.drop('SUBJECT_ID', inplace=True, axis=1)
```


## 特征工程

这部分代码指的是选择和转换数据的相关特征，以创建更好地表示问题领域的新特征的过程。特征工程的目的是通过减少数据中的噪声、提高预测的准确性以及使模型更具可解释性，从而改善机器学习算法的性能。特征工程需要对问题领域和所使用的数据有深刻的理解，并且需要了解可用的特征工程技术及其对模型性能的影响。

```python
# 将数据分为特征（X）和目标（y）

# 从数据集中删除目标列'TYPE'，剩下的列作为特征X存储在cancer_X_train中
cancer_X_train = cancer_data.drop('TYPE', axis=1)

# 将'TYPE'列单独提取出来作为目标变量y，并存储在cancer_y_train中
cancer_y_train = cancer_data['TYPE']
```

```python
# 使用 MRMR 算法选择前 10 个重要特征

# 从 mrmr 模块中导入 mrmr_classif 函数
from mrmr import mrmr_classif

# 使用 mrmr_classif 函数从数据中选择前 18 个最重要的特征
# X 为特征数据，y 为目标变量，K 表示要选择的特征数量
selected_features = mrmr_classif(X=cancer_X_train, y=cancer_y_train, K=18)
```

### 结果解读：
运行结果显示，MRMR 算法成功完成了对特征的重要性排序，并选择了前 18 个最重要的特征。进度条表示该过程已经100%完成，并且算法运行速度为31.08个特征/秒。最终，你将得到一个包含18个被选中重要特征的列表`selected_features`。



## 初步数据分析

这一部分的代码涉及对数据集的初步检查，以了解其结构、内容和质量。此过程包括检查缺失或错误数据的范围、探索变量的分布、识别任何异常值，并计算摘要统计信息。该部分的目的是获取数据集的洞察并为后续的数据处理步骤提供信息。它还包括使用各种绘图技术对数据进行可视化，以揭示变量之间的模式或关系。

### 代码逐行注释

```python
selected_features = mrmr_classif(X=cancer_X_train, y=cancer_y_train, K=18)
```
- `selected_features`：这是变量的名称，用于存储所选择的特征列表。
- `mrmr_classif(X=cancer_X_train, y=cancer_y_train, K=18)`：这行代码使用mRMR（最大相关最小冗余）算法来选择数据集中的特征。
  - `X=cancer_X_train`：输入特征集（排除了目标变量后的数据）。
  - `y=cancer_y_train`：目标变量，即要预测的类别标签。
  - `K=18`：选择的特征数量，此处选择了18个最重要的特征。

### 结果解读

结果显示了根据mRMR算法选择出的前18个重要特征。这些特征是：

1. **Age**（年龄）
2. **CEA**（癌胚抗原）
3. **IBIL**（间接胆红素）
4. **NEU**（中性粒细胞）
5. **Menopause**（绝经）
6. **CA125**（癌抗原125）
7. **ALB**（白蛋白）
8. **HE4**（人附睾蛋白4）
9. **GLO**（球蛋白）
10. **LYM%**（淋巴细胞百分比）
11. **AST**（天冬氨酸氨基转移酶）
12. **PLT**（血小板计数）
13. **HGB**（血红蛋白）
14. **ALP**（碱性磷酸酶）
15. **LYM#**（淋巴细胞绝对值）
16. **PCT**（血小板压积）
17. **Ca**（钙）
18. **CA19-9**（癌抗原19-9）

这些特征被认为是最具预测性的变量，可以帮助提高模型的性能。



### 代码逐行注释

```python
# 选择用于可视化相关性的特征列和目标列
selected_cols = selected_features + ['TYPE']
```
- `selected_cols`：创建一个包含所有选定特征列以及目标列 `'TYPE'` 的列表，用于后续分析和可视化。

```python
# 计算训练数据中选定特征与目标列之间的相关性
target_corr = cancer_data[selected_cols].corr()['TYPE']
```
- `cancer_data[selected_cols].corr()`：计算选择的特征与目标列之间的相关性矩阵。
- `['TYPE']`：从相关性矩阵中提取与目标列 `'TYPE'` 相关的所有列的相关性值。

```python
# 显示每个特征与目标列之间的相关性值，并按降序排列
print("Correlation of selected features with target column \n")
print(target_corr)
```
- `print("Correlation of selected features with target column \n")`：输出提示信息，表示即将显示特征与目标列之间的相关性。
- `print(target_corr)`：输出每个特征与目标列之间的相关性值，并按降序排列。

### 结果解读

```plaintext
Correlation of selected features with target column 

Age         -0.514098
CEA         -0.164260
IBIL         0.200451
NEU         -0.353062
Menopause   -0.455770
CA125       -0.372262
ALB          0.375415
HE4         -0.350991
GLO         -0.195630
LYM%         0.315035
AST         -0.215888
PLT         -0.270182
HGB          0.197863
ALP         -0.213249
LYM#         0.256494
PCT         -0.243719
Ca           0.187119
CA19-9      -0.155981
TYPE         1.000000
```

- **Age**（年龄）：与目标变量具有负相关性，相关系数为 -0.514098，表明随着年龄增加，目标变量值（可能是疾病的严重性）降低。
  
- **CEA**（癌胚抗原）：与目标变量轻微负相关，相关系数为 -0.164260。

- **IBIL**（间接胆红素）：与目标变量轻微正相关，相关系数为 0.200451。

- **NEU**（中性粒细胞）：负相关，相关系数为 -0.353062。

- **Menopause**（绝经）：与目标变量负相关，相关系数为 -0.455770。

- **CA125**（癌抗原125）：与目标变量负相关，相关系数为 -0.372262。

- **ALB**（白蛋白）：与目标变量正相关，相关系数为 0.375415。

- **HE4**（人附睾蛋白4）：与目标变量负相关，相关系数为 -0.350991。

- **GLO**（球蛋白）：轻微负相关，相关系数为 -0.195630。

- **LYM%**（淋巴细胞百分比）：正相关，相关系数为 0.315035。

- **AST**（天冬氨酸氨基转移酶）：轻微负相关，相关系数为 -0.215888。

- **PLT**（血小板计数）：与目标变量轻微负相关，相关系数为 -0.270182。

- **HGB**（血红蛋白）：轻微正相关，相关系数为 0.197863。

- **ALP**（碱性磷酸酶）：轻微负相关，相关系数为 -0.213249。

- **LYM#**（淋巴细胞绝对值）：正相关，相关系数为 0.256494。

- **PCT**（血小板压积）：与目标变量轻微负相关，相关系数为 -0.243719。

- **Ca**（钙）：与目标变量轻微正相关，相关系数为 0.187119。

- **CA19-9**（癌抗原19-9）：轻微负相关，相关系数为 -0.155981。

- **TYPE**：与目标变量完全相关，相关系数为 1.000000，这是因为这是与自己进行的相关性计算。

总体来看，这些相关性值帮助我们了解哪些特征与目标变量关系密切，从而在模型构建中可以给予更多的关注。


### 代码逐行注释

```python
# 选择与目标列（TYPE）相关的行或列
target_corr = cancer_data[selected_cols].corr()['TYPE']
```
- 计算数据集中选定特征列与目标列 `'TYPE'` 之间的相关性。

```python
# 计算每个相关性的绝对值，获取相关性的大小
target_corr = np.sqrt(target_corr ** 2)
```
- 使用平方再开方的方法来获得相关性值的绝对值，确保所有相关性值为正。

```python
# 按降序显示每个特征与目标列之间的相关性值
print(target_corr.sort_values(ascending=False))
```
- 将计算出的相关性值按降序排列，并打印出来，以便查看哪些特征与目标列的相关性最强。

### 结果解读

```plaintext
TYPE         1.000000
Age          0.514098
Menopause    0.455770
ALB          0.375415
CA125        0.372262
NEU          0.353062
HE4          0.350991
LYM%         0.315035
PLT          0.270182
LYM#         0.256494
PCT          0.243719
AST          0.215888
ALP          0.213249
IBIL         0.200451
HGB          0.197863
GLO          0.195630
Ca           0.187119
CEA          0.164260
CA19-9       0.155981
Name: TYPE, dtype: float64
```

- **TYPE**：相关性为 1.000000，这是因为它与自己计算的相关性，总是等于1。
- **Age**（年龄）：相关性为 0.514098，表示年龄与目标变量存在中等程度的正相关性。
- **Menopause**（绝经）：相关性为 0.455770，表示绝经与目标变量存在中等程度的正相关性。
- **ALB**（白蛋白）：相关性为 0.375415，表示白蛋白与目标变量存在中等程度的正相关性。
- **CA125**（癌抗原125）：相关性为 0.372262，表示CA125与目标变量存在中等程度的正相关性。
- **NEU**（中性粒细胞）：相关性为 0.353062，表示中性粒细胞与目标变量存在中等程度的正相关性。
- **HE4**（人附睾蛋白4）：相关性为 0.350991，表示HE4与目标变量存在中等程度的正相关性。
- **LYM%**（淋巴细胞百分比）：相关性为 0.315035，表示淋巴细胞百分比与目标变量存在中等程度的正相关性。
- **PLT**（血小板计数）：相关性为 0.270182，表示血小板计数与目标变量存在较弱的正相关性。
- **LYM#**（淋巴细胞绝对值）：相关性为 0.256494，表示淋巴细胞绝对值与目标变量存在较弱的正相关性。

其他特征如 **PCT**、**AST**、**ALP** 等的相关性较弱，表明这些特征与目标变量之间的线性关系不是特别显著。

通过查看这些相关性，能够帮助我们了解哪些特征对目标变量影响较大，从而在模型构建中可能给予更多的关注。


### 代码逐行注释

```python
import matplotlib.pyplot as plt
```
- 导入 `matplotlib.pyplot` 模块并将其命名为 `plt`，用于创建和显示图形。

```python
# 创建散点图矩阵
pd.plotting.scatter_matrix(cancer_data[selected_features], figsize=(12, 12))
```
- 使用 `pandas` 库中的 `scatter_matrix` 函数创建一个散点图矩阵。这个矩阵展示了所选特征之间的两两关系。`figsize=(12, 12)` 参数指定了图形的大小。

```python
plt.show()
```
- 显示创建的散点图矩阵。

### 结果解读

生成的图是一个散点图矩阵，其中每个子图显示两个特征之间的关系：

1. **对角线上的直方图**：对角线上的图表示每个特征的直方图，展示了该特征的数据分布情况。例如，你可以看到年龄（Age）、CEA、IBIL 等特征的分布形态。
  
2. **非对角线的散点图**：每个非对角线位置的图都是一个散点图，显示了两个不同特征之间的关系。例如，`Age` 与 `Menopause` 之间的散点图显示了它们的关系。 

3. **模式识别**：通过这些散点图，可以尝试识别不同特征之间的线性或非线性关系，以及可能存在的相关性。例如，某些特征之间可能会出现明显的相关模式（如呈现线性关系的特征对），而其他特征对之间可能没有明显的相关性。

4. **群体分布**：一些散点图可能显示出聚集或分离的群体，表明某些特征的组合可能有助于区分数据中的不同类别或群体。

这个散点图矩阵为探索数据特征之间的关系提供了一个直观的方式，有助于识别可能影响模型性能的变量间关系。




### 代码逐行注释

```python
# 可视化训练数据中目标变量的分布
target_variable = cancer_data['TYPE']
```
- 将数据集中 `TYPE` 列提取为目标变量 `target_variable`，该列表示分类标签。

```python
sns.countplot(x=target_variable, data=cancer_data)
```
- 使用 Seaborn 库的 `countplot` 函数生成目标变量的计数图。`x=target_variable` 指定 `TYPE` 列为 x 轴，`data=cancer_data` 指定数据源。

```python
plt.title('Distribution of Target Variable')
```
- 为图表添加标题“Distribution of Target Variable”，帮助说明图表内容。

```python
plt.show()
```
- 显示生成的图表。

### 图表解读

该图显示了 `TYPE` 变量的分布情况。`TYPE` 变量有两个类别，通常代表二分类问题中的两个类别，如 `0` 和 `1`。图中显示出每个类别的计数：

- **均衡的分类数据**：两个柱子高度非常接近，表示这两个类别在数据集中几乎均衡。这对于训练分类模型非常重要，因为均衡的类别分布可以避免模型偏向于某个特定的类别。
  
- **类别 `0` 和 `1`**：类别 `0` 和 `1` 各自的数量接近，可以推断出这是一个合理的均衡数据集，适合用于分类任务。

这种分布图有助于快速了解数据集中类别标签的分布，验证数据的均衡性，并帮助设计进一步的模型训练步骤。



### 代码逐行注释

```python
import seaborn as sns
```
- 导入 `seaborn` 库，`seaborn` 是一个基于 `matplotlib` 的高级数据可视化库，特别擅长绘制统计图形。

```python
# 创建一个相关矩阵
corr_matrix = np.around(cancer_data[selected_cols].corr(), 2)
```
- 使用 Pandas 计算数据集中选定列的相关矩阵。`cancer_data[selected_cols].corr()` 计算每对特征之间的相关系数，`np.around(..., 2)` 将相关系数四舍五入保留两位小数。

```python
# 使用 seaborn 绘制热图
plt.figure(figsize=(12,10))
```
- 设置图表大小，`figsize=(12,10)` 表示图表宽12英寸、高10英寸。

```python
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```
- 使用 Seaborn 的 `heatmap` 函数绘制相关矩阵的热图。`annot=True` 参数表示在每个单元格中显示相关系数的数值，`cmap='coolwarm'` 设置热图的颜色映射为冷暖色调。

```python
plt.title('Correlation Heatmap of Top Features')
```
- 为图表添加标题“Correlation Heatmap of Top Features”。

```python
plt.show()
```
- 显示生成的热图。

### 图表解读

该图是一个相关性热图，用于显示在数据集中选定特征之间的相关性。

- **颜色表示相关性**：热图中的颜色从深蓝色到深红色，表示相关系数的范围从 -1 到 1。红色表示正相关，蓝色表示负相关，颜色越深，相关性越强。

- **对角线为 1**：在相关矩阵中，对角线上的所有元素都为 1，因为每个特征与自身的相关性都是完全正相关。

- **负相关性较强的特征**：
  - `Age` 与 `TYPE` 显示出强负相关性（-0.51），这表明随着年龄的增加，`TYPE` 类别（可能是病情的严重程度）更低。
  - `Menopause` 与 `TYPE` 的负相关性也很高（-0.46），这可能意味着更年期状态与病情类别相关。
  
- **正相关性较强的特征**：
  - `ALB` 与 `TYPE` 具有较强的正相关性（0.38），这表明这两个变量之间可能存在某种关联。
  
- **相关性强的特征**：
  - `NEU` 和 `LYM%` 之间的相关性非常高（-0.85），表明它们之间存在强烈的负相关。

该热图帮助揭示了数据集中变量之间的关系，可以用于特征选择、特征工程等进一步的分析步骤。这些相关性信息对于理解数据和改进模型性能非常重要。



### 代码逐行注释

```python
# 过滤数据，仅包含 "TYPE" 目标列值为 0 的数据
positive_data = cancer_data[cancer_data['TYPE'] == 0]
```
- 从 `cancer_data` 数据集中筛选出 `TYPE` 列等于 0 的行。`TYPE` 列为 0 的行可能表示数据集中一种特定的类别或条件（例如，健康人群或非患病者）。

```python
# 使用 Seaborn 创建直方图
sns.histplot(data=positive_data, x="Age")
```
- 使用 Seaborn 的 `histplot` 函数为筛选后的数据集 `positive_data` 绘制 `Age` 列的直方图。`x="Age"` 表示将 `Age` 列的数据作为 X 轴来绘制直方图。

### 图表解读

- **直方图显示了“Age”列的年龄分布**，但仅限于 `TYPE` 等于 0 的数据（可能是健康人群或特定类别的样本）。直方图的 X 轴显示不同的年龄段，Y 轴显示每个年龄段中的样本数量。

- 从图中可以看出，**年龄在50岁左右**的人群数量最多，其次是60岁左右的群体。较少的样本分布在年轻（20岁左右）和年老（80岁左右）的群体中。

- 这种分布可能表明研究样本集中在中年到老年群体之间，而在极端的年龄段（如非常年轻或非常老的群体）样本较少。这种分布可能与研究的目标群体有关，比如某种疾病的易感年龄群体。


### 代码逐行注释

```python
# 计算每一列中缺失值的比例
missing_ratio = cancer_data_missing.isnull().sum() / len(cancer_data_missing)
```
- 这行代码计算了 `cancer_data_missing` 数据集中每一列中缺失值的比例。`isnull().sum()` 计算每列的缺失值个数，将其除以数据集的总行数以获得缺失值的比例。

```python
# 筛选出有缺失值的列
missing_ratio = missing_ratio[missing_ratio > 0]
```
- 这行代码筛选出有缺失值的列，仅保留那些缺失值比例大于 0 的列。

```python
# 绘制条形图
missing_ratio.plot(kind='bar')
plt.title('Ratio of missing values in columns with missing values')
plt.xlabel('Column name')
plt.ylabel('Ratio of missing values')
plt.show()
```
- 通过 `plot(kind='bar')` 函数绘制条形图，显示各列中缺失值的比例。`plt.title`、`plt.xlabel` 和 `plt.ylabel` 分别设置了图表的标题、X 轴和 Y 轴的标签。



### 图表解读

- **条形图展示了具有缺失值的各列的缺失比例**。X 轴显示了列名，Y 轴显示了各列的缺失值比例。

- 从图中可以看出，**CA72-4** 列的缺失值比例非常高，超过了 70%。另外，**NEU** 列的缺失比例也较高，接近 30%。

- 其余列的缺失值比例较低，多数在 10% 以下。

- 这些信息对于数据清洗和处理非常重要。高缺失比例的列可能需要特殊处理，如删除该列或进行缺失值填补（如使用平均值、中位数或预测模型）。






## 模型训练

这部分代码涉及使用机器学习算法来构建检测卵巢癌的预测模型。该部分包括选择适当的算法，将数据分为训练集和测试集，在训练数据上训练模型，并在测试数据上评估其性能。

本节的目标是找出最准确和有效的集成学习方法，用于检测卵巢癌。集成学习是一种机器学习技术，通过结合多个模型（称为基模型）来提高整体预测的性能和准确性。








Base Models Checked:

SVM
KNN
Decision Trees
Ensemble Learning Techniques Checked:

Max Voting
Stacking
Bagging
Boosting
Stacking of Various Ensemble Learning Techniques



```python
from sklearn.model_selection import train_test_split

# 将数据分割为训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(cancer_X_train, cancer_y_train, test_size=0.3, random_state=42)
# 使用selected_features中的特征对训练集进行选择
X_train = X_train[selected_features]
# 使用selected_features中的特征对测试集进行选择
X_test = X_test[selected_features]
```

### 代码解读
1. **导入`train_test_split`函数**: 从`sklearn.model_selection`模块中导入`train_test_split`函数，用于将数据集划分为训练集和测试集。

2. **将数据划分为训练集和验证集**:
   - `X_train`, `X_test`: 分别是特征数据的训练集和测试集。
   - `y_train`, `y_test`: 分别是标签数据的训练集和测试集。
   - `test_size=0.3`: 表示将 30% 的数据用于测试，其余 70% 用于训练。
   - `random_state=42`: 固定随机种子，以便每次运行代码时生成相同的分割结果。

3. **选择特定特征进行训练**:
   - `X_train = X_train[selected_features]`: 在训练集中只保留之前选择的特征列。
   - `X_test = X_test[selected_features]`: 在测试集中只保留之前选择的特征列。



```python
# 创建一个DataFrame来存储基模型的准确率，以便后续分析
basemodel_df = pd.DataFrame(columns=['Base Model', 'Accuracy'])
```

### 代码解读
1. **创建一个DataFrame**:
   - `basemodel_df` 是一个新的空的 `DataFrame`，将用于存储基模型的名称和其对应的准确率。
   
2. **指定列名**:
   - `columns=['Base Model', 'Accuracy']`: 指定 `DataFrame` 的列名为 `'Base Model'` 和 `'Accuracy'`。这两列分别用于存储基模型的名称以及其在测试集上的准确率。

这段代码的目的是为后续基模型的准确率分析创建一个结构化的表格，以便能够方便地比较不同模型的性能。



### 逐行中文注释

```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Training the SVM model
svm_model = SVC(kernel='linear')  # 创建一个支持向量机模型，使用线性核函数
svm_model.fit(X_train, y_train)  # 在训练数据上训练模型

# Predicting the target values for test data
test_preds = svm_model.predict(X_test)  # 使用训练好的模型对测试数据进行预测

accuracy = svm_model.score(X_test, y_test)  # 计算模型在测试集上的准确率
basemodel_df = basemodel_df.append({'Base Model': "SVM", 'Accuracy': accuracy}, ignore_index=True)  # 将SVM模型的名称和准确率添加到basemodel_df中

# evaluate the model on the test set
print("SVM:")  # 打印 "SVM:" 作为输出的标题
print(classification_report(y_test, test_preds))  # 输出模型的分类报告，包括精确度、召回率、F1分数等指标
```

### 结果解读

```plaintext
SVM:
              precision    recall  f1-score   support

           0       0.86      0.83      0.85        53
           1       0.83      0.87      0.85        52

    accuracy                           0.85       105
   macro avg       0.85      0.85      0.85       105
weighted avg       0.85      0.85      0.85       105
```

#### 结果解读：
- **Precision（精确率）**:
  - 对于类别 `0`：精确率为 `0.86`，即模型预测为 `0` 的所有样本中，实际为 `0` 的比例是 `86%`。
  - 对于类别 `1`：精确率为 `0.83`，即模型预测为 `1` 的所有样本中，实际为 `1` 的比例是 `83%`。
  
- **Recall（召回率）**:
  - 对于类别 `0`：召回率为 `0.83`，即实际为 `0` 的所有样本中，模型正确识别的比例是 `83%`。
  - 对于类别 `1`：召回率为 `0.87`，即实际为 `1` 的所有样本中，模型正确识别的比例是 `87%`。

- **F1-Score**:
  - 对于类别 `0` 和 `1` 的 `F1-Score` 都是 `0.85`，这是精确率和召回率的调和平均值，综合考虑了模型的表现。

- **Accuracy（准确率）**:
  - 模型在测试集上的总体准确率为 `0.85`，即在 `105` 个测试样本中，模型正确预测了 `85%` 的样本。

- **Macro Avg 和 Weighted Avg**:
  - `Macro Avg` 是对两个类别的精确率、召回率、F1-Score 的简单平均，反映了模型在每个类别上的平均表现。
  - `Weighted Avg` 是对各类别的精确率、召回率、F1-Score 进行加权平均，考虑了每个类别样本数量的影响。

总体来说，SVM 模型在测试集上的表现较好，具有较高的准确率和均衡的精确率与召回率。



### 逐行中文注释

```python
from sklearn.neighbors import KNeighborsClassifier

# create KNN classifier
knn = KNeighborsClassifier()  # 创建一个k近邻分类器

# train the model
knn.fit(X_train, y_train)  # 在训练数据上训练KNN模型

# Predicting the target values for test data
test_preds = knn.predict(X_test)  # 使用训练好的KNN模型对测试数据进行预测

accuracy = knn.score(X_test, y_test)  # 计算模型在测试集上的准确率
basemodel_df = basemodel_df.append({'Base Model': "KNN", 'Accuracy': accuracy}, ignore_index=True)  # 将KNN模型的名称和准确率添加到basemodel_df中

# evaluate the model on the test set
print("KNN:")  # 打印 "KNN:" 作为输出的标题
print(classification_report(y_test, test_preds))  # 输出模型的分类报告，包括精确度、召回率、F1分数等指标
```

### 结果解读

```plaintext
KNN:
              precision    recall  f1-score   support

           0       0.82      0.70      0.76        53
           1       0.73      0.85      0.79        52

    accuracy                           0.77       105
   macro avg       0.78      0.77      0.77       105
weighted avg       0.78      0.77      0.77       105
```

#### 结果解读：
- **Precision（精确率）**:
  - 对于类别 `0`：精确率为 `0.82`，即模型预测为 `0` 的所有样本中，实际为 `0` 的比例是 `82%`。
  - 对于类别 `1`：精确率为 `0.73`，即模型预测为 `1` 的所有样本中，实际为 `1` 的比例是 `73%`。
  
- **Recall（召回率）**:
  - 对于类别 `0`：召回率为 `0.70`，即实际为 `0` 的所有样本中，模型正确识别的比例是 `70%`。
  - 对于类别 `1`：召回率为 `0.85`，即实际为 `1` 的所有样本中，模型正确识别的比例是 `85%`。

- **F1-Score**:
  - 对于类别 `0` 的 `F1-Score` 是 `0.76`，对于类别 `1` 的 `F1-Score` 是 `0.79`，这是精确率和召回率的调和平均值，综合考虑了模型的表现。

- **Accuracy（准确率）**:
  - 模型在测试集上的总体准确率为 `0.77`，即在 `105` 个测试样本中，模型正确预测了 `77%` 的样本。

- **Macro Avg 和 Weighted Avg**:
  - `Macro Avg` 是对两个类别的精确率、召回率、F1-Score 的简单平均，反映了模型在每个类别上的平均表现。
  - `Weighted Avg` 是对各类别的精确率、召回率、F1-Score 进行加权平均，考虑了每个类别样本数量的影响。

总体来说，KNN 模型在测试集上的表现较为一般，相比于前面提到的 SVM 模型，KNN 的准确率略低，但在类别 `1` 上表现更好，召回率达到了 `85%`。


### 逐行中文注释

```python
from sklearn.tree import DecisionTreeClassifier

# create decision tree classifier
clf = DecisionTreeClassifier(random_state=0)  # 创建一个决策树分类器，并设置随机种子为0

# fit the model to the training data
clf.fit(X_train, y_train)  # 将决策树模型拟合到训练数据

# predict on the test data
y_pred = clf.predict(X_test)  # 使用训练好的模型对测试数据进行预测

accuracy = clf.score(X_test, y_test)  # 计算模型在测试集上的准确率
basemodel_df = basemodel_df.append({'Base Model': "Decision Tree", 'Accuracy': accuracy}, ignore_index=True)  # 将决策树模型的名称和准确率添加到basemodel_df中

# evaluate the model on the test set
print("Decision Trees:")  # 打印 "Decision Trees:" 作为输出的标题
print(classification_report(y_test, y_pred))  # 输出模型的分类报告，包括精确度、召回率、F1分数等指标
```

### 结果解读

```plaintext
Decision Trees:
              precision    recall  f1-score   support

           0       0.80      0.85      0.83        53
           1       0.84      0.79      0.81        52

    accuracy                           0.82       105
   macro avg       0.82      0.82      0.82       105
weighted avg       0.82      0.82      0.82       105
```

#### 结果解读：
- **Precision（精确率）**:
  - 对于类别 `0`：精确率为 `0.80`，即模型预测为 `0` 的所有样本中，实际为 `0` 的比例是 `80%`。
  - 对于类别 `1`：精确率为 `0.84`，即模型预测为 `1` 的所有样本中，实际为 `1` 的比例是 `84%`。

- **Recall（召回率）**:
  - 对于类别 `0`：召回率为 `0.85`，即实际为 `0` 的所有样本中，模型正确识别的比例是 `85%`。
  - 对于类别 `1`：召回率为 `0.79`，即实际为 `1` 的所有样本中，模型正确识别的比例是 `79%`。

- **F1-Score**:
  - 对于类别 `0` 的 `F1-Score` 是 `0.83`，对于类别 `1` 的 `F1-Score` 是 `0.81`，这是精确率和召回率的调和平均值，综合考虑了模型的表现。

- **Accuracy（准确率）**:
  - 模型在测试集上的总体准确率为 `0.82`，即在 `105` 个测试样本中，模型正确预测了 `82%` 的样本。

- **Macro Avg 和 Weighted Avg**:
  - `Macro Avg` 是对两个类别的精确率、召回率、F1-Score 的简单平均，反映了模型在每个类别上的平均表现。
  - `Weighted Avg` 是对各类别的精确率、召回率、F1-Score 进行加权平均，考虑了每个类别样本数量的影响。

总体来说，决策树模型在测试集上的表现相对均衡，精确率和召回率都较高，准确率达到 `82%`，在类别 `0` 和 `1` 上的表现相对接近。


```python
from sklearn.metrics import accuracy_score  # 从sklearn.metrics导入accuracy_score函数，用于计算模型的准确率

#  Create a dataframe to store the accuracy of ensemble models for further analysis
ensemble_df = pd.DataFrame(columns=['Ensemble Model', 'Accuracy'])  # 创建一个DataFrame，用于存储集成模型的名称及其对应的准确率，列名为'Ensemble Model'和'Accuracy'
``` 

### 代码解释：
1. **导入库**:
   - `accuracy_score`: 这是一个用于计算分类模型准确率的函数。它将预测值与真实标签进行比较，并返回正确预测的比例。

2. **创建DataFrame**:
   - 使用 `pd.DataFrame()` 创建一个空的 DataFrame，并指定列名为 `Ensemble Model` 和 `Accuracy`。
   - 该 DataFrame 将用于保存不同集成模型的名称及其对应的准确率，以便进一步分析和比较模型的性能。


### 代码逐行注释

```python
# importing voting classifier
from sklearn.ensemble import VotingClassifier  # 从sklearn.ensemble模块导入VotingClassifier，用于实现投票分类器

# Making the final model using voting classifier
vote_model = VotingClassifier(estimators=[('svc', svm_model), ('knn', knn), ('tree', clf)], voting='hard')  
# 创建一个投票分类器模型vote_model，其中包含三个基模型：SVM（svc）、KNN（knn）和决策树（tree）
# voting='hard'表示使用硬投票，即每个基模型的预测结果通过多数投票决定最终预测

# training all the model on the train dataset
vote_model.fit(X_train, y_train)  # 在训练集上训练投票分类器模型

# predicting the output on the test dataset
pred_final = vote_model.predict(X_test)  # 使用训练好的投票分类器模型对测试集进行预测，并将预测结果存储在pred_final中

accuracy = vote_model.score(X_test, y_test)  # 计算投票分类器模型在测试集上的准确率
ensemble_df = ensemble_df.append({'Ensemble Model': "Max Voting", 'Accuracy': accuracy}, ignore_index=True)  
# 将投票分类器的模型名称和准确率添加到ensemble_df数据框中

# evaluate the model on the test set
print("Max Voting:")  # 打印投票分类器模型的名称
print(classification_report(y_test, pred_final))  # 打印投票分类器模型在测试集上的评估报告，包括precision、recall、f1-score和支持数
```

### 代码结果解读

```plaintext
Max Voting:
              precision    recall  f1-score   support

           0       0.86      0.83      0.85        53
           1       0.83      0.87      0.85        52

    accuracy                           0.85       105
   macro avg       0.85      0.85      0.85       105
weighted avg       0.85      0.85      0.85       105
```

- **Precision** (精确率): 对于类别 `0`，模型的精确率为 `0.86`，对于类别 `1`，模型的精确率为 `0.83`。精确率表示模型预测的正例中有多少是正确的。

- **Recall** (召回率): 对于类别 `0`，模型的召回率为 `0.83`，对于类别 `1`，模型的召回率为 `0.87`。召回率表示在所有实际为正例的数据中，模型能够正确识别出多少。

- **F1-Score** (F1得分): F1得分是精确率和召回率的调和平均值，对于类别 `0` 和 `1`，F1得分均为 `0.85`。

- **Accuracy** (准确率): 模型的整体准确率为 `0.85`，即在105个测试样本中，有85%的样本被正确分类。

- **Support** (支持数): `support` 指的是每个类别在测试集中的样本数，类别 `0` 和 `1` 分别有 `53` 和 `52` 个样本。

总的来说，这表明投票分类器在该数据集上表现良好，准确率达到了85%。



### 代码逐行注释

```python
# importing stacking lib for Stack Method
from vecstack import stacking  # 从vecstack库中导入stacking函数，用于实现模型堆叠

# putting all base model objects in one list
all_models = [svm_model, clf, knn]  # 将所有基模型（SVM、决策树、KNN）放入一个列表中

# computing the stack features
s_train, s_test = stacking(all_models,                     # 基模型的列表
                           X_train, y_train, X_test,   # 训练数据、训练标签和测试数据
                           regression=False,           # 设置为分类任务（如果是回归任务，则设置为True）
                           n_folds=5,                  # 交叉验证的折数，设置为5折
                           shuffle=False,               # 是否在进行交叉验证时打乱数据，设置为False
                           random_state=None,             # 随机种子，以确保结果可复现
                           verbose=1)                  # 设置为1以打印详细信息

# initializing the second-level model
final_model = clf  # 初始化第二层模型，使用决策树分类器

# fitting the second level model with stack features
final_model = final_model.fit(s_train, y_train)  # 使用堆叠生成的特征训练第二层模型

# predicting the final output using stacking
pred_final = final_model.predict(s_test)  # 使用训练好的第二层模型对测试集进行预测，并将结果存储在pred_final中

accuracy = accuracy_score(y_test, pred_final)  # 计算堆叠模型在测试集上的准确率
ensemble_df = ensemble_df.append({'Ensemble Model': "Stacking", 'Accuracy': accuracy}, ignore_index=True)  
# 将堆叠模型的名称和准确率添加到ensemble_df数据框中

# calculate accuracy score
# evaluate the model on the test set
print("Stacking:")  # 打印堆叠模型的名称
print(classification_report(y_test, pred_final))  # 打印堆叠模型在测试集上的评估报告，包括precision、recall、f1-score和支持数
```

### 代码结果解读

#### 堆叠过程中的基模型评估：

```plaintext
task:         [classification]
n_classes:    [2]
metric:       [accuracy_score]
mode:         [oof_pred_bag]
n_models:     [3]

model  0:     [SVC]
    ----
    MEAN:     [0.81113946] + [0.05545391]
    FULL:     [0.81147541]

model  1:     [DecisionTreeClassifier]
    ----
    MEAN:     [0.79923469] + [0.04700805]
    FULL:     [0.79918033]

model  2:     [KNeighborsClassifier]
    ----
    MEAN:     [0.79506803] + [0.04284229]
    FULL:     [0.79508197]
```

- 堆叠过程中的三个基模型分别是SVC、决策树分类器和KNN分类器。上面的结果显示了每个模型在交叉验证过程中的平均准确率（`MEAN`）和全体数据上的准确率（`FULL`）。
- SVC模型的平均准确率为`81.11%`，决策树分类器为`79.92%`，KNN分类器为`79.51%`。

#### 堆叠模型的最终评估：

```plaintext
Stacking:
              precision    recall  f1-score   support

           0       0.88      0.79      0.83        53
           1       0.81      0.88      0.84        52

    accuracy                           0.84       105
   macro avg       0.84      0.84      0.84       105
weighted avg       0.84      0.84      0.84       105
```

- **Precision** (精确率): 对于类别 `0`，堆叠模型的精确率为 `0.88`，对于类别 `1`，堆叠模型的精确率为 `0.81`。
- **Recall** (召回率): 对于类别 `0`，堆叠模型的召回率为 `0.79`，对于类别 `1`，堆叠模型的召回率为 `0.88`。
- **F1-Score** (F1得分): F1得分是精确率和召回率的调和平均值，对于类别 `0` 和 `1`，F1得分分别为 `0.83` 和 `0.84`。
- **Accuracy** (准确率): 堆叠模型的整体准确率为 `0.84`，即在105个测试样本中，有84%的样本被正确分类。

总的来说，堆叠模型在该数据集上表现良好，整体准确率达到了84%。这表明通过组合多个模型，可以有效地提高预测性能。





### 代码逐行注释

```python
# importing bagging module for Bagging Method
from sklearn.ensemble import BaggingClassifier  # 从sklearn.ensemble导入BaggingClassifier，用于实现Bagging集成方法

# initializing the bagging model using XGboost as base model with default parameters
bag_model = BaggingClassifier(base_estimator=svm_model)  # 初始化Bagging模型，使用SVM模型作为基模型

# training model
bag_model.fit(X_train, y_train)  # 训练Bagging模型，使用训练集X_train和y_train

# predicting the output on the test dataset
pred = bag_model.predict(X_test)  # 使用训练好的Bagging模型对测试集进行预测
pred = np.around(pred).astype("int64")  # 将预测结果四舍五入并转换为整数类型

# calculate accuracy score
accuracy = accuracy_score(y_test, pred)  # 计算Bagging模型在测试集上的准确率
ensemble_df = ensemble_df.append({'Ensemble Model': "Bagging", 'Accuracy': accuracy}, ignore_index=True)  
# 将Bagging模型的名称和准确率添加到ensemble_df数据框中

# evaluate the model on the test set
print("Bagging:")  # 打印模型名称
print(classification_report(y_test, pred))  # 打印Bagging模型在测试集上的评估报告，包括precision、recall、f1-score和支持数
```

### 代码结果解读

```plaintext
Bagging:
              precision    recall  f1-score   support

           0       0.83      0.85      0.84        53
           1       0.84      0.83      0.83        52

    accuracy                           0.84       105
   macro avg       0.84      0.84      0.84       105
weighted avg       0.84      0.84      0.84       105
```

- **Precision** (精确率): 对于类别 `0`（阴性），Bagging模型的精确率为 `0.83`，对于类别 `1`（阳性），精确率为 `0.84`。精确率表示模型在所有预测为该类的样本中，正确预测的比例。
  
- **Recall** (召回率): 对于类别 `0`，Bagging模型的召回率为 `0.85`，对于类别 `1`，召回率为 `0.83`。召回率表示在所有真实属于该类的样本中，被正确预测为该类的比例。

- **F1-Score** (F1得分): F1得分是精确率和召回率的调和平均值，对于类别 `0` 和 `1`，F1得分分别为 `0.84` 和 `0.83`。

- **Accuracy** (准确率): Bagging模型的整体准确率为 `0.84`，即在105个测试样本中，有84%的样本被正确分类。

总体来说，Bagging模型在测试集上的表现较好，达到了84%的准确率。这表明通过使用Bagging集成方法，可以有效地提高分类模型的鲁棒性和预测性能。


### 代码逐行注释

```python
# importing machine learning models for prediction
from sklearn.ensemble import GradientBoostingClassifier  # 导入梯度提升分类器
from xgboost import XGBClassifier  # 导入XGBoost分类器

# initializing the boosting module with default parameters
model = GradientBoostingClassifier()  # 初始化梯度提升模型
xgb_model = XGBClassifier()  # 初始化XGBoost模型

# training the model on the train dataset
#model.fit(X_train, y_train)  # 训练梯度提升模型，这行被注释掉了
xgb_model.fit(X_train, y_train)  # 训练XGBoost模型

# predicting the output on the test dataset
pred_final = xgb_model.predict(X_test)  # 使用XGBoost模型对测试数据进行预测
 
accuracy = accuracy_score(y_test, pred_final)  # 计算预测的准确率
ensemble_df = ensemble_df.append({'Ensemble Model': "Boosting", 'Accuracy': accuracy}, ignore_index=True)  
# 将Boosting模型的名称和准确率添加到ensemble_df数据框中

# evaluate the model on the test set
print("Boosting:")  # 打印模型名称
print(classification_report(y_test, pred_final))  # 打印Boosting模型在测试集上的评估报告
```

### 代码结果解读

```plaintext
Boosting:
              precision    recall  f1-score   support

           0       0.85      0.89      0.87        53
           1       0.88      0.85      0.86        52

    accuracy                           0.87       105
   macro avg       0.87      0.87      0.87       105
weighted avg       0.87      0.87      0.87       105
```

- **Precision** (精确率): 对于类别 `0`（阴性），Boosting模型的精确率为 `0.85`，对于类别 `1`（阳性），精确率为 `0.88`。精确率表示模型在所有预测为该类的样本中，正确预测的比例。
  
- **Recall** (召回率): 对于类别 `0`，Boosting模型的召回率为 `0.89`，对于类别 `1`，召回率为 `0.85`。召回率表示在所有真实属于该类的样本中，被正确预测为该类的比例。

- **F1-Score** (F1得分): F1得分是精确率和召回率的调和平均值，对于类别 `0` 和 `1`，F1得分分别为 `0.87` 和 `0.86`。

- **Accuracy** (准确率): Boosting模型的整体准确率为 `0.87`，即在105个测试样本中，有87%的样本被正确分类。

总体来说，Boosting模型在测试集上的表现较优，达到了87%的准确率，这表明使用Boosting集成方法可以有效地提高分类模型的性能和预测精度。


### 代码逐行注释

```python
# Combining all ensemble models; (bagging, boosting, max_vote) with stacking
# importing stacking lib
from vecstack import stacking  # 导入vecstack库用于堆叠模型
 
# putting all base model objects in one list
all_models = [xgb_model, vote_model, bag_model]  # 将所有基础模型放入一个列表中
 
# computing the stack features
s_train, s_test = stacking(
    all_models,               # 使用的基础模型列表
    X_train, y_train, X_test, # 训练数据、训练标签和测试数据
    regression=False,         # 指定任务为分类任务（如果需要回归任务，则设为True）
    n_folds=5,                # 使用5折交叉验证
    shuffle=False,            # 不对数据进行打乱
    random_state=None,        # 不设定随机种子，确保结果可重复
    verbose=1)                # 打印所有信息
 
# initializing the second-level model
final_model = xgb_model  # 初始化第二层的模型，使用XGBoost作为最终模型
 
# fitting the second level model with stack features
final_model = final_model.fit(s_train, y_train)  # 使用堆叠特征训练最终模型
 
# predicting the final output using stacking
pred_final = final_model.predict(s_test)  # 使用堆叠模型进行预测
 
# calculate accuracy score
accuracy = accuracy_score(y_test, pred_final)  # 计算最终模型的准确率
ensemble_df = ensemble_df.append({'Ensemble Model': "Ensemble Combination", 'Accuracy': accuracy}, ignore_index=True)
# 将组合模型的名称和准确率添加到ensemble_df数据框中

# evaluate the model on the test set
print("Stacking:")  # 打印模型名称
print(classification_report(y_test, pred_final))  # 打印组合模型在测试集上的评估报告
```

### 代码结果解读

```plaintext
task:         [classification]
n_classes:    [2]
metric:       [accuracy_score]
mode:         [oof_pred_bag]
n_models:     [3]

model  0:     [XGBClassifier]
    ----
    MEAN:     [0.86479592] + [0.03542154]
    FULL:     [0.86475410]

model  1:     [VotingClassifier]
    ----
    MEAN:     [0.82772109] + [0.04820941]
    FULL:     [0.82786885]

model  2:     [BaggingClassifier]
    ----
    MEAN:     [0.80280612] + [0.06622638]
    FULL:     [0.80327869]

Stacking:
              precision    recall  f1-score   support

           0       0.87      0.91      0.89        53
           1       0.90      0.87      0.88        52

    accuracy                           0.89       105
   macro avg       0.89      0.89      0.89       105
weighted avg       0.89      0.89      0.89       105
```

- **Precision** (精确率): 对于类别 `0`（阴性），组合模型的精确率为 `0.87`，对于类别 `1`（阳性），精确率为 `0.90`。精确率表示模型在所有预测为该类的样本中，正确预测的比例。
  
- **Recall** (召回率): 对于类别 `0`，组合模型的召回率为 `0.91`，对于类别 `1`，召回率为 `0.87`。召回率表示在所有真实属于该类的样本中，被正确预测为该类的比例。

- **F1-Score** (F1得分): F1得分是精确率和召回率的调和平均值，对于类别 `0` 和 `1`，F1得分分别为 `0.89` 和 `0.88`。

- **Accuracy** (准确率): 组合模型的整体准确率为 `0.89`，即在105个测试样本中，有89%的样本被正确分类。

总体来看，组合模型（堆叠）通过结合多个基础模型的预测，显著提高了分类的准确率，并且在测试数据上的表现非常稳定，达到了89%的准确率。



## Analysis of Ensemble Methods 

这部分代码通过使用适当的评估指标来评估在模型训练部分中使用的集成学习方法的性能，并比较其结果。目的是确定在检测卵巢癌肿瘤方面最有效和高效的方法，并找出导致某一集成学习技术表现优于其他技术的因素。


```python
# select top 10 features using mRMR
# 使用 mRMR 方法选择前 10 个特征
from mrmr import mrmr_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
# 将数据拆分为训练集和验证集
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(
    cancer_X_train,  # 特征数据
    cancer_y_train,  # 目标变量
    test_size=0.3,   # 测试集占总数据的 30%
    random_state=42  # 设置随机种子，保证结果可重复
)
```


































































































































































































































































































































































































































































































