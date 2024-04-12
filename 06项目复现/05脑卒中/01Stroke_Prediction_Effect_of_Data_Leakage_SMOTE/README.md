# Stroke Prediction: Effect of Data Leakage | SMOTE
## 基本信息

[网址](https://www.kaggle.com/code/tanmay111999/stroke-prediction-effect-of-data-leakage-smote)

2023年发布

9，331次浏览

132个人支持

100人复现


![0封面](01图片/0封面.png)


## 0. 简介
### 0.1. Problem Statement :

According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. It is another health issue that has found to be rising throughout the world due to the adoption of lifestyle changes that disregards healthy lifestyle & good eating habits. Thus, new emerging electronic devices that record the health vitals have paved the way for creating an automated solution with AI techniques at it's core. Thus, similar to heart diseases, efforts have begun to create lab tests that predict stroke. The dataset presented here has many factors that highlight the lifestyle of the patients and hence gives us an opportunity to create an AI-based solution for it.

### 0.2. Aim :
* To classify / predict whether a patient can suffer a stroke.
* It is a binary classification problem with multiple numerical and categorical features.

### 0.3. Dataset Attributes :¶
* id : unique identifier
* gender : "Male", "Female" or "Other"
* age : age of the patient
* hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
* heart_disease : 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
* ever_married : "No" or "Yes"
* work_type : "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
* Residence_type : "Rural" or "Urban"
* avg_glucose_level : average glucose level in blood
* bmi : body mass index
* smoking_status : "formerly smoked", "never smoked", "smokes" or "Unknown"*
* stroke : 1 if the patient had a stroke or 0 if not

### 0.4. Notebook Contents :
* Dataset Information
* Exploratory Data Analysis (EDA)
* Summary of EDA & Comparison with Domain Information
* Feature Engineering (Data Leakage & Data Balancing)
* Modeling
* Conclusion
### 0.5. What you will learn :
* Data Visualization
* Data Balancing using SMOTE
* Data Leakage
* Statistical Tests for Feature Selection
* Modeling and visualization of results for algorithms


Lets get started!


## 1. Data Visualization
### 1.1. Import the Necessary Libraries :

这段代码是Python编程语言中的一段示例，用于数据科学和机器学习任务。下面是对每行代码的详细中文注释：

```python
# 导入pandas库，并使用别名pd。Pandas是一个强大的数据结构和数据分析工具。
import pandas as pd

# 导入numpy库，并使用别名np。Numpy提供了对多维数组对象的支持以及各种派生对象（如掩码数组和矩阵）。
import numpy as np

# 导入matplotlib.pyplot模块，并使用别名plt。这个模块可以用来绘制图形和数据可视化。
import matplotlib.pyplot as plt

# 这行代码是Jupyter Notebook的魔法命令，用于在Notebook内部直接显示图表。
%matplotlib inline

# 导入seaborn库，并使用别名sns。Seaborn是基于matplotlib的高级数据可视化库，提供了更多样化的绘图风格和接口。
import seaborn as sns

# 设置pandas显示选项，使浮点数显示为两位小数。
pd.options.display.float_format = '{:.2f}'.format

# 导入warnings模块，用于控制警告的显示。
import warnings

# 从tqdm模块导入tqdm函数，它是一个快速、可扩展的进度条工具，可以在Python长循环中添加一个进度提示。
from tqdm import tqdm

# 使用warnings过滤器来忽略警告信息，使得在运行代码时不会出现烦人的警告信息。
warnings.filterwarnings('ignore')

# 从sklearn.preprocessing模块导入LabelEncoder类，用于将标签转换为范围从0到n_classes-1的整数。
from sklearn.preprocessing import LabelEncoder
```
这段代码主要涉及数据处理和可视化的库的导入和设置，为后续的数据科学工作做好准备。


```python
data = pd.read_csv('/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')
data.head()
```
![1head](01图片/1.1head.png)

### 1.2. Data Info :

```python
data.shape
```

```python
(5110, 12)
```

```python
data.columns
# 这行代码用于获取Pandas DataFrame 'data' 的所有列名。
# 'data' 是一个Pandas DataFrame对象，它是一个二维的、表格型的数据结构，非常适合于数据处理和分析。
# 'columns' 是DataFrame对象的一个属性，它返回一个Index对象，包含了DataFrame中所有列的名称。
# 当执行这行代码时，它会打印出DataFrame 'data' 中每一列的名称，通常用于了解数据集的结构或者进行数据的初步探索。
data.columns
```


```python
data.info()
# 调用DataFrame对象'data'的info()方法，该方法会输出DataFrame的概要信息。
# 这个方法通常用于初步了解数据集的结构和内容，包括每列的数据类型、非空值的数量等。
# 执行这行代码后，Pandas会打印出一份报告，其中包含了以下信息：
#   - 数据集的行数和列数。
#   - 每列的名称、数据类型（例如整数、浮点数、对象等）以及非空值的数量。
#   - 数据集的内存使用情况。
# 这个概要信息有助于在数据分析的早期阶段识别潜在的问题，如缺失值、数据类型不一致等。
data.info()
```

当你执行`data.info()`方法时，Pandas会输出一个表格，其中列出了DataFrame的每一列以及相关的信息。这个表格通常会包含列名、数据类型、非空值的数量，以及每列的内存使用情况。这个功能对于快速检查数据集的完整性和准确性非常有用，可以帮助你做出是否需要进行数据清洗和预处理的决策。



```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB
```

根据提供的`data.info()`方法的输出结果，我们可以对数据集`data`进行以下详细分析：

1. **DataFrame 类型与索引**:
   - `data`是一个Pandas的DataFrame对象。
   - 使用RangeIndex索引，共有5110个数据条目，索引范围从0到5109。

2. **数据概览**:
   - 共有12列数据，每一列的名称和数据类型如下：
     - `id`列：数据类型为`int64`，表示整数类型，共有5110个非空值。
     - `gender`列：数据类型为`object`，通常表示字符串类型，共有5110个非空值。
     - `age`列：数据类型为`float64`，表示双精度浮点数类型，共有5110个非空值。
     - `hypertension`列：数据类型为`int64`，共有5110个非空值。
     - `heart_disease`列：数据类型为`int64`，共有5110个非空值。
     - `ever_married`列：数据类型为`object`，共有5110个非空值。
     - `work_type`列：数据类型为`object`，共有5110个非空值。
     - `Residence_type`列：数据类型为`object`，共有5110个非空值。
     - `avg_glucose_level`列：数据类型为`float64`，共有5110个非空值。
     - `bmi`列：数据类型为`float64`，但有201个空值（5110 - 4909 = 201），表示有201条记录缺失BMI信息。
     - `smoking_status`列：数据类型为`object`，共有5110个非空值。
     - `stroke`列：数据类型为`int64`，共有5110个非空值。

3. **数据类型统计**:
   - 总共有3列`float64`类型（即浮点数类型），4列`int64`类型（即整数类型），以及5列`object`类型（通常指字符串类型）。

4. **内存使用情况**:
   - 数据集大约使用了479.2 KB的内存。这个信息有助于了解数据集在存储和处理时可能对系统资源的需求。

通过这些信息，我们可以了解到数据集`data`的结构和基本属性。所有列除了`bmi`列有201个缺失值外，其他列都是完整的，没有缺失数据。这可能需要进一步的数据分析来确定缺失值的处理方法，例如通过插值、删除或使用其他技术来填补这些缺失值。
此外，数据集中包含的变量类型多样，有分类变量（如`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`）和数值变量（如`id`, `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `stroke`），这表明数据集可能用于回归分析、分类任务或其他机器学习算法。






下面这行代码使用了Seaborn库中的`heatmap`函数来创建一个热图，用于可视化Pandas DataFrame `data`中的缺失数据情况。下面是对这行代码的详细中文注释：

```python
# 导入Seaborn库，并使用别名sns。Seaborn是一个基于Matplotlib的高级数据可视化库，提供了丰富的绘图类型和美观的默认主题。
import seaborn as sns

# 调用DataFrame对象'data'的isnull()方法，该方法会返回一个与原DataFrame形状相同的布尔型DataFrame。
# 在这个布尔型DataFrame中，每个位置的值表示原DataFrame对应位置的值是否为空（True表示缺失，False表示非空）。
data.isnull()

# 使用Seaborn库的heatmap函数创建一个热图，输入参数为上述得到的布尔型DataFrame。
# 热图是一种数据可视化技术，可以显示矩阵数据中的数值大小，通常用颜色的深浅来表示。
sns.heatmap(data.isnull(),

    # cmap参数用于指定热图的颜色映射方案，'magma'是一种颜色映射方案，它会生成从紫色到黄色的颜色渐变。
    cmap = 'magma',

    # cbar参数用于控制是否显示颜色条，颜色条是热图旁边用于表示颜色深浅对应数值大小的图例。
    # 设置为False表示不显示颜色条。
    cbar = False
);

# 执行这行代码后，会在Python的绘图环境中生成一个热图，其中颜色越深（如紫色）表示数据越缺失，颜色越浅（如黄色）表示数据越完整。
# 通过这个热图，我们可以快速地识别出数据集中哪些位置存在缺失值，以及缺失数据的分布情况。
```

这段代码通过可视化的方式提供了数据集中缺失值的直观展示，有助于我们在进行数据分析或数据清洗之前，了解数据的完整性和需要采取的处理措施。


![1.2缺失值](01图片/1.2缺失值.png)

**A few null values** are present in the **bmi** feature!

这行代码是用于调用Pandas库中的DataFrame对象的`describe()`方法，以获取DataFrame的统计摘要。下面是对这行代码的详细中文注释：

```python
data.describe()
# 调用DataFrame对象'data'的describe()方法，该方法会输出DataFrame的统计摘要。
# 这个方法通常用于快速了解数据集的数值型列的分布情况，包括均值、标准差、最小值、25%分位数、中位数、75%分位数和最大值。
# 执行这行代码后，Pandas会打印出一份报告，其中包含了以下信息：
#   - 数据集中每一列的计数（即非空值的数量）。
#   - 每一列的平均值（mean），即所有数值的总和除以数值的个数。
#   - 每一列的标准差（std），表示数值分布的离散程度。
#   - 每一列的最小值（min），即所有数值中的最小值。
#   - 每一列的25%分位数（25%），即所有数值从小到大排列后位于25%位置的值。
#   - 每一列的中位数（50%），即所有数值从小到大排列后位于中间位置的值。
#   - 每一列的75%分位数（75%），即所有数值从小到大排列后位于75%位置的值。
#   - 每一列的最大值（max），即所有数值中的最大值。
data.describe()
```

当你执行`data.describe()`方法时，Pandas会输出一个表格，其中列出了DataFrame的每一列数值型数据的统计摘要。默认情况下，`describe()`方法只对数值型列进行统计分析，不包括分类数据（如字符串类型）。这个功能对于快速分析数据集的数值特征非常有用，可以帮助你了解数据的集中趋势、离散程度以及潜在的异常值。如果你需要对非数值型数据进行描述性统计分析，你可能需要使用其他方法或自定义函数来实现。

![1.3数据摘要](01图片/1.3数据摘要.png)

根据提供的`data.describe()`方法的输出结果，我们可以对数据集`data`中的数值型列进行以下详细分析：

1. **id**:
   - `count`: 共有5110个非空的id值。
   - `mean`: id的平均值为36517.83，这可能是某种形式的编码或者编号。
   - `std`: id的标准差为21161.72，表明id值的分布相对分散。
   - `min`: id的最小值为67，最大值为72940，这表明id的范围很广。
   - `25%`: 25%的id值低于17741.25。
   - `50%`: 中位数id值为36932，这是数据集中的中位编号。
   - `75%`: 75%的id值低于54682。
   - `max`: 最大id值为72940。

2. **age**:
   - `count`: 共有5110个非空的年龄值。
   - `mean`: 平均年龄为43.23岁。
   - `std`: 年龄的标准差为22.61岁，表明年龄分布有一定的离散程度。
   - `min`: 最小年龄为0.08岁，这可能是数据输入错误或特殊情形。
   - `25%`: 25%的人年龄在25岁以下。
   - `50%`: 中位数年龄为45岁。
   - `75%`: 75%的人年龄在61岁以上。
   - `max`: 最大年龄为82岁。

3. **hypertension**:
   - `count`: 共有5110个非空的高血压值。
   - `mean`: 平均值为0.10，表明大约10%的人有高血压。
   - `std`: 标准差为0.30，表明高血压数据的分布有一定的波动。
   - `min`: 最小值为0，表明没有人的高血压值为0（0可能表示没有高血压）。
   - `25%`: 25%的人高血压值为0。
   - `50%`: 中位数为0，表明一半的人没有高血压。
   - `75%`: 75%的人高血压值为0。
   - `max`: 最大值为1，表明有一部分人有高血压。

4. **heart_disease**:
   - `count`: 共有5110个非空的心脏病值。
   - `mean`: 平均值为0.05，表明大约5%的人有心脏病。
   - `std`: 标准差为0.23，表明心脏病数据的分布有一定的波动。
   - `min`: 最小值为0，表明没有人的心脏病值为0（0可能表示没有心脏病）。
   - `25%`: 25%的人心脏病值为0。
   - `50%`: 中位数为0，表明一半的人没有心脏病。
   - `75%`: 75%的人心脏病值为0。
   - `max`: 最大值为1，表明有一部分人有心脏病。

5. **avg_glucose_level**:
   - `count`: 共有5110个非空的平均血糖水平值。
   - `mean`: 平均血糖水平为106.15 mg/dL。
   - `std`: 标准差为45.28 mg/dL，表明血糖水平有一定的波动。
   - `min`: 最小血糖水平为55.12 mg/dL。
   - `25%`: 25%的人血糖水平低于77.25 mg/dL。
   - `50%`: 中位数血糖水平为91.88 mg/dL。
   - `75%`: 75%的人血糖水平低于114.09 mg/dL。
   - `max`: 最高血糖水平为271.74 mg/dL，可能表明有严重的高血糖情况。

6. **bmi**:
   - `count`: 共有4909个非空的身体质量指数（BMI）值，有201个缺失值。
   - `mean`: 平均BMI为28.89，属于过重范围（根据世界卫生组织的标准，18.5-24.9为正常范围）。
   - `std`: 标准差为7.85，表明BMI值分布有一定的离散程度。
   - `min`: 最小BMI为10.30，这可能是数据输入错误、极端情况或特殊人群（如运动员）。
   - `25%`: 25%的人BMI低于23.50。
   - `50%`: 中位数BMI为28.10。
   - `75%`: 75%的人BMI低于33.10。
   - `max`: 最大BMI为97.60，这可能表明有严重的肥胖情况。

7. **stroke**:
   - `count`: 共有5110个非空的中风值。
   - `mean`: 平均值为0.05，表明大约5%的人有中风病史。
   - `std`: 标准差为0.22，表明中风数据的分布有一定的波动。
   - `min`: 最小值为0，表明没有人的中风值为0（0可能表示没有中风）。
   - `25%`: 25%的人中风值为0。
   - `50%`: 中位数为0，表明一半的人没有中风。
   - `75%`: 75%的人中风值为0。
   - `max`: 最大值为1，表明有一部分人有中风病史。

通过这些统计摘要，我们可以了解到数据集中的一些基本特征和潜在的问题，如年龄和BMI的分布情况，以及高血压、心脏病、中风等健康状况的普遍性。这些信息对于进一步的数据分析和建模具有重要的参考价值。同时，我们也可以注意到，某些列如`id`和`age`的最小值异常低，可能需要进一步的数据清洗和验证。




下面这段代码使用了Pandas和Seaborn库来创建两个热图，分别展示有中风病史和无中风病史的数据集的统计描述。下面是对每行代码的详细中文注释：

```python
# 定义一个新的DataFrame 'stroke'，它通过筛选原始DataFrame 'data' 中 'stroke' 列值为1的行来创建。
# 这意味着 'stroke' DataFrame 只包含有中风病史的记录。
stroke = data[data['stroke'] == 1].describe().T

# 定义一个新的DataFrame 'no_stroke'，它通过筛选原始DataFrame 'data' 中 'stroke' 列值为0的行来创建。
# 这意味着 'no_stroke' DataFrame 只包含无中风病史的记录。
no_stroke = data[data['stroke'] == 0].describe().T

# 定义一个颜色列表 'colors'，包含两种颜色的十六进制代码，这些颜色将用于热图中不同的数据点。
colors = ['#3C1053','#DF6589']

# 使用Matplotlib的subplots函数创建一个包含两个子图的图形对象 'fig' 和轴对象 'ax'。
# nrows = 1 表示子图将垂直排列，ncols = 2 表示有两个子图并排排列，figsize = (5,5) 设置了图形的大小。
fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (5,5))

# 激活第一个子图（位置为1,2的第一个位置），并使用Seaborn的heatmap函数创建一个热图。
# 热图的数据来源于 'stroke' DataFrame的 'mean' 列。
# annot = True 表示在热图的每个单元格内显示数值。
# cmap = colors 表示使用之前定义的颜色映射。
# linewidths = 0.4 设置单元格之间线条的宽度。
# linecolor = 'black' 设置线条颜色为黑色。
# cbar = False 表示不显示颜色条。
# fmt = '.2f' 表示数值的格式化字符串，保留两位小数。
plt.subplot(1,2,1)
sns.heatmap(stroke[['mean']], annot = True, cmap = colors, linewidths = 0.4, linecolor = 'black', cbar = False, fmt = '.2f')
# 设置子图的标题为 'Stroke Suffered'。
plt.title('Stroke Suffered');

# 激活第二个子图（位置为1,2的第二个位置），并使用Seaborn的heatmap函数创建一个热图。
# 热图的数据来源于 'no_stroke' DataFrame的 'mean' 列。
plt.subplot(1,2,2)
# 同上，创建热图并设置相应的参数。
sns.heatmap(no_stroke[['mean']], annot = True, cmap = colors, linewidths = 0.4, linecolor = 'black', cbar = False, fmt = '.2f')
# 设置子图的标题为 'No Stroke Suffered'。
plt.title('No Stroke Suffered');

# 使用Matplotlib的tight_layout函数调整子图的布局，使子图之间填充更紧凑，pad = 0 设置了布局的填充间距。
fig.tight_layout(pad = 0)
```

这段代码通过创建两个热图来比较有中风病史和无中风病史患者的统计数据。通过这种可视化方式，可以直观地看出两组数据在各个数值特征上的差异，例如平均年龄、血糖水平、BMI等。这对于分析中风与其他健康指标之间的关联非常有用。此外，通过设置不同的颜色和格式化选项，热图的可读性和美观性得到了增强。

![1.4不同变量的均值](01图片/1.4不同变量的均值.png)


## 2. Data Balancing using SMOTE




## 3. Data Leakage




## 4. Statistical Tests for Feature Selection




## 5. Modeling and visualization of results for algorithms






















