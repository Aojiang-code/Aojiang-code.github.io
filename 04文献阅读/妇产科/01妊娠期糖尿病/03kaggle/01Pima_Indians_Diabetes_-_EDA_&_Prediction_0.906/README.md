# Pima Indians Diabetes - EDA & Prediction (0.906)

> 网址：[Pima Indians Diabetes - EDA & Prediction (0.906)](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906)


> 此代码用于 **jupyter notebook**


* Accuracy - 5 Folds - LightGBM : 89.8%
* Accuracy - 5 Folds - LightGBM & KNN : 90.6%

> Vincent Lugat
> 
> February 2019

## 目录

1. Load libraries and read the data

1.1. Load libraries

1.2. Read the data

2. Overview

2.1. Head

2.2. Target

2.3. Missing values

3. Replace missing values and EDA

3.1. Insulin

3.2. Glucose

3.3. SkinThickness

3.4. BloodPressure

3.5. BMI

4. New features (16) and EDA

5. Prepare dataset

5.1. StandardScaler and LabelEncoder

5.2. Correlation Matrix

5.3. X and y

5.4. Model Performance

5.5. Scores Table

6.Machine Learning


6.1. RandomSearch + LightGBM - Accuracy = 89.8%


6.3. GridSearch + LightGBM & KNN- Accuracy = 90.6%

6.4. LightGBM & KNN - Discrimination Threshold

7. Credits


## 0. 前言
Hello All !!

This notebook is a guide to end to end a complete study in machine learning with different concepts like :

* Completing missing values (most important part)
* Exploratory data analysis
* Creating new features (to increase accuracy)
* Encoding features
* Using LightGBM and optimize hyperparameters
* Adding a KNN to LGBM to beat 90% accuracy (voting classifier)

### Who is Pima Indians ?

"The Pima (or Akimel O'odham, also spelled Akimel O'otham, "River People", formerly known as Pima) are a group of Native Americans living in an area consisting of what is now central and southern Arizona. The majority population of the surviving two bands of the Akimel O'odham are based in two reservations: the Keli Akimel O'otham on the Gila River Indian Community (GRIC) and the On'k Akimel O'odham on the Salt River Pima-Maricopa Indian Community (SRPMIC)." Wikipedia


## 1. Load libraries and read the data


### 1.1. Load libraries
Loading the libraries

#### 原代码
```python
# Python libraries
# Classic,data manipulation and linear algebra
import pandas as pd
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import squarify

# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
import lightgbm as lgbm
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from yellowbrick.classifier import DiscriminationThreshold

# Stats
import scipy.stats as ss
from scipy import interp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Time
from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 
```
#### 添加中文注释

```python

# Python 库导入
# 经典库、数据操作和线性代数
import pandas as pd  # 数据分析和操作库
import numpy as np  # 数值计算库，用于处理大型多维数组和矩阵

# 图表绘制
import seaborn as sns  # 基于matplotlib的高级绘图库，用于数据可视化
import matplotlib.pyplot as plt  # matplotlib绘图库，用于创建图表和可视化数据
%matplotlib inline  # Jupyter Notebook中内联显示matplotlib图表
import plotly.offline as py  # Plotly离线绘图库，用于交互式图表
import plotly.graph_objs as go  # Plotly图形对象库，用于创建图表的元素
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot  # Plotly离线绘图相关函数
import plotly.tools as tls  # Plotly工具库，包含一些辅助函数
import plotly.figure_factory as ff  # Plotly图表生成器库，用于快速创建图表
py.init_notebook_mode(connected=True)  # 初始化Plotly的Notebook模式
import squarify  # 用于绘制矩形树图的库

# 数据处理、评估指标和建模
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 数据预处理库，用于标准化和编码
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV  # 模型选择库，用于交叉验证和参数搜索
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, roc_auc_score  # 评估库，用于计算各种评估指标
import lightgbm as lgbm  # LightGBM库，用于梯度提升决策树算法
from sklearn.ensemble import VotingClassifier  # 集成学习库，用于投票分类器
from sklearn.neighbors import KNeighborsClassifier  # 邻居分类库，用于K近邻算法
from sklearn.metrics import roc_curve, auc  # 评估库，用于计算接收者操作特征曲线和曲线下面积
from sklearn.model_selection import KFold  # 模型选择库，用于K折交叉验证
from sklearn.model_selection import cross_val_predict  # 模型选择库，用于交叉验证预测
from yellowbrick.classifier import DiscriminationThreshold  # Yellowbrick库，用于绘制分类器的判别阈值图

# 统计
import scipy.stats as ss  # SciPy统计库，提供各种统计函数
from scipy import interp  # SciPy插值库，用于数据插值
from scipy.stats import randint as sp_randint  # SciPy随机整数生成器
from scipy.stats import uniform as sp_uniform  # SciPy均匀分布随机数生成器

# 时间
from contextlib import contextmanager  # 上下文管理库，用于定义上下文相关的操作
@contextmanager
def timer(title):  # 定义一个计时器上下文管理器
    t0 = time.time()  # 开始计时
    yield  # 执行操作
    print("{} - done in {:.0f}s".format(title, time.time() - t0))  # 打印操作耗时

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息，保持输出的清洁
```

### 1.2. Read the data
Loading dataset with pandas (pd)

```python
data = pd.read_csv('../input/diabetes.csv')
```


## 2. Overview



### 2.1. Head
Checking data head and info

```python
display(data.info(),data.head())
```
#### 结果
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
Pregnancies                 768 non-null int64
Glucose                     768 non-null int64
BloodPressure               768 non-null int64
SkinThickness               768 non-null int64
Insulin                     768 non-null int64
BMI                         768 non-null float64
DiabetesPedigreeFunction    768 non-null float64
Age                         768 non-null int64
Outcome                     768 non-null int64
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
None
```

#### 结果解读
上述内容是一个Python环境中使用pandas库对数据集进行查看和展示的示例。具体解释如下：

首先，使用`data.info()`函数展示了数据集的概要信息，包括每列的名称、非空值的数量、数据类型等。接着，使用`data.head()`函数显示了数据集的前几行，通常是前五行。

- `Pregnancies`列包含768个非空的整数值，表示孕妇的怀孕次数。
- `Glucose`列包含768个非空的整数值，表示葡萄糖耐量测试的结果。
- `BloodPressure`列包含768个非空的整数值，表示孕妇的血压。
- `SkinThickness`列包含768个非空的整数值，表示皮肤的厚度。
- `Insulin`列包含768个非空的整数值，表示胰岛素水平。
- `BMI`列包含768个非空的浮点数值，表示孕妇的身体质量指数。
- `DiabetesPedigreeFunction`列包含768个非空的浮点数值，表示糖尿病家族史的一个计算指标。
- `Age`列包含768个非空的整数值，表示孕妇的年龄。
- `Outcome`列包含768个非空的整数值，表示糖尿病的发病结果，通常用于作为分类的目标变量。

数据类型方面，有两列是浮点型（`float64`），其余七列是整型（`int64`）。整型数据占用的空间较小，而浮点型数据可以表示小数。

内存使用方面，整个数据集占用了54.1 KB，这是一个相对较小的数据集。

`RangeIndex: 768 entries, 0 to 767`表明数据集共有768行数据，索引范围从0到767。这通常意味着数据集没有缺失行。


The datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

#### What is diabetes ?
Acccording to NIH, "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy. Sometimes your body doesn’t make enough—or any—insulin or doesn’t use insulin well. Glucose then stays in your blood and doesn’t reach your cells.

Over time, having too much glucose in your blood can cause health problems. Although diabetes has no cure, you can take steps to manage your diabetes and stay healthy.

Sometimes people call diabetes “a touch of sugar” or “borderline diabetes.” These terms suggest that someone doesn’t really have diabetes or has a less serious case, but every case of diabetes is serious.

##### **What are the different types of diabetes?** 

The most common types of diabetes are type 1, type 2, and gestational diabetes.

###### **Type 1 diabetes **

If you have type 1 diabetes, your body does not make insulin. Your immune system attacks and destroys the cells in your pancreas that make insulin. Type 1 diabetes is usually diagnosed in children and young adults, although it can appear at any age. People with type 1 diabetes need to take insulin every day to stay alive.

###### **Type 2 diabetes**

 If you have type 2 diabetes, your body does not make or use insulin well. You can develop type 2 diabetes at any age, even during childhood. However, this type of diabetes occurs most often in middle-aged and older people. Type 2 is the most common type of diabetes.

###### **Gestational diabetes**

 Gestational diabetes develops in some women when they are pregnant. Most of the time, this type of diabetes goes away after the baby is born. However, if you’ve had gestational diabetes, you have a greater chance of developing type 2 diabetes later in life. Sometimes diabetes diagnosed during pregnancy is actually type 2 diabetes.


###### **Other types of diabetes**

 Less common types include monogenic diabetes, which is an inherited form of diabetes, and cystic fibrosis-related diabetes ."



### 2.2. Target
What's target's distribution ?

The above graph shows that the data is unbalanced. The number of non-diabetic is 268 the number of diabetic patients is 500
#### 源代码
```python
# 2 datasets
D = data[(data['Outcome'] != 0)]
H = data[(data['Outcome'] == 0)]

#------------COUNT-----------------------
def target_count():
    trace = go.Bar( x = data['Outcome'].value_counts().values.tolist(), 
                    y = ['healthy','diabetic' ], 
                    orientation = 'h', 
                    text=data['Outcome'].value_counts().values.tolist(), 
                    textfont=dict(size=15),
                    textposition = 'auto',
                    opacity = 0.8,marker=dict(
                    color=['lightskyblue', 'gold'],
                    line=dict(color='#000000',width=1.5)))

    layout = dict(title =  'Count of Outcome variable')

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)

#------------PERCENTAGE-------------------
def target_percent():
    trace = go.Pie(labels = ['healthy','diabetic'], values = data['Outcome'].value_counts(), 
                   textfont=dict(size=15), opacity = 0.8,
                   marker=dict(colors=['lightskyblue', 'gold'], 
                               line=dict(color='#000000', width=1.5)))


    layout = dict(title =  'Distribution of Outcome variable')

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)
```


```python
target_count()
target_percent()
```


#### 添加中文注释
```python
# 定义两个数据集
# D 为Outcome列不为0的数据集，即糖尿病患者的数据
D = data[(data['Outcome'] != 0)]
# H 为Outcome列为0的数据集，即健康人的数据
H = data[(data['Outcome'] == 0)]

#------------计数---------------------
# 定义一个函数，该函数用于绘制目标变量的计数直方图
def target_count():
    # 创建一个条形图对象，其中x轴代表Outcome的计数，y轴代表标签，orientation为'h'表示条形图是垂直显示的
    trace = go.Bar(
        x=data['Outcome'].value_counts().values.tolist(),  # x轴数据是Outcome列的计数，将其转换为列表形式
        y=['healthy', 'diabetic'],  # y轴标签，分别代表健康和糖尿病两类
        orientation='h',  # 设置条形图为垂直方向
        text=data['Outcome'].value_counts().values.tolist(),  # 条形图上显示的文本是Outcome的计数数值
        textfont=dict(size=15),  # 设置文本的字体大小为15
        textposition='auto',  # 文本位置自动调整以适应图表
        opacity=0.8,  # 设置图形的不透明度为0.8
        marker=dict(  # 定义标记的样式
            color=['lightskyblue', 'gold'],  # 标记颜色分别为天蓝色和金色，代表不同的类别
            line=dict(color='#000000', width=1.5)  # 标记边框颜色为黑色，宽度为1.5
        )
    )

    # 设置图表的布局，包括标题
    layout = dict(title='目标变量的计数')  # 图表标题设置为“目标变量的计数”

    # 创建图表对象，包含数据和布局
    fig = dict(data=[trace], layout=layout)  # fig是一个字典，包含数据和布局信息

    # 使用Plotly的iplot函数在Jupyter Notebook中绘制图表
    py.iplot(fig)  # 调用iplot函数展示图表


#上面这段代码定义了一个名为`target_count`的函数，用于生成并展示一个条形图，该图表显示了数据集中目标变量（Outcome）的计数。图表中，'healthy'和'diabetic'分别代表没有糖尿病和有糖尿病的两类数据。每个条形上的文本显示了每个类别的计数数值，颜色分别设置为天蓝色和金色，以区分不同的类别。图表的不透明度设置为0.8，增加了视觉效果。最后，使用Plotly的`iplot`函数在Jupyter Notebook中展示图表。

#------------百分比---------------------
# 定义一个函数，该函数用于绘制目标变量的分布饼图
def target_percent():
    # 创建一个饼图对象，labels为标签，values为对应的值
    trace = go.Pie(
        labels=['healthy', 'diabetic'],  # 设置饼图的标签，分别为'healthy'和'diabetic'
        values=data['Outcome'].value_counts(),  # values为Outcome列的计数，代表每个类别的数量
        textfont=dict(size=15),  # 设置饼图上文本的字体大小为15
        opacity=0.8,  # 设置饼图的透明度为0.8
        # 设置标记的样式
        marker=dict(
            colors=['lightskyblue', 'gold'],  # 标记颜色分别为天蓝色和金色，代表不同的类别
            line=dict(color='#000000', width=1.5)  # 标记边框颜色为黑色，宽度为1.5
        )
    )

    # 设置图表布局，包括标题
    layout = dict(title='目标变量的分布')  # 设置图表的标题为“目标变量的分布”

    # 创建图表对象并包含数据和布局
    fig = dict(data=[trace], layout=layout)  # 创建一个包含数据和布局信息的图表对象

    # 使用Plotly的iplot函数在Jupyter Notebook中绘制图表
    py.iplot(fig)  # 调用iplot函数展示饼图

# 调用函数绘制图表
target_count()  # 调用target_count函数绘制目标变量的计数直方图
target_percent()  # 调用target_percent函数绘制目标变量的分布饼图


#这段代码定义了一个名为`target_percent`的函数，用于生成并展示一个饼图，该图表显示了数据集中目标变量（Outcome）的分布情况。图表中，'healthy'和'diabetic'分别代表没有糖尿病和有糖尿病的两类数据。每个扇区的文本显示了Outcome的计数，颜色分别设置为天蓝色和金色，以区分不同的类别。图表的透明度设置为0.8，增加了视觉效果。最后，使用Plotly的`iplot`函数在Jupyter Notebook中展示图表。函数定义完成后，通过调用`target_count()`和`target_percent()`函数来绘制并展示相应的图表。
```

上述代码中，首先定义了两个数据集`D`和`H`，分别代表糖尿病患者和健康人的数据。然后定义了两个函数`target_count()`和`target_percent()`，用于绘制目标变量（糖尿病Outcome）的计数直方图和分布饼图。最后调用这两个函数来展示糖尿病患者和健康人的分布情况。







### 2.3. Missing values
We saw on data.head() that some features contain 0, it doesn't make sense here and this indicates missing value Below we replace 0 value by NaN :

```python
# 将数据集中'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'列中的0值替换为NaN
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
```

这行代码的作用是在整个`data` DataFrame中，针对'Glucose'（葡萄糖）、'BloodPressure'（血压）、'SkinThickness'（皮肤厚度）、'Insulin'（胰岛素）和'BMI'（身体质量指数）这几列，将所有的0值替换为`np.NaN`，即“不是一个数字”的浮点数值，这通常用于表示缺失数据。这样的替换可能是为了后续的数据处理步骤，比如在数据分析中通常会忽略或填充缺失值。

Now, we can look at where are missing values :


```python
# Define missing plot to detect all missing values in dataset
# 定义一个函数用于绘制数据集中所有缺失值的图表
def missing_plot(dataset, key) :
    # 计算每一列非空值的数量，并创建一个DataFrame来存储这些计数
    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])
    # 计算每一列缺失值的比例，并创建一个DataFrame来存储这些比例
    percentage_null = pd.DataFrame((len(dataset[key]) - (len(dataset[key]) - dataset.isnull().sum()))/len(dataset[key]) * 100, columns = ['Count'])
    # 将比例四舍五入到小数点后两位
    percentage_null = percentage_null.round(2)

    # 创建一个条形图对象，x轴为列名，y轴为缺失值的计数，文本显示为缺失值的比例
    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, text = percentage_null['Count'], 
                  textposition = 'auto', marker=dict(color = '#7EC0EE',  # 设置条形图颜色为浅蓝色
                      line=dict(color='#000000', width=1.5)))  # 设置条形图边框为黑色，宽度为1.5

    # 设置图表布局，包括标题
    layout = dict(title =  "缺失值（计数 & 百分比）")

    # 创建图表对象并包含数据和布局
    fig = dict(data = [trace], layout=layout)
    # 使用Plotly的iplot函数在Jupyter Notebook中绘制图表
    py.iplot(fig)
```

这段代码定义了一个名为`missing_plot`的函数，用于生成并展示一个条形图，该图表显示了数据集中每一列的缺失值数量及其占比。函数接收两个参数：`dataset`表示数据集，`key`表示要检查的列名。首先，计算每一列的非空值数量和缺失值比例，并将这些信息存储在两个DataFrame中。然后，创建一个条形图对象，其中x轴为列名，y轴为缺失值的计数，文本显示为缺失值的比例。最后，使用Plotly的`iplot`函数在Jupyter Notebook中展示图表。

```python
# Plotting 
# 绘制数据集中'Outcome'列的缺失值图表
missing_plot(data, 'Outcome')
```

这行代码调用了`missing_plot`函数，用于绘制数据集`data`中`'Outcome'`列的缺失值情况。这个函数将展示该列中缺失值的数量和所占的百分比，以条形图的形式表现。这样的可视化有助于了解目标变量`'Outcome'`的完整性，从而评估数据集的质量和进行后续的数据处理决策。

## 3. Replace missing values and EDA

### 3.1. Insulin


### 3.2. Glucose


### 3.3. SkinThickness


### 3.4. BloodPressure


### 3.5. BMI




## 4. New features (16) and EDA



## 5. Prepare dataset



### 5.1. StandardScaler and LabelEncoder



### 5.2. Correlation Matrix


### 5.3. X and y


### 5.4. Model Performance


### 5.5. Scores Table


### 6.Machine Learning

### 6.1. RandomSearch + LightGBM - Accuracy = 89.8%


### 6.2. LightGBM - Discrimination Threshold


### 6.3. GridSearch + LightGBM & KNN- Accuracy = 90.6%


### 6.4. LightGBM & KNN - Discrimination Threshold



## 7. Credits

