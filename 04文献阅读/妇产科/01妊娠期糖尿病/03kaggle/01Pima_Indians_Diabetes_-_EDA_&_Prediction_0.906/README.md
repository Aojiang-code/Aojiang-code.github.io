# Pima Indians Diabetes - EDA & Prediction (0.906)

> 网址：[Pima Indians Diabetes - EDA & Prediction (0.906)](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906)


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


## 2. Overview



### 2.1. Head


### 2.2. Target


### 2.3. Missing values


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

