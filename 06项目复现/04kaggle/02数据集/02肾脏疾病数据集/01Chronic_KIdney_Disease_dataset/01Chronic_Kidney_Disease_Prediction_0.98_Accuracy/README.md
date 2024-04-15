# Chronic_Kidney_Disease_Prediction_(98%_Accuracy)

![0封面](01图片/0.0封面.png)

## 0. Table of Contents
* EDA
* Data Pre Processing
* Feature Encoding
* 
* Model Building
* * Knn
* * Decision Tree Classifier
* * Random Forest Classifier
* * Ada Boost Classifier
* * Gradient Boosting Classifier
* * Stochastic Gradient Boosting (SGB)
* * XgBoost
* * Cat Boost Classifier
* * Extra Trees Classifier
* * LGBM Classifier
* 
* Models Comparison

## 导入包

下面这段代码包含了一些用于数据操作和可视化的Python库的导入语句，以及一些环境配置的设置。以下是对每行代码的详细中文注释：

```python
# 导入pandas库，并使用别名pd。Pandas是一个强大的数据结构和数据分析工具，提供了DataFrame等数据结构。
import pandas as pd

# 导入numpy库，并使用别名np。Numpy是用于进行科学计算的库，提供了多维数组对象和一系列处理数组的函数。
import numpy as np

# 导入matplotlib.pyplot模块，并使用别名plt。这个模块是matplotlib库的一部分，提供了MATLAB风格的绘图接口。
import matplotlib.pyplot as plt

# 导入seaborn库，并使用别名sns。Seaborn是基于matplotlib的高级数据可视化库，提供了更多样化的绘图风格和接口。
import seaborn as sns

# 导入plotly.express模块，使用别名px。Plotly是一个交互式图表库，plotly.express是其快速绘图的接口。
import plotly.express as px

# 导入warnings模块，用于控制警告信息的显示。
import warnings

# 使用filterwarnings函数忽略所有的警告信息。这可以在开发过程中减少不必要的警告信息的干扰。
warnings.filterwarnings('ignore')

# 使用plt.style.use函数设置绘图风格为'fivethirtyeight'，这是基于FiveThirtyEight新闻网站风格的绘图风格。
plt.style.use('fivethirtyeight')

# 使用%matplotlib inline魔法命令，这通常在Jupyter Notebook中使用，用于在Notebook内部直接显示matplotlib生成的图表。
%matplotlib inline

# 设置pandas的显示选项，使得DataFrame可以显示更多的列。这里设置为最多显示26列。
pd.set_option('display.max_columns', 26)
```

执行这段代码后，Python环境将配置好用于数据分析和可视化所需的库和风格。这对于后续的数据操作、统计分析和数据可视化非常有用。特别是在Jupyter Notebook中，这些设置可以提高工作效率，使得数据探索和结果展示更加直观和美观。


下面这段代码用于加载存储在CSV文件中的肾脏疾病数据集，并显示数据集的前几行。以下是对每行代码的详细中文注释：

```python
# 加载数据

# 使用pandas库的read_csv函数读取位于'../input/ckdisease/'路径下的'kidney_disease.csv'文件，并将其内容存储到DataFrame对象df中。
# '../input/ckdisease/'是文件的存放路径，可能是相对于当前工作目录的上级目录中的'input'文件夹下的'ckdisease'文件夹。
# 'kidney_disease.csv'是包含肾脏疾病数据的CSV文件名。
df = pd.read_csv('../input/ckdisease/kidney_disease.csv')

# 调用DataFrame对象df的head方法，显示其前五行数据。
# head方法默认显示前五行，但如果需要可以传入一个参数指定显示的行数。
# 这通常用于快速查看数据集的结构和前几行的样本数据，以便于进行初步的数据探索。
df.head()
```

执行这段代码后，会在Python环境中输出名为`df`的DataFrame的前五行数据。这有助于用户了解肾脏疾病数据集的基本信息，例如数据列的名称、数据类型、缺失值情况等。在数据分析和机器学习的前期阶段，这种快速的数据预览是非常重要的，因为它可以帮助用户确定后续数据处理和分析的方向。


![0.1head](01图片/0.1head.png)


下面这行代码用于获取DataFrame `df`的维度信息。以下是对这行代码的详细中文注释：

```python
# 获取DataFrame df的行数和列数。
# df.shape是一个属性，返回一个元组，其中第一个元素是行数（即DataFrame中的行数），第二个元素是列数（即DataFrame中的列数）。
# 这可以帮助用户了解数据集的大小，即有多少个样本和特征。
df.shape
```

执行这段代码后，会在Python环境中输出一个元组，其中包含两个整数值。第一个值表示数据集中的样本数量（通常称为观测数），第二个值表示数据集中的特征数量（通常称为变量数或列数）。这个信息对于理解数据集的规模和结构非常重要，特别是在进行数据分析和机器学习任务时，了解数据集的大小可以帮助决定适当的数据处理方法和模型选择。


```python
(400, 26)
```

执行`df.shape`后得到的结果是`(400, 26)`，这表示DataFrame `df`包含400行和26列。以下是对这个结果的分析：

1. **行数（400）**：这意味着数据集中有400个观测记录，或者可以理解为有400个个体或实例的数据。每一行代表一个观测，例如可能是一个病人的医疗记录、一项实验的观测结果或者一个时间序列的数据点。

2. **列数（26）**：这意味着数据集中共有26个特征或变量。每一列代表一个特定的属性或度量，例如可能包括年龄、性别、血压、血糖水平等医疗指标，或者是与研究主题相关的其他类型的数据。

这个结果提供了数据集的基本维度信息，有助于我们了解数据集的规模。在数据分析和机器学习的上下文中，了解数据集的大小对于确定分析方法、设计特征工程策略、选择合适的模型以及评估模型性能都是非常重要的。例如，如果数据集较小，我们可能需要更谨慎地选择特征以避免过拟合；如果特征数量较多，我们可能需要进行特征选择或降维处理以提高模型的效率和性能。



下面这段代码用于从DataFrame `df`中删除名为`id`的列，并更新原始DataFrame。以下是对这行代码的详细中文注释：

```python
# 删除id列

# 使用DataFrame的drop方法从df中删除名为'id'的列。
# 'id'是列名，它是我们希望从数据集中移除的列的标识符。
# axis=1指定了要删除的轴向，1代表列（axis=0代表行）。
# inplace=True意味着更改将直接应用到原始的DataFrame上，而不是创建一个新的DataFrame副本。
# 如果inplace设置为False（默认值），则drop操作会返回一个新的DataFrame，而原始的df不会改变。
df.drop('id', axis=1, inplace=True)
```

执行这段代码后，`df`中的`id`列将被删除，且这个更改会直接作用于原始的DataFrame。这种操作在数据分析中很常见，尤其是在预处理阶段，当我们认为某些列对于分析或建模没有帮助或者有隐私风险时，我们会选择删除这些列。在这个例子中，可能是因为`id`列是一个唯一标识符，对于数据分析和机器学习任务没有实际用途，因此选择将其移除。需要注意的是，`inplace=True`参数确保了更改是永久性的，因此在执行这一操作前应确保这是我们想要的结果。



下面这段代码用于将DataFrame `df`中的列名重命名为更友好、更易于理解的名称。以下是对这行代码的详细中文注释：

```python
# 重命名列名，使其更加用户友好

# 将df的列名重新赋值为新的列名列表。
# 这个列表包含了所有的新列名，顺序需要与原始DataFrame中的列顺序相匹配。
# 例如，原始DataFrame中的第一列将被重命名为'age'，第二列将被重命名为'blood_pressure'，以此类推。
# 这样的重命名有助于清晰地识别每一列数据的含义，便于后续的数据分析和模型构建。
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']
```

执行这段代码后，`df`中的每个列将被赋予一个新的名称，这些名称更加直观地描述了每一列数据的特征。例如，`blood_pressure`列可能包含了血压的测量值，`diabetes_mellitus`列可能标识了是否患有糖尿病等。这种重命名操作是数据预处理的一个重要步骤，它有助于提高数据集的可读性，使得数据集的结构和内容更容易被理解和使用。在进行数据分析、数据可视化或构建机器学习模型时，清晰的列名可以帮助分析师和开发者更快地理解和处理数据。


```python
df.head()
```

![0.2head](01图片/0.2head.png)



下面这行代码用于生成DataFrame `df`的描述性统计摘要。以下是对这行代码的详细中文注释：

```python
# 生成DataFrame df的描述性统计摘要。

# 使用DataFrame的describe方法来获取数据的统计概览。
# describe方法默认计算数值型列的统计信息，包括均值、标准差、最小值、25%分位数、中位数、75%分位数和最大值。
# 这有助于快速了解数据的分布情况，例如中心趋势、离散程度和潜在的异常值。
df.describe()
```

执行这段代码后，会在Python环境中输出一个表格，其中包含了DataFrame中每个数值型列的描述性统计信息。这个摘要对于初步了解数据特征非常有用，可以帮助用户识别数据的集中趋势、变异性以及可能的异常值。例如，均值提供了数据的平均水平，标准差反映了数据的离散程度，而最大值和最小值可以帮助识别数据的范围和可能的异常值。这些统计信息是数据分析和机器学习任务中不可或缺的一部分，它们为数据清洗、特征工程和模型训练提供了重要的参考依据。

![0.3describe](01图片/0.3describe.png)

下面执行`df.describe()`后得到的结果显示了数据集中数值型列的描述性统计摘要。以下是对结果的详细分析：

1. **年龄（age）**:
   - `count`: 有391个有效数据点。
   - `mean`: 平均年龄为51.48岁。
   - `std`: 标准差为17.17岁，表明年龄分布有一定的离散程度。
   - `min`: 最小年龄为2岁，这可能是输入错误或特殊情况。
   - `25%`: 25%的个体年龄在42岁以下。
   - `50%`: 中位数年龄为55岁。
   - `75%`: 75%的个体年龄在64.5岁以下。
   - `max`: 最大年龄为90岁。

2. **血压（blood_pressure）**:
   - `count`: 有388个有效数据点。
   - `mean`: 平均血压为76.47 mmHg。
   - `std`: 标准差为13.68 mmHg，表明血压值有一定的波动。
   - `min`: 最低血压为50 mmHg。
   - `25%`: 25%的个体血压在70 mmHg以下。
   - `50%`: 中位数血压为80 mmHg。
   - `75%`: 75%的个体血压在80 mmHg以下。
   - `max`: 最高血压为180 mmHg，这可能表明有高血压的情况。

3. **比重（specific_gravity）**:
   - `count`: 有353个有效数据点。
   - `mean`: 平均比重为1.0174。
   - `std`: 标准差为0.00572，比重的分布相对集中。
   - `min`: 最低比重为1.005。
   - `25%`: 25%的个体比重在1.01以下。
   - `50%`: 中位数比重为1.02。
   - `75%`: 75%的个体比重在1.02以下。
   - `max`: 最大比重为1.025。

4. **白蛋白（albumin）**:
   - `count`: 有354个有效数据点。
   - `mean`: 平均白蛋白水平为1.0169。
   - `std`: 标准差为1.3527，白蛋白水平的分布较为分散。
   - `min`: 最低白蛋白水平为0，这可能是由于测量误差或病理状态。
   - `25%`: 25%的个体白蛋白水平在0以下。
   - `50%`: 中位数白蛋白水平为1.017。
   - `75%: 75%的个体白蛋白水平在2以下。
   - `max`: 最高白蛋白水平为5，这可能是由于测量误差或病理状态。

其他列（如血糖、尿素、肌酐等）也提供了类似的统计信息，可以帮助我们了解数据的分布情况。这些统计摘要对于初步了解数据特征、识别异常值和进行后续的数据分析非常重要。例如，血压和白蛋白的分布可能需要进一步的医学解释，因为它们的异常值可能与健康状况有关。此外，数据的缺失情况（如某些列的`count`值小于总样本数）也需要关注，因为它们可能影响分析结果的准确性。









下面这行代码用于获取DataFrame `df`的详细内容信息，包括每列的数据类型、非空值的数量以及内存使用情况。以下是对这行代码的详细中文注释：

```python
# 获取DataFrame df的详细内容信息。

# 使用DataFrame的info方法输出数据集的详细报告。
# info方法提供了关于DataFrame的有用信息，如每列的名称、非空值计数、数据类型以及每列的内存使用情况。
# 这个报告对于理解数据集的结构和内容非常有用，特别是在进行数据清洗和预处理时。
df.info()
```

执行这段代码后，会在Python环境中输出DataFrame `df`的详细内容信息。输出结果通常包括以下几个部分：

1. **列名**：每列的名称。
2. **非空值计数**：每列中非空（即有效）值的数量。
3. **数据类型**：每列的数据类型，例如整数（int64）、浮点数（float64）或对象（object，通常指字符串）。
4. **内存使用情况**：DataFrame使用的内存量，通常以字节为单位。

这个信息对于数据分析师和数据科学家来说非常重要，因为它可以帮助他们了解数据集的完整性、数据类型是否正确以及是否存在潜在的内存使用问题。例如，如果某列的数据类型不是预期的类型，可能需要进行数据类型转换。如果某列有很多缺失值，可能需要进一步的数据清洗或分析来确定缺失值的处理策略。此外，了解内存使用情况有助于在处理大型数据集时优化资源分配和提高计算效率。

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 25 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   age                      391 non-null    float64
 1   blood_pressure           388 non-null    float64
 2   specific_gravity         353 non-null    float64
 3   albumin                  354 non-null    float64
 4   sugar                    351 non-null    float64
 5   red_blood_cells          248 non-null    object 
 6   pus_cell                 335 non-null    object 
 7   pus_cell_clumps          396 non-null    object 
 8   bacteria                 396 non-null    object 
 9   blood_glucose_random     356 non-null    float64
 10  blood_urea               381 non-null    float64
 11  serum_creatinine         383 non-null    float64
 12  sodium                   313 non-null    float64
 13  potassium                312 non-null    float64
 14  haemoglobin              348 non-null    float64
 15  packed_cell_volume       330 non-null    object 
 16  white_blood_cell_count   295 non-null    object 
 17  red_blood_cell_count     270 non-null    object 
 18  hypertension             398 non-null    object 
 19  diabetes_mellitus        398 non-null    object 
 20  coronary_artery_disease  398 non-null    object 
 21  appetite                 399 non-null    object 
 22  peda_edema               399 non-null    object 
 23  aanemia                  399 non-null    object 
 24  class                    400 non-null    object 
dtypes: float64(11), object(14)
memory usage: 78.2+ KB
```




执行`df.info()`后得到的结果显示了DataFrame `df`的结构和内容信息。以下是对结果的详细分析：

1. **DataFrame 类型**:
   - `<class 'pandas.core.frame.DataFrame'>`: 表明`df`是一个Pandas的DataFrame对象。

2. **索引范围**:
   - `RangeIndex: 400 entries, 0 to 399`: 表明DataFrame有400个条目，索引从0到399。

3. **数据列信息**:
   - 总共有25列数据。
   - 每列的名称、非空值的数量和数据类型都被列出。

4. **非空值计数和数据类型**:
   - `age`列有391个非空值，数据类型为`float64`。
   - `blood_pressure`列有388个非空值，数据类型为`float64`。
   - `specific_gravity`列有353个非空值，数据类型为`float64`。
   - `albumin`列有354个非空值，数据类型为`float64`。
   - `sugar`列有351个非空值，数据类型为`float64`。
   - `red_blood_cells`列有248个非空值，数据类型为`object`（通常指字符串）。
   - `pus_cell`列有335个非空值，数据类型为`object`。
   - `pus_cell_clumps`列有396个非空值，数据类型为`object`。
   - `bacteria`列有396个非空值，数据类型为`object`。
   - 其他列的数据信息以此类推。

5. **数据类型统计**:
   - `dtypes: float64(11), object(14)`: 表明有11列的数据类型是`float64`，14列的数据类型是`object`。

6. **内存使用情况**:
   - `memory usage: 78.2+ KB`: 表明DataFrame使用的内存量约为78.2KB。

从这个分析中，我们可以了解到数据集中存在一些缺失值，特别是`red_blood_cells`、`pus_cell`、`serum_creatinine`等列，它们的非空值数量少于总样本数。这可能意味着需要进一步的数据清洗工作，例如填充缺失值或删除含有缺失值的行。此外，大部分数值型数据使用`float64`类型，而一些可能是分类变量的数据使用`object`类型，这可能需要进一步的数据类型转换或处理。最后，内存使用情况显示数据集不是特别大，应该不会对大多数现代计算机的内存造成压力。




As we can see that 'packed_cell_volume', 'white_blood_cell_count' and 'red_blood_cell_count' are object type. We need to change them to numerical dtype.


下面这段代码用于将DataFrame `df`中的某些列转换为数值型数据，如果转换过程中遇到无法转换为数值的值，则将其设置为NaN。以下是对每行代码的详细中文注释：

```python
# 将必要的列转换为数值型

# 使用pd.to_numeric函数尝试将'packed_cell_volume'列转换为数值型。
# 如果在转换过程中遇到无法转换为数值的值，则使用errors='coerce'参数将这些值设置为NaN。
# 转换后的数值型数据将替换原来的'packed_cell_volume'列。
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')

# 使用pd.to_numeric函数尝试将'white_blood_cell_count'列转换为数值型。
# 同样的，如果转换过程中遇到无法转换的值，则使用errors='coerce'参数将这些值设置为NaN。
# 转换后的数值型数据将替换原来的'white_blood_cell_count'列。
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')

# 使用pd.to_numeric函数尝试将'red_blood_cell_count'列转换为数值型。
# 遇到无法转换的值时，同样使用errors='coerce'参数将这些值设置为NaN。
# 转换后的数值型数据将替换原来的'red_blood_cell_count'列。
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')
```

执行这段代码后，`df`中的`packed_cell_volume`、`white_blood_cell_count`和`red_blood_cell_count`列将被转换为数值型数据。如果这些列中原本包含非数值型的文本（例如“高”、“低”或其他描述性文本），使用`errors='coerce'`参数可以确保这些无法转换的值被设置为NaN，而不是引发错误。这样的处理对于数据清洗和后续的数据分析非常重要，因为数值型数据是进行数学计算和统计分析的基础。同时，将无法转换的值设置为NaN也有助于后续进一步处理这些缺失值，例如通过填充或删除含有缺失值的记录。




下面这行代码用于获取DataFrame `df`的详细内容信息，包括每列的数据类型、非空值的数量以及内存使用情况。以下是对这行代码的详细中文注释：

```python
# 获取DataFrame df的详细内容信息。

# 使用DataFrame的info方法输出数据集的详细报告。
# info方法提供了关于DataFrame的有用信息，如每列的名称、非空值计数、数据类型以及每列的内存使用情况。
# 这个报告对于理解数据集的结构和内容非常有用，特别是在进行数据清洗和预处理时。
df.info()
```

执行这段代码后，会在Python环境中输出DataFrame `df`的详细内容信息。输出结果通常包括以下几个部分：

1. **列名**：每列的名称。
2. **非空值计数**：每列中非空（即有效）值的数量。
3. **数据类型**：每列的数据类型，例如整数（int64）、浮点数（float64）或对象（object，通常指字符串）。
4. **内存使用情况**：DataFrame使用的内存量，通常以字节为单位。

这个信息对于数据分析师和数据科学家来说非常重要，因为它可以帮助他们了解数据集的完整性、数据类型是否正确以及是否存在潜在的内存使用问题。例如，如果某列的数据类型不是预期的类型，可能需要进行数据类型转换。如果某列有很多缺失值，可能需要进一步的数据清洗或分析来确定缺失值的处理策略。此外，了解内存使用情况有助于在处理大型数据集时优化资源分配和提高计算效率。


```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 25 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   age                      391 non-null    float64
 1   blood_pressure           388 non-null    float64
 2   specific_gravity         353 non-null    float64
 3   albumin                  354 non-null    float64
 4   sugar                    351 non-null    float64
 5   red_blood_cells          248 non-null    object 
 6   pus_cell                 335 non-null    object 
 7   pus_cell_clumps          396 non-null    object 
 8   bacteria                 396 non-null    object 
 9   blood_glucose_random     356 non-null    float64
 10  blood_urea               381 non-null    float64
 11  serum_creatinine         383 non-null    float64
 12  sodium                   313 non-null    float64
 13  potassium                312 non-null    float64
 14  haemoglobin              348 non-null    float64
 15  packed_cell_volume       329 non-null    float64
 16  white_blood_cell_count   294 non-null    float64
 17  red_blood_cell_count     269 non-null    float64
 18  hypertension             398 non-null    object 
 19  diabetes_mellitus        398 non-null    object 
 20  coronary_artery_disease  398 non-null    object 
 21  appetite                 399 non-null    object 
 22  peda_edema               399 non-null    object 
 23  aanemia                  399 non-null    object 
 24  class                    400 non-null    object 
dtypes: float64(14), object(11)
memory usage: 78.2+ KB
```


下面执行`df.info()`后得到的结果显示了DataFrame `df`的结构和内容信息。以下是对结果的详细分析：

1. **DataFrame 类型**:
   - `<class 'pandas.core.frame.DataFrame'>`: 表明`df`是一个Pandas的DataFrame对象。

2. **索引范围**:
   - `RangeIndex: 400 entries, 0 to 399`: 表明DataFrame有400个条目，索引从0到399。

3. **数据列信息**:
   - 总共有25列数据。
   - 每列的名称、非空值的数量和数据类型都被列出。

4. **非空值计数和数据类型**:
   - `age`列有391个非空值，数据类型为`float64`。
   - `blood_pressure`列有388个非空值，数据类型为`float64`。
   - `specific_gravity`列有353个非空值，数据类型为`float64`。
   - `albumin`列有354个非空值，数据类型为`float64`。
   - `sugar`列有351个非空值，数据类型为`float64`。
   - `red_blood_cells`列有248个非空值，数据类型为`object`（通常指字符串）。
   - `pus_cell`列有335个非空值，数据类型为`object`。
   - `pus_cell_clumps`列有396个非空值，数据类型为`object`。
   - `bacteria`列有396个非空值，数据类型为`object`。
   - `blood_glucose_random`列有356个非空值，数据类型为`float64`。
   - `blood_urea`列有381个非空值，数据类型为`float64`。
   - `serum_creatinine`列有383个非空值，数据类型为`float64`。
   - `sodium`列有313个非空值，数据类型为`float64`。
   - `potassium`列有312个非空值，数据类型为`float64`。
   - `haemoglobin`列有348个非空值，数据类型为`float64`。
   - `packed_cell_volume`列有329个非空值，数据类型转换为`float64`。
   - `white_blood_cell_count`列有294个非空值，数据类型转换为`float64`。
   - `red_blood_cell_count`列有269个非空值，数据类型转换为`float64`。
   - 其他几列的数据信息以此类推。

5. **数据类型统计**:
   - `dtypes: float64(14), object(11)`: 表明有14列的数据类型是`float64`，11列的数据类型是`object`。

6. **内存使用情况**:
   - `memory usage: 78.2+ KB`: 表明DataFrame使用的内存量约为78.2KB。

从这个分析中，我们可以了解到数据集中存在一些缺失值，特别是`red_blood_cells`、`pus_cell`、`serum_creatinine`等列，它们的非空值数量少于总样本数。这可能意味着需要进一步的数据清洗工作，例如填充缺失值或删除含有缺失值的行。此外，大部分数值型数据使用`float64`类型，而一些可能是分类变量的数据使用`object`类型，这可能需要进一步的数据类型转换或处理。最后，内存使用情况显示数据集不是特别大，应该不会对大多数现代计算机的内存造成压力。



下面这段代码使用列表推导式来从DataFrame `df`中提取分类（categorical）列和数值（numerical）列的名称。以下是对每行代码的详细中文注释：

```python
# 提取分类和数值列

# 使用列表推导式创建一个名为cat_cols的新列表。
# 遍历df中的所有列，检查每一列的数据类型。
# 如果某一列的数据类型为'object'（通常表示字符串类型），则将该列的名称添加到cat_cols列表中。
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

# 使用列表推导式创建一个名为num_cols的新列表。
# 遍历df中的所有列，检查每一列的数据类型。
# 如果某一列的数据类型不是'object'（即数值类型，如int64或float64），则将该列的名称添加到num_cols列表中。
num_cols = [col for col in df.columns if df[col].dtype != 'object']
```

执行这段代码后，`cat_cols`列表将包含DataFrame `df`中所有分类列的名称，而`num_cols`列表将包含所有数值列的名称。这种区分对于数据分析和机器学习任务非常重要，因为分类数据和数值数据通常需要不同的处理方法。例如，分类数据可能需要进行编码（如独热编码或标签编码），而数值数据可能需要进行标准化或归一化。通过将列分为两类，我们可以更有针对性地对数据进行预处理和分析。




下面这段代码用于遍历DataFrame `df`中的分类列，并打印每个分类列的唯一值数量。以下是对每行代码的详细中文注释：

```python
# 查看分类列中的唯一值

# 使用for循环遍历cat_cols列表中的每个元素，即DataFrame df的分类列。
for col in cat_cols:
    # 使用f-string格式化字符串，打印当前遍历到的列名col和该列中所有唯一值的列表。
    # df[col]访问DataFrame df中的列col，并使用.unique()方法获取该列的所有唯一值。
    # 打印的结果将展示每个分类列的名称以及它包含的独特值的数量。
    print(f"{col} has {df[col].unique()} values\n")
```

执行这段代码后，将在Python环境中逐行输出每个分类列的名称和该列中包含的唯一值的列表。这有助于了解分类数据的多样性和分布情况，例如，如果一个分类列包含很多唯一值，这可能意味着该特征在数据中具有较高的区分度。相反，如果一个分类列的唯一值数量较少，这可能表明该特征在区分不同样本时的作用有限。在数据分析和机器学习中，了解分类特征的唯一值数量对于特征选择和模型构建是非常重要的。

```python
red_blood_cells has [nan 'normal' 'abnormal'] values

pus_cell has ['normal' 'abnormal' nan] values

pus_cell_clumps has ['notpresent' 'present' nan] values

bacteria has ['notpresent' 'present' nan] values

hypertension has ['yes' 'no' nan] values

diabetes_mellitus has ['yes' 'no' ' yes' '\tno' '\tyes' nan] values

coronary_artery_disease has ['no' 'yes' '\tno' nan] values

appetite has ['good' 'poor' nan] values

peda_edema has ['no' 'yes' nan] values

aanemia has ['no' 'yes' nan] values

class has ['ckd' 'ckd\t' 'notckd'] values
```

执行上述代码后，我们得到了DataFrame `df`中分类列的唯一值列表。以下是对结果的详细分析：

1. **red_blood_cells**:
   - 包含3个唯一值：`nan`、`normal`和`abnormal`。这表明红细胞检查结果有缺失值（`nan`），以及正常和异常两种状态。

2. **pus_cell**:
   - 包含3个唯一值：`normal`、`abnormal`和`nan`。这表示脓细胞检查结果也有缺失值。

3. **pus_cell_clumps**:
   - 包含3个唯一值：`notpresent`、`present`和`nan`。这表示脓细胞团检查结果中有缺失值，以及存在和不存在两种状态。

4. **bacteria**:
   - 包含3个唯一值：`notpresent`、`present`和`nan`。这表示细菌检查结果中有缺失值，以及存在和不存在两种状态。

5. **hypertension**:
   - 包含3个唯一值：`yes`、`no`和`nan`。这表示高血压状态中有缺失值，以及有和无高血压两种状态。

6. **diabetes_mellitus**:
   - 包含多个唯一值：`yes`、`no`、`nan`、`\tno`和`\tyes`。这表示糖尿病状态中有缺失值，以及有和无糖尿病两种状态。注意，`\tno`和`\tyes`可能是由于数据输入时的格式问题导致的，可能需要进一步的清理。

7. **coronary_artery_disease**:
   - 包含4个唯一值：`no`、`yes`、`\tno`和`nan`。这表示冠状动脉疾病状态中有缺失值，以及有和无冠状动脉疾病两种状态。

8. **appetite**:
   - 包含3个唯一值：`good`、`poor`和`nan`。这表示食欲状态中有缺失值，以及良好和差两种状态。

9. **peda_edema**:
   - 包含3个唯一值：`no`、`yes`和`nan`。这表示脚部水肿状态中有缺失值，以及有和无脚部水肿两种状态。

10. **aanemia**:
    - 包含3个唯一值：`no`、`yes`和`nan`。这表示贫血状态中有缺失值，以及有和无贫血两种状态。

11. **class**:
    - 包含3个唯一值：`ckd`、`ckd\t`和`notckd`。这表示肾脏疾病分类中有慢性肾脏病（CKD）和非CKD两种状态。注意，`ckd\t`可能是由于数据输入时的格式问题导致的，可能需要进一步的清理。

从这些结果可以看出，数据集中的分类特征存在缺失值（`nan`），这可能需要进一步的数据清洗，例如通过填充缺失值或删除含有缺失值的记录。此外，一些列中的字符串值可能包含额外的空白字符（如`\t`），这也可能需要清理。了解这些唯一值对于后续的数据预处理和特征工程非常重要。


There is some ambugity present in the columns we have to remove that.


下面这段代码用于替换DataFrame `df`中某些列中的不正确或不一致的值。以下是对每行代码的详细中文注释：

```python
# 替换不正确的值

# 使用replace方法替换'diabetes_mellitus'列中的特定值。
# to_replace字典包含了要被替换的值和对应的新值。
# {'\tno':'no', '\tyes':'yes', ' yes':'yes'}表示将带有额外制表符的'no'和'yes'以及带有多余空格的'yes'替换为正确的形式。
# inplace=True参数表示直接在原始DataFrame上进行替换，而不是创建一个新的DataFrame。
df['diabetes_mellitus'].replace(to_replace={'\tno':'no', '\tyes':'yes', ' yes':'yes'}, inplace=True)

# 使用replace方法替换'coronary_artery_disease'列中的特定值。
# to_replace参数指定了要被替换的值，这里是'\tno'。
# value参数指定了新值，这里是'no'。
# 这个操作将所有的'\tno'替换为'no'，移除了值前的额外制表符。
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')

# 使用replace方法替换'class'列中的特定值。
# to_replace字典包含了要被替换的值和对应的新值。
# {'ckd\t': 'ckd', 'notckd': 'not ckd'}表示将带有额外制表符的'ckd'替换为没有制表符的'ckd'，并将'notckd'替换为'not ckd'。
# 这个操作旨在统一列中的值，使其更加一致。
df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})
```

执行这段代码后，`df`中的`diabetes_mellitus`、`coronary_artery_disease`和`class`列将被更新，所有指定的不正确或不一致的值都将被替换为新的值。这种数据清洗步骤对于确保数据质量和后续分析的准确性非常重要。在机器学习和数据分析中，处理这类数据问题是一个常见的预处理步骤，它有助于提高模型的性能和解释性。


下面这段代码用于将DataFrame `df`中的`class`列中的分类标签转换为数值型数据，并处理可能的转换错误。以下是对每行代码的详细中文注释：

```python
# 将'class'列中的分类标签映射为数值型数据

# 使用map函数将'ckd'映射为0，将'not ckd'映射为1。
# 这样，原本的分类标签就被转换成了数值型数据，便于进行数值计算和机器学习模型的训练。
df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})

# 使用pd.to_numeric函数尝试将'class'列转换为数值型。
# 如果在转换过程中遇到无法转换为数值的值，则使用errors='coerce'参数将这些值设置为NaN。
# 这个步骤确保了所有的值都是数值型，即使是在之前的映射过程中可能出现的任何非预期的值。
df['class'] = pd.to_numeric(df['class'], errors='coerce')
```

执行这段代码后，`df`中的`class`列将完全由数值型数据组成，其中`ckd`被转换为0，`not ckd`被转换为1。如果`map`操作后存在任何非预期的值，`pd.to_numeric`将确保这些值被处理为NaN，从而避免了潜在的数据错误。这种转换是数据预处理中的常见步骤，特别是在准备数据用于机器学习模型训练时，因为大多数模型都需要数值型特征。



下面这段代码用于遍历指定的列，并打印每个列的唯一值列表。以下是对每行代码的详细中文注释：

```python
# 定义一个包含特定列名的列表cols。
cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class']

# 使用for循环遍历cols列表中的每个元素，即指定的列名。
for col in cols:
    # 使用f-string格式化字符串，打印当前遍历到的列名col和该列中所有唯一值的列表。
    # df[col]访问DataFrame df中的列col，并使用.unique()方法获取该列的所有唯一值。
    # 打印的结果将展示每个指定列的名称以及它包含的独特值的数量。
    print(f"{col} has {df[col].unique()} values\n")
```

执行这段代码后，将在Python环境中逐行输出`diabetes_mellitus`、`coronary_artery_disease`和`class`这三个列的名称和每个列中包含的唯一值的列表。这有助于确认之前的数据清洗和转换操作是否成功，以及每个分类列中值的分布情况。了解每个分类列的唯一值对于后续的数据预处理、特征工程和模型训练是非常重要的。


```python
diabetes_mellitus has ['yes' 'no' nan] values

coronary_artery_disease has ['no' 'yes' nan] values

class has [0 1] values
```

执行上述代码后，我们得到了DataFrame `df`中指定分类列的唯一值列表。以下是对结果的详细分析：

1. **diabetes_mellitus**:
   - 包含3个唯一值：`'yes'`、`'no'`和`nan`。这表明糖尿病状态列中有两种疾病状态（有糖尿病和无糖尿病），以及一些缺失值（`nan`）。

2. **coronary_artery_disease**:
   - 包含3个唯一值：`'no'`、`'yes'`和`nan`。这表示冠状动脉疾病状态列中有两种疾病状态（有冠状动脉疾病和无冠状动脉疾病），以及一些缺失值（`nan`）。

3. **class**:
   - 包含2个唯一值：`0`和`1`。这表示肾脏疾病分类列已经被转换为数值型数据，其中`0`可能代表“无慢性肾脏病”（not ckd），`1`代表“有慢性肾脏病”（ckd）。没有出现`nan`值，这意味着之前的转换操作已经成功地将所有的分类标签映射为了数值型数据。

从这些结果可以看出，数据集中的分类特征已经通过之前的清洗和转换步骤被处理为了更加一致和数值型的形式。这种处理使得数据集准备好了进行后续的数据分析和机器学习任务。然而，仍然存在一些缺失值（`nan`），这可能需要进一步的处理，例如通过填充缺失值或删除含有缺失值的记录。在进行模型训练之前，确保数据的完整性是非常重要的。


下面这段代码用于创建一个图形，其中包含DataFrame `df`中数值型特征的分布图。以下是对每行代码的详细中文注释：

```python
# 检查数值型特征的分布情况

# 创建一个新的图形对象，并设置图形的大小为宽20英寸、高15英寸。
plt.figure(figsize=(20, 15))

# 初始化一个名为plotnumber的变量，用于跟踪当前的子图位置。
plotnumber = 1

# 遍历num_cols列表中的每个元素，即DataFrame df的数值型列。
for column in num_cols:
    # 如果plotnumber小于或等于14，意味着我们为前14个数值型特征创建分布图。
    # 这是因为我们通常在一张图中展示多个分布图，这里假设只展示前14个特征。
    if plotnumber <= 14:
        # 使用subplot函数创建一个新的子图。
        # 3行5列的布局中，plotnumber指定了当前子图的位置。
        ax = plt.subplot(3, 5, plotnumber)
        
        # 使用Seaborn的distplot函数绘制指定列的分布图。
        sns.distplot(df[column])
        
        # 设置x轴标签为当前特征的列名。
        plt.xlabel(column)
        
    # 每次循环后，plotnumber增加1，以便在下一次迭代中创建新的子图。
    plotnumber += 1

# 使用tight_layout函数自动调整子图参数，使之填充整个图形区域并且子图之间没有重叠。
plt.tight_layout()

# 使用show函数显示图形。
plt.show()
```

执行这段代码后，会在Python环境中显示一个包含多个分布图的图形，每个分布图对应`df`中的一个数值型特征。这有助于了解数值型特征的分布情况，例如数据的集中趋势、离散程度以及是否存在异常值。通过可视化分布图，我们可以更容易地识别数据的特性，这对于数据预处理、特征选择和模型构建等后续步骤非常重要。





![0.4distribution](01图片/0.4distribution.png)



Skewness is present in some of the columns.


下面这段代码用于创建一个图形，其中包含DataFrame `df`中分类特征的计数图。以下是对每行代码的详细中文注释：

```python
# 查看分类特征

# 创建一个新的图形对象，并设置图形的大小为宽20英寸、高15英寸。
plt.figure(figsize=(20, 15))

# 初始化一个名为plotnumber的变量，用于跟踪当前的子图位置。
plotnumber = 1

# 遍历cat_cols列表中的每个元素，即DataFrame df的分类列。
for column in cat_cols:
    # 如果plotnumber小于或等于11，意味着我们为前11个分类特征创建计数图。
    # 这是因为我们通常在一张图中展示多个计数图，这里假设只展示前11个特征。
    if plotnumber <= 11:
        # 使用subplot函数创建一个新的子图。
        # 3行4列的布局中，plotnumber指定了当前子图的位置。
        ax = plt.subplot(3, 4, plotnumber)
        
        # 使用Seaborn的countplot函数绘制指定列的计数图。
        # palette='rocket'参数设置颜色映射方案，'rocket'是一种从紫色到红色的颜色渐变。
        sns.countplot(df[column], palette='rocket')
        
        # 设置x轴标签为当前特征的列名。
        plt.xlabel(column)
        
    # 每次循环后，plotnumber增加1，以便在下一次迭代中创建新的子图。
    plotnumber += 1

# 使用tight_layout函数自动调整子图参数，使之填充整个图形区域并且子图之间没有重叠。
plt.tight_layout()

# 使用show函数显示图形。
plt.show()
```

执行这段代码后，会在Python环境中显示一个包含多个计数图的图形，每个计数图对应`df`中的一个分类特征。这有助于了解分类特征的分布情况，例如每个类别的计数和相对频率。通过可视化计数图，我们可以更容易地识别数据中的模式，例如哪些类别出现得更频繁，以及是否存在某些类别的样本数量非常少。这对于数据预处理、特征选择和模型构建等后续步骤非常重要。



![0.5categorical](01图片/0.5categorical.png)



下面这段代码用于创建DataFrame `df`中数值型特征之间的相关性热图。以下是对每行代码的详细中文注释：

```python
# 创建数据的相关性热图

# 创建一个新的图形对象，并设置图形的大小为宽15英寸、高8英寸。
plt.figure(figsize=(15, 8))

# 使用Seaborn的heatmap函数绘制相关性矩阵的热图。
# df.corr()计算DataFrame中数值型列之间的相关系数矩阵。
# annot=True参数表示在热图的每个单元格内显示相关系数的数值。
# linewidths=2参数设置单元格之间线条的宽度。
# linecolor='lightgrey'参数设置线条的颜色。
sns.heatmap(df.corr(), annot=True, linewidths=2, linecolor='lightgrey')

# 使用show函数显示图形。
plt.show()
```

执行这段代码后，会在Python环境中显示一个热图，它展示了DataFrame `df`中数值型特征之间的相关性。热图中的每个单元格表示两个特征之间的相关系数，颜色的深浅表示相关性的强度，从浅到深的颜色通常表示从弱到强的相关性。相关系数的范围是-1到1，其中1表示完全正相关，-1表示完全负相关，0表示没有相关性。这个热图有助于理解数据特征之间的关系，对于特征选择和模型构建等后续步骤非常重要。通过观察热图，我们可以识别哪些特征之间存在强相关性，这可能意味着在模型中不需要所有这些特征，或者需要考虑多重共线性的问题。


![0.6heatmap](01图片/0.6heatmap.png)



下面这行代码用于获取DataFrame `df`的所有列名。以下是对这行代码的详细中文注释：

```python
# 获取DataFrame df的所有列名

# df.columns是一个属性，它返回一个包含DataFrame中所有列名的Index对象。
# 这个Index对象可以用于迭代，以获取每个列的名称。
df.columns
```

执行这段代码后，会在Python环境中返回一个包含DataFrame `df`中所有列名的索引对象。这个信息有助于了解数据集的结构，包括有多少列、列的名称是什么，以及每一列可能代表的数据类型（例如，列名可能是描述性文本、度量值或者分类标签）。在数据分析和机器学习的上下文中，了解数据集的列结构是非常重要的第一步，它为后续的数据探索、特征工程和模型构建提供了基础。


```python
Index(['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
       'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
       'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
       'potassium', 'haemoglobin', 'packed_cell_volume',
       'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
       'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
       'peda_edema', 'aanemia', 'class'],
      dtype='object')
```


执行`df.columns`得到的结果是DataFrame `df`中所有列名的索引对象。以下是对结果的详细分析：

1. **列名列表**:
   - 包含了DataFrame `df`中的所有列名，例如`age`、`blood_pressure`、`specific_gravity`等。
   - 这些列名代表了数据集中的特征，每一列可能包含不同类型的数据，如数值型、分类型等。

2. **列名的多样性**:
   - 列名反映了数据集可能包含的各种医疗指标和病人状况，如血压(`blood_pressure`)、血糖(`blood_glucose_random`)、是否有糖尿病(`diabetes_mellitus`)等。

3. **数据类型**:
   - 结果中的`dtype='object'`表明列名是以对象类型存储的，这在Pandas中通常指的是字符串类型。
   - 这意味着列名是文本数据，而不是数值型数据。

4. **数据分析的准备**:
   - 了解列名对于数据分析非常重要，因为它帮助我们理解数据集的结构和内容。
   - 在数据预处理、特征选择、数据可视化和机器学习模型构建等步骤中，列名提供了关键的信息。

5. **后续操作**:
   - 根据列名，我们可以决定哪些列可能需要进一步的清洗或转换（如将分类型数据转换为数值型）。
   - 我们还可以基于列名选择特定的列进行分析，例如，如果我们只对数值型特征感兴趣，我们可以选择那些明显是数值型数据的列。

通过分析列名，我们可以对数据集有一个初步的了解，并为后续的数据处理和分析工作做好准备。


## 1. Exploratory Data Analysis (EDA)
















































































