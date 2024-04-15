# Chronic_Kidney_Disease_Prediction_(98%_Accuracy)

![0封面](01图片/0封面.png)

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


















































































































































