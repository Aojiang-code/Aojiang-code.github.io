# 第一部分代码
# 第一部分主要涉及库的加载、数据处理和特征提取的准备工作。

### **Python代码改写（第一部分）**

import os
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# 设置随机种子
np.random.seed(289)

# 定义输入参数
input_params = {
    'egfryears': [2, 4],
    'minegfrobs': 3,
    'minbpobs': 0,
    'minhrobs': 0,
    'train_pc': 80
}

# 定义预测模式
predict_mode = 'esrd'  # 或 'death'
outcomes = {
    'esrd': ['Non-ES', 'ESRD'],
    'death': ['Survival', 'Death']
}

# 定义目标和评估指标
objective = 'binary:logistic'
eval_metric = 'mcc'  # 或 'aucpr'

# 定义是否使用SMOTE
smote_flag = False
smote_over = 2000
smote_under = 200

# 定义自动XGBoost标志
autoxgb_flag = False

# 定义临床测试标志
clintest_flag = True

# 定义重新抽样标志
resample_flag = True

# 定义快速模式标志
quick_flag = True

# 定义时间序列数量
num_ts = 2

# 定义特征列表
features = [
    'egfr', 'creatinine', 'glucose', 'height', 'bmi', 'weight', 
    'sit_dia', 'sit_sys', 'sta_dia', 'sta_sys', 'sit_hr', 'sta_hr', 
    'hgba1c', 'protcreatinine', 'proteinuria', 'diff_dia', 'diff_sys', 
    'diff_hr', 'sit_pp', 'sta_pp', 'diff_pp'
]

# 根据num_ts的值定义额外特征
if num_ts == 19:
    vars_extra = features[2:]
elif num_ts == 13:
    vars_extra = ['egfr', 'glucose', 'bmi', 'sta_sys', 'sta_dia', 'sta_hr', 
                  'sit_sys', 'sit_dia', 'sit_hr', 'diff_hr', 'sit_pp', 'sta_pp', 
                  'diff_pp']
elif num_ts == 9:
    vars_extra = ['egfr', 'glucose', 'bmi', 'sta_sys', 'sta_dia', 'sta_hr', 
                  'sit_sys', 'sit_dia', 'sit_hr']
elif num_ts == 6:
    vars_extra = ['egfr', 'glucose', 'bmi', 'sta_sys', 'sta_dia', 'sta_hr']
elif num_ts == 2:
    vars_extra = ['egfr', 'glucose']
else:
    vars_extra = []

# 定义交叉验证折数
cv_fold = 10

# 定义卡方阈值
thresh_chisq = 0.999999999

# 定义颜色特征
color_feature = 'age.init'

# 定义SHAP特征数量
shap_top = 20

# 定义SHAP列数
shap_ncol = 5

# 定义SHAP分组
shap_groups = 6

# 定义SHAP X轴边界
shap_xbound = None

# 定义图例列数
legend_ncol = 5

# 定义分页标志
paging = True

# 定义图高
plot_height = "1024px"

# 定义轴文本大小
axis_text_size = 6

# 定义图高数值
plot_height_num = int(''.join(filter(str.isdigit, plot_height)))

# 定义颜色向量
col_vector = [
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', 
    '#808000', '#ffd8b1', '#000075', '#808080', '#e6194b', '#3cb44b', 
    '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
    '#bcf60c', '#fabebe', '#e8e8e8', '#000000'
]

# 定义非包含操作符
def ni(x, y):
    return x not in y

# 定义打印函数
def printly(m):
    print(m)

# 定义临床初始日期和结束日期
clin_init = pd.to_datetime('2016-11-15')
clin_last = clin_init + pd.DateOffset(years=2)

# 定义阈值
thresh_p = 0.5
thresh_2y = 2.14
thresh_10y = 11.14

# 定义摘要后缀
dig_suffix = ''

# 定义数据文件的本地目录
dir = '/mnt/appTrendal/data'
os.chdir(dir)

# 设置警告选项
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 设置工作目录
import os
os.chdir(dir)

# 设置knitr的根目录和代码块选项
# 这部分在Python中没有直接对应的设置，但可以通过配置Jupyter Notebook或Python脚本来实现类似功能

### **说明**
# 1. **库加载**：加载了必要的Python库，包括`pandas`、`numpy`、`tsfresh`、`xgboost`等。
# 2. **参数设置**：定义了输入参数、预测模式、目标和评估指标等。
# 3. **特征列表**：定义了特征列表和额外特征。
# 4. **日期和阈值**：设置了临床初始日期、结束日期和阈值。
# 5. **工作目录**：设置了数据文件的本地目录。


# 第二部分代码
# 第二部分主要涉及数据加载、预处理和特征提取。

### **Python代码改写（第二部分）**

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# 定义数据加载函数
def load_data(file_name, col_names, na_strings, date_col):
    df = pd.read_csv(file_name, names=col_names, na_values=na_strings)
    df[date_col] = pd.to_datetime(df[date_col])
    return df

# 加载死亡数据
death = load_data('Death.csv', ['id', 'date'], 'NULL', 'date')
set_key(death, 'id', 'date')

# 加载糖尿病数据
diabetes = load_data('Diabetes.csv', ['id', 'date', 'glucose', 'glucose_str', 'hgba1c', 'hgba1c_str'], 'NULL', 'date')
diabetes = pd.melt(diabetes, id_vars=['id', 'date'], value_vars=['glucose', 'hgba1c'], var_name='variable', value_name='value')
set_key(diabetes, 'id', 'date')

# 加载体检数据
exam = load_data('Exam.corrected.csv', ['id', 'date', 'height', 'height_raw', 'weight', 'bmi_raw', 'bmi', 'sit_dia', 'sit_sys', 'sit_hr', 'sta_dia', 'sta_sys', 'sta_hr'], '\\N', 'date')
exam.drop(columns=['height_raw', 'bmi_raw'], inplace=True)
exam['diff_dia'] = exam['sit_dia'] - exam['sta_dia']
exam['diff_sys'] = exam['sit_sys'] - exam['sta_sys']
exam['diff_hr'] = exam['sit_hr'] - exam['sta_hr']
exam['sit_pp'] = exam['sit_sys'] - exam['sit_dia']
exam['sta_pp'] = exam['sta_sys'] - exam['sta_dia']
exam['diff_pp'] = exam['sit_pp'] - exam['sta_pp']
exam = pd.melt(exam, id_vars=['id', 'date'], value_vars=['height', 'weight', 'bmi', 'sit_dia', 'sit_sys', 'sit_hr', 'sta_dia', 'sta_sys', 'sta_hr', 'diff_dia', 'diff_sys', 'diff_hr', 'sit_pp', 'sta_pp', 'diff_pp'], var_name='variable', value_name='value')
set_key(exam, 'id', 'date')

# 加载肾功能数据
kidney = load_data('KidneyFunction.csv', ['id', 'date', 'creatinine', 'egfr'], 'NULL', 'date')
kidney = pd.melt(kidney, id_vars=['id', 'date'], value_vars=['creatinine', 'egfr'], var_name='variable', value_name='value')
set_key(kidney, 'id', 'date')

# 加载尿蛋白肌酐比值数据
protcreatinine = load_data('UrineProteinCreatinineRatios.csv', ['id', 'date', 'test', 'other_test', 'result', 'other_result', 'protcreatinine', 'result_str', 'norm_range', 'units', 'provider'], 'NULL', 'date')
protcreatinine = pd.melt(protcreatinine, id_vars=['id', 'date'], value_vars=['protcreatinine'], var_name='variable', value_name='value')
set_key(protcreatinine, 'id', 'date')

# 加载24小时尿蛋白数据
proteinuria = load_data('Urine24hrProtein.csv', ['id', 'date', 'test', 'other_test', 'result', 'other_result', 'proteinuria', 'result_str', 'norm_range', 'units', 'provider'], 'NULL', 'date')
proteinuria = pd.melt(proteinuria, id_vars=['id', 'date'], value_vars=['proteinuria'], var_name='variable', value_name='value')
set_key(proteinuria, 'id', 'date')

# 加载患者数据
patient = load_data('PtID.csv', ['id', 'birth_year', 'gender'], 'NULL', None)
set_key(patient, 'id')

# 加载治疗方式数据
modality = load_data('Modality.csv', ['id', 'date', 'end_date', 'modality', 'setting'], 'NULL', 'date')
modality['end_date'] = pd.to_datetime(modality['end_date'])
set_key(modality, 'id')
modality = modality.sort_values(by=['id', 'date'])

# 合并数据
data = pd.concat([kidney, diabetes, exam, protcreatinine, proteinuria], ignore_index=True)
data = data.drop_duplicates()

# 定义设置键的函数
def set_key(df, *cols):
    df.set_index(list(cols), inplace=True)

# 定义提取特征的函数
def extract_features_from_data(data, id_col, time_col, value_col, features, vars_extra):
    data = data[[id_col, time_col, value_col]].copy()
    data = data.dropna()
    data = data.rename(columns={value_col: 'value'})
    data['variable'] = data.groupby(id_col)[value_col].transform(lambda x: x.name)
    data = data.drop(columns=[value_col])
    
    # 提取特征
    extracted_features = extract_features(data, column_id=id_col, column_sort=time_col, 
                                          default_fc_parameters=EfficientFCParameters() if var in vars_extra else MinimalFCParameters(),
                                          disable_progressbar=True)
    return extracted_features

# 提取特征
extracted_features = pd.DataFrame()
for var in features:
    if var != 'creatinine':  # 排除肌酐
        temp_data = data[data['variable'] == var]
        temp_features = extract_features_from_data(temp_data, 'id', 'date', 'value', features, vars_extra)
        extracted_features = pd.concat([extracted_features, temp_features], axis=1)

# 保存提取的特征
extracted_features.to_csv('extracted_features.csv', index=False)

### **说明**
# 1. **数据加载**：使用`pandas.read_csv`加载数据，并将日期列转换为`datetime`格式。
# 2. **数据预处理**：对每个数据表进行预处理，包括去除缺失值、计算新的特征列（如血压差、脉压差等）。
# 3. **特征提取**：使用`tsfresh.extract_features`从时间序列数据中提取特征。根据变量是否在`vars_extra`中，选择不同的特征提取参数。
# 4. **特征保存**：将提取的特征保存为CSV文件。


# 第三部分代码
# 第三部分主要涉及数据处理、特征提取的进一步细化，以及模型训练的准备工作。

### **Python代码改写（第三部分）**

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# 定义数据处理函数
def process_data(data, filtered, input_params, outcomes, predict_mode):
    # 合并数据
    merged_data = pd.merge(data, filtered, on='id', how='inner')
    
    # 计算相对年份
    merged_data['rel_year'] = (merged_data['date'] - clin_init).dt.days / 365.25
    
    # 提取特征
    features_extracted = extract_features(merged_data, column_id='id', column_sort='rel_year', 
                                          default_fc_parameters=EfficientFCParameters() if var in vars_extra else MinimalFCParameters(),
                                          disable_progressbar=True)
    
    # 处理缺失值
    features_extracted.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_extracted.fillna(0, inplace=True)
    
    # 保存特征
    features_extracted.to_csv('features_extracted.csv', index=False)
    
    return features_extracted

# 加载过滤后的数据
filtered = pd.read_csv('filtered.csv')

# 提取特征
features_extracted = process_data(data, filtered, input_params, outcomes, predict_mode)

# 定义模型训练函数
def train_model(features_extracted, filtered, input_params, outcomes, predict_mode):
    # 准备训练和测试数据
    train_ids = filtered.sample(frac=input_params['train_pc'] / 100, random_state=289)['id']
    train_data = features_extracted[features_extracted.index.isin(train_ids)]
    test_data = features_extracted[~features_extracted.index.isin(train_ids)]
    
    # 准备标签
    train_labels = filtered[filtered['id'].isin(train_ids)][predict_mode].values
    test_labels = filtered[~filtered['id'].isin(train_ids)][predict_mode].values
    
    # 转换为XGBoost的DMatrix格式
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dtest = xgb.DMatrix(test_data, label=test_labels)
    
    # 定义XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'mcc',
        'eta': 0.1,
        'max_depth': 4,
        'subsample': 0.75,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1
    }
    
    # 训练模型
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # 保存模型
    model.save_model('xgb_model.bento')
    
    return model, dtrain, dtest

# 训练模型
model, dtrain, dtest = train_model(features_extracted, filtered, input_params, outcomes, predict_mode)

# 定义评估函数
def evaluate_model(model, dtest, test_labels):
    # 进行预测
    predictions = model.predict(dtest)
    predicted_labels = (predictions >= 0.5).astype(int)
    
    # 计算评估指标
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    mcc = matthews_corrcoef(test_labels, predicted_labels)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(test_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# 评估模型
evaluate_model(model, dtest, test_labels)
```

### **说明**
# 1. **数据处理**：
#    - 合并数据并计算相对年份。
#    - 使用`tsfresh`提取特征。
#    - 处理缺失值和无穷值。

# 2. **模型训练**：
#    - 准备训练和测试数据。
#    - 转换为XGBoost的DMatrix格式。
#    - 定义XGBoost参数并训练模型。
#    - 保存训练好的模型。

# 3. **模型评估**：
#    - 进行预测并计算评估指标（准确率、精确率、召回率、F1分数、MCC）。
#    - 绘制混淆矩阵。


# 第四部分代码
# 第四部分主要涉及模型的详细训练过程、超参数优化以及模型性能的评估。

### **Python代码改写（第四部分）**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 定义评估函数
def eval_mcc(y_true, y_prob):
    from sklearn.metrics import matthews_corrcoef
    best_threshold = 0.5
    best_mcc = 0
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    return best_mcc, best_threshold

# 定义XGBoost模型训练和评估函数
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # 定义XGBoost分类器
    xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='mcc')
    
    # 定义超参数网格
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [4, 6, 8],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    # 使用GridSearchCV进行超参数优化
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='matthews_corrcoef', cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 进行预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc, best_threshold = eval_mcc(y_test, y_pred_proba)
    
    print(f'Best Threshold: {best_threshold}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return best_model

# 加载特征数据
features_extracted = pd.read_csv('features_extracted.csv', index_col=0)

# 加载过滤后的数据
filtered = pd.read_csv('filtered.csv')

# 准备训练和测试数据
train_ids = filtered.sample(frac=input_params['train_pc'] / 100, random_state=289)['id']
X_train = features_extracted.loc[train_ids]
X_test = features_extracted.loc[~features_extracted.index.isin(train_ids)]
y_train = filtered.set_index('id').loc[train_ids, predict_mode].values
y_test = filtered.set_index('id').loc[~filtered['id'].isin(train_ids), predict_mode].values

# 训练和评估模型
best_model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

### **说明**
# 1. **评估函数**：
#    - 定义了`eval_mcc`函数，用于计算Matthews相关系数（MCC）并找到最佳阈值。

# 2. **模型训练和评估**：
#    - 使用`XGBClassifier`定义XGBoost分类器。
#    - 使用`GridSearchCV`进行超参数优化，优化目标为MCC。
#    - 训练模型并进行预测。
#    - 计算评估指标（准确率、精确率、召回率、F1分数、MCC）。
#    - 绘制混淆矩阵。

# 3. **数据准备**：
#    - 加载提取的特征数据和过滤后的数据。
#    - 准备训练和测试数据集。

# 4. **模型训练**：
#    - 调用`train_and_evaluate_model`函数，传入训练和测试数据，训练模型并评估性能。



# 第五部分代码

# 第五部分主要涉及模型的进一步评估，特别是SHAP值的计算和可视化，以及与临床医生预测结果的比较。

### **Python代码改写（第五部分）**

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import seaborn as sns

# 加载训练好的模型
best_model = XGBClassifier()
best_model.load_model('xgb_model.bento')

# 加载特征数据
features_extracted = pd.read_csv('features_extracted.csv', index_col=0)

# 加载过滤后的数据
filtered = pd.read_csv('filtered.csv')

# 准备训练和测试数据
train_ids = filtered.sample(frac=input_params['train_pc'] / 100, random_state=289)['id']
X_train = features_extracted.loc[train_ids]
X_test = features_extracted.loc[~features_extracted.index.isin(train_ids)]
y_train = filtered.set_index('id').loc[train_ids, predict_mode].values
y_test = filtered.set_index('id').loc[~filtered['id'].isin(train_ids), predict_mode].values

# SHAP值计算和可视化
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)

# SHAP值总结图
shap.summary_plot(shap_values, X_train, max_display=20)

# SHAP值依赖图
for feature in features:
    shap.dependence_plot(feature, shap_values[:, feature], X_train)

# 保存SHAP值
shap_values_df = pd.DataFrame(shap_values.values, columns=features)
shap_values_df.to_csv('shap_values.csv', index=False)

# 与临床医生预测结果的比较
# 假设临床医生的预测结果已经加载到 clinician_predictions DataFrame 中
clinician_predictions = pd.read_csv('ClinicianPredictions.csv')

# 计算临床医生的预测性能
def evaluate_clinician_predictions(clinician_predictions, y_test):
    clinician_predictions['actual'] = y_test
    clinician_predictions['correct'] = (clinician_predictions['prediction'] == clinician_predictions['actual']).astype(int)
    
    accuracy = clinician_predictions['correct'].mean()
    precision = clinician_predictions[clinician_predictions['actual'] == 1]['correct'].mean()
    recall = clinician_predictions[clinician_predictions['prediction'] == 1]['correct'].mean()
    f1 = 2 * precision * recall / (precision + recall)
    mcc = matthews_corrcoef(clinician_predictions['actual'], clinician_predictions['prediction'])
    
    print(f'Clinician Accuracy: {accuracy:.4f}')
    print(f'Clinician Precision: {precision:.4f}')
    print(f'Clinician Recall: {recall:.4f}')
    print(f'Clinician F1 Score: {f1:.4f}')
    print(f'Clinician MCC: {mcc:.4f}')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(clinician_predictions['actual'], clinician_predictions['prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Clinician Confusion Matrix')
    plt.show()

# 评估临床医生的预测性能
evaluate_clinician_predictions(clinician_predictions, y_test)

### **说明**
# 1. **SHAP值计算和可视化**：
#    - 使用`shap.Explainer`计算SHAP值。
#    - 使用`shap.summary_plot`生成SHAP值总结图。
#    - 使用`shap.dependence_plot`生成SHAP值依赖图。
#    - 将SHAP值保存为CSV文件。

# 2. **与临床医生预测结果的比较**：
#    - 假设临床医生的预测结果已经加载到`clinician_predictions` DataFrame中。
#    - 计算临床医生的预测性能（准确率、精确率、召回率、F1分数、MCC）。
#    - 绘制临床医生的混淆矩阵。

# ### **假设**
# - 假设临床医生的预测结果已经保存在`ClinicianPredictions.csv`文件中，文件格式如下：
#   - `id`: 患者ID
#   - `prediction`: 临床医生的预测结果（0或1）
#   - `actual`: 实际结果（0或1）


# 第六部分代码
# 第六部分主要涉及临床医生预测结果的详细分析和可视化，以及与机器学习模型预测结果的比较。

### **Python代码改写（第六部分）**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 加载临床医生预测结果
clinician_predictions = pd.read_csv('ClinicianPredictions.csv')

# 加载测试集的实际结果
y_test = pd.read_csv('y_test.csv')['actual']

# 计算临床医生的预测性能
def evaluate_clinician_predictions(clinician_predictions, y_test):
    clinician_predictions['actual'] = y_test.values
    clinician_predictions['correct'] = (clinician_predictions['prediction'] == clinician_predictions['actual']).astype(int)
    
    accuracy = clinician_predictions['correct'].mean()
    precision = clinician_predictions[clinician_predictions['actual'] == 1]['correct'].mean()
    recall = clinician_predictions[clinician_predictions['prediction'] == 1]['correct'].mean()
    f1 = 2 * precision * recall / (precision + recall)
    mcc = matthews_corrcoef(clinician_predictions['actual'], clinician_predictions['prediction'])
    
    print(f'Clinician Accuracy: {accuracy:.4f}')
    print(f'Clinician Precision: {precision:.4f}')
    print(f'Clinician Recall: {recall:.4f}')
    print(f'Clinician F1 Score: {f1:.4f}')
    print(f'Clinician MCC: {mcc:.4f}')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(clinician_predictions['actual'], clinician_predictions['prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Clinician Confusion Matrix')
    plt.show()

# 评估临床医生的预测性能
evaluate_clinician_predictions(clinician_predictions, y_test)

# 与机器学习模型预测结果的比较
# 假设机器学习模型的预测结果已经保存在 ml_predictions DataFrame 中
ml_predictions = pd.read_csv('ml_predictions.csv')

# 计算机器学习模型的预测性能
def evaluate_ml_predictions(ml_predictions, y_test):
    ml_predictions['actual'] = y_test.values
    ml_predictions['correct'] = (ml_predictions['prediction'] == ml_predictions['actual']).astype(int)
    
    accuracy = ml_predictions['correct'].mean()
    precision = ml_predictions[ml_predictions['actual'] == 1]['correct'].mean()
    recall = ml_predictions[ml_predictions['prediction'] == 1]['correct'].mean()
    f1 = 2 * precision * recall / (precision + recall)
    mcc = matthews_corrcoef(ml_predictions['actual'], ml_predictions['prediction'])
    
    print(f'ML Accuracy: {accuracy:.4f}')
    print(f'ML Precision: {precision:.4f}')
    print(f'ML Recall: {recall:.4f}')
    print(f'ML F1 Score: {f1:.4f}')
    print(f'ML MCC: {mcc:.4f}')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(ml_predictions['actual'], ml_predictions['prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('ML Confusion Matrix')
    plt.show()

# 评估机器学习模型的预测性能
evaluate_ml_predictions(ml_predictions, y_test)

# 临床医生预测结果的可视化
def plot_clinician_predictions(clinician_predictions):
    clinician_predictions['correct'] = (clinician_predictions['prediction'] == clinician_predictions['actual']).astype(int)
    
    # 绘制预测结果的分布
    fig = px.histogram(clinician_predictions, x='prediction', color='correct', barmode='group', title='Clinician Predictions Distribution')
    fig.show()
    
    # 绘制预测结果的箱线图
    fig = px.box(clinician_predictions, x='prediction', y='actual', title='Clinician Predictions Boxplot')
    fig.show()

# 可视化临床医生的预测结果
plot_clinician_predictions(clinician_predictions)

# 机器学习模型预测结果的可视化
def plot_ml_predictions(ml_predictions):
    ml_predictions['correct'] = (ml_predictions['prediction'] == ml_predictions['actual']).astype(int)
    
    # 绘制预测结果的分布
    fig = px.histogram(ml_predictions, x='prediction', color='correct', barmode='group', title='ML Predictions Distribution')
    fig.show()
    
    # 绘制预测结果的箱线图
    fig = px.box(ml_predictions, x='prediction', y='actual', title='ML Predictions Boxplot')
    fig.show()

# 可视化机器学习模型的预测结果
plot_ml_predictions(ml_predictions)

### **说明**
# 1. **临床医生预测结果的评估**：
#    - 加载临床医生的预测结果和测试集的实际结果。
#    - 计算临床医生的预测性能（准确率、精确率、召回率、F1分数、MCC）。
#    - 绘制临床医生的混淆矩阵。

# 2. **机器学习模型预测结果的评估**：
#    - 加载机器学习模型的预测结果。
#    - 计算机器学习模型的预测性能（准确率、精确率、召回率、F1分数、MCC）。
#    - 绘制机器学习模型的混淆矩阵。

# 3. **预测结果的可视化**：
#    - 使用`plotly`绘制临床医生和机器学习模型的预测结果分布和箱线图。

### **假设**
# - 假设临床医生的预测结果已经保存在`ClinicianPredictions.csv`文件中，文件格式如下：
#   - `id`: 患者ID
#   - `prediction`: 临床医生的预测结果（0或1）
#   - `actual`: 实际结果（0或1）
# - 假设机器学习模型的预测结果已经保存在`ml_predictions.csv`文件中，文件格式如下：
#   - `id`: 患者ID
#   - `prediction`: 机器学习模型的预测结果（0或1）
#   - `actual`: 实际结果（0或1）

# 第七部分代码

# 第七部分主要涉及对临床医生预测结果和机器学习模型预测结果的综合比较，包括性能指标的对比和可视化展示。

### **Python代码改写（第七部分）**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 加载临床医生预测结果
clinician_predictions = pd.read_csv('ClinicianPredictions.csv')

# 加载机器学习模型预测结果
ml_predictions = pd.read_csv('ml_predictions.csv')

# 加载测试集的实际结果
y_test = pd.read_csv('y_test.csv')['actual']

# 定义评估函数
def evaluate_predictions(predictions, y_test, name):
    predictions['actual'] = y_test.values
    predictions['correct'] = (predictions['prediction'] == predictions['actual']).astype(int)
    
    accuracy = accuracy_score(predictions['actual'], predictions['prediction'])
    precision = precision_score(predictions['actual'], predictions['prediction'])
    recall = recall_score(predictions['actual'], predictions['prediction'])
    f1 = f1_score(predictions['actual'], predictions['prediction'])
    mcc = matthews_corrcoef(predictions['actual'], predictions['prediction'])
    
    print(f'{name} Accuracy: {accuracy:.4f}')
    print(f'{name} Precision: {precision:.4f}')
    print(f'{name} Recall: {recall:.4f}')
    print(f'{name} F1 Score: {f1:.4f}')
    print(f'{name} MCC: {mcc:.4f}')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(predictions['actual'], predictions['prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

# 评估临床医生的预测性能
evaluate_predictions(clinician_predictions, y_test, 'Clinician')

# 评估机器学习模型的预测性能
evaluate_predictions(ml_predictions, y_test, 'ML')

# 绘制性能指标的对比图
def plot_performance_comparison(clinician_predictions, ml_predictions, y_test):
    clinician_metrics = {
        'Accuracy': accuracy_score(y_test, clinician_predictions['prediction']),
        'Precision': precision_score(y_test, clinician_predictions['prediction']),
        'Recall': recall_score(y_test, clinician_predictions['prediction']),
        'F1 Score': f1_score(y_test, clinician_predictions['prediction']),
        'MCC': matthews_corrcoef(y_test, clinician_predictions['prediction'])
    }
    
    ml_metrics = {
        'Accuracy': accuracy_score(y_test, ml_predictions['prediction']),
        'Precision': precision_score(y_test, ml_predictions['prediction']),
        'Recall': recall_score(y_test, ml_predictions['prediction']),
        'F1 Score': f1_score(y_test, ml_predictions['prediction']),
        'MCC': matthews_corrcoef(y_test, ml_predictions['prediction'])
    }
    
    metrics_df = pd.DataFrame([clinician_metrics, ml_metrics], index=['Clinician', 'ML'])
    
    # 绘制条形图
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Performance Metrics Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(loc='upper left')
    plt.show()

# 绘制性能指标的对比图
plot_performance_comparison(clinician_predictions, ml_predictions, y_test)

# 绘制预测结果的分布图
def plot_prediction_distribution(clinician_predictions, ml_predictions):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(clinician_predictions['prediction'], ax=axes[0], kde=True)
    axes[0].set_title('Clinician Predictions Distribution')
    
    sns.histplot(ml_predictions['prediction'], ax=axes[1], kde=True)
    axes[1].set_title('ML Predictions Distribution')
    
    plt.show()

# 绘制预测结果的分布图
plot_prediction_distribution(clinician_predictions, ml_predictions)

### **说明**
# 1. **评估函数**：
#    - 定义了`evaluate_predictions`函数，用于计算和打印预测性能指标（准确率、精确率、召回率、F1分数、MCC）。
#    - 绘制混淆矩阵。

# 2. **性能指标的对比图**：
#    - 定义了`plot_performance_comparison`函数，用于绘制临床医生和机器学习模型的性能指标对比图。
#    - 使用`pandas.DataFrame`存储性能指标，并使用`matplotlib`绘制条形图。

# 3. **预测结果的分布图**：
#    - 定义了`plot_prediction_distribution`函数，用于绘制临床医生和机器学习模型的预测结果分布图。
#    - 使用`seaborn.histplot`绘制直方图和核密度估计（KDE）。

### **假设**
# - 假设临床医生的预测结果已经保存在`ClinicianPredictions.csv`文件中，文件格式如下：
#   - `id`: 患者ID
#   - `prediction`: 临床医生的预测结果（0或1）
# - 假设机器学习模型的预测结果已经保存在`ml_predictions.csv`文件中，文件格式如下：
#   - `id`: 患者ID
#   - `prediction`: 机器学习模型的预测结果（0或1）
# - 假设测试集的实际结果已经保存在`y_test.csv`文件中，文件格式如下：
#   - `actual`: 实际结果（0或1）
