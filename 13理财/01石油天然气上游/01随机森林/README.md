## 背景

我使用python语言构建了一个关于一只基金的预测模型，但是模型的准确性似乎不是太高，我需要你的帮助，我需要你帮助我优化代码，提高模型准确性。
此外，在提高模型准确性之后，我希望再增加一些python代码，用于预测未来一周的基金走向趋势，并绘制图形。
接下来我将给您我的代码，请问你准备好帮助我了吗？


## 代码


第一部分代码如下：

第二部分代码如下：


第三部分代码如下：

第四部分代码如下：

第五部分代码如下：

第六部分代码如下：


```python
第一部分代码如下：

import requests
import pandas as pd
import re
from html import unescape
from io import StringIO

def fetch_fund_history(code, per=20):
    """从天天基金网抓取指定基金的历史净值数据，并合并到一个 DataFrame 中。"""
    page = 1
    all_dfs = []
    total_pages = None

    while True:
        url = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page={page}&per={per}"
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'

        # 提取 content 字段中的 HTML 表格
        match = re.search(r'content:"(.*?)",records', response.text)
        if not match:
            break
        html_str = unescape(match.group(1))
        # 通过 pandas 读取 HTML 表格
        df_list = pd.read_html(StringIO(html_str))
        if df_list:
            all_dfs.append(df_list[0])

        # 获取总页数
        if total_pages is None:
            pages_match = re.search(r'pages:(\d+)', response.text)
            if pages_match:
                total_pages = int(pages_match.group(1))
            else:
                break

        # 如果已经抓取完全部页数，则跳出循环
        if page >= total_pages:
            break
        page += 1

    if not all_dfs:
        raise ValueError("未能抓取到任何历史净值数据，请检查基金代码或网络设置。")
    return pd.concat(all_dfs, ignore_index=True)

# 抓取基金历史净值
fund_code = '007844'  # 目标基金代码
print("正在抓取基金历史数据，请稍候...")
raw_df = fetch_fund_history(fund_code)
print(f"已抓取 {len(raw_df)} 条记录。")

# 保存原始数据（可选）
raw_df.to_csv('/workspace/input/' + f'{fund_code}_history_raw.csv', index=False, encoding='utf-8-sig')
print(f"原始数据已保存至 {fund_code}_history_raw.csv")


第二部分代码如下：



import pandas as pd

def clean_data(df):
    """对原始数据进行清洗，转换列名，并去除无用数据列"""
    
    # 检查数据是否加载正确
    if df.empty:
        raise ValueError("数据加载失败，表格为空。")
    
    # 将中文列名转换为英文
    df.columns = ['date', 'unit_nav', 'cum_nav', 'daily_rate', 'purchase', 'redeem', 'dividend']
    
    # 查看原始数据的前几行，确认数据格式
    print("原始数据前几行：")
    print(df.head())

    # 去除最后两列，因为其值相同
    if 'purchase' in df.columns and 'redeem' in df.columns:
        df = df.drop(columns=['purchase', 'redeem'])
    
    # 删除“单位净值”和“累计净值”其中一列，因为它们的值是相同的
    if 'cum_nav' in df.columns:
        df = df.drop(columns=['cum_nav'])

    # 进行日期转换，并检查是否存在转换失败的日期
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # 错误数据会变成 NaT
    print(f"转换日期后，NaT 数据数量：{df['date'].isna().sum()}")

    # 将“日增长率”列转化为小数形式，并检查是否有非数字字符
    df['daily_rate'] = df['daily_rate'].str.strip('%')
    df['daily_rate'] = pd.to_numeric(df['daily_rate'], errors='coerce')
    print(f"转换日增长率后，NaN 数据数量：{df['daily_rate'].isna().sum()}")

    # 将“单位净值”列转换为数值型，处理错误值
    df['unit_nav'] = pd.to_numeric(df['unit_nav'], errors='coerce')
    print(f"转换单位净值后，NaN 数据数量：{df['unit_nav'].isna().sum()}")

    # 计算收益率和涨跌方向
    df['return'] = df['unit_nav'].pct_change()
    df['direction'] = (df['return'] > 0).astype(int)  # 涨为1，跌为0
    
    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    # 去除 NaN 和空值，只删除含有 NaN 的行，而不是整个数据集
    df = df.dropna(subset=['unit_nav', 'daily_rate'])

    # 检查清洗后的数据是否为空
    if df.empty:
        raise ValueError("数据清洗后为空，请检查是否有有效数据。")
    
    return df


# 读取数据（假设文件已经上传）
file_path = '/workspace/input/007844_history_raw.csv'  # 更新为正确的路径
df = pd.read_csv(file_path)

# 清洗数据
cleaned_df = clean_data(df)

# 保存清洗后的数据
cleaned_file_path = '/workspace/input/007844_history_cleaned.csv'
cleaned_df.to_csv(cleaned_file_path, index=False, encoding='utf-8-sig')

print(f"数据清洗完成，已保存至：{cleaned_file_path}")



第三部分代码如下：


# 第一步：载入清洗后的数据，并做基础特征工程（适配小样本）

import pandas as pd
import numpy as np

# 读取清洗后的数据
file_path = '/workspace/input/007844_history_cleaned.csv'
df = pd.read_csv(file_path, parse_dates=['date'])

# 按时间排序（确保时序正确）
df = df.sort_values('date').reset_index(drop=True)

# 打印原始数据行数
print(f"原始数据行数：{len(df)}")

# 创建技术指标（适配较小数据量）
df['sma_3'] = df['unit_nav'].rolling(window=3, min_periods=1).mean()
df['sma_5'] = df['unit_nav'].rolling(window=5, min_periods=1).mean()
df['sma_7'] = df['unit_nav'].rolling(window=7, min_periods=1).mean()
df['volatility_3'] = df['unit_nav'].rolling(window=3, min_periods=1).std()

# 计算涨跌方向（如果未包含 direction，可以手动补）
if 'direction' not in df.columns:
    df['return'] = df['unit_nav'].pct_change()
    df['direction'] = (df['return'] > 0).astype(int)

# 去除最前面计算不出标准差的那几行（只对波动率为空的行做处理）
df = df.dropna(subset=['volatility_3']).reset_index(drop=True)

# 再次打印保留的数据行数
print(f"保留的数据行数：{len(df)}")

# 查看结果
df.head()



第四部分代码如下：


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 确认特征列
features = ['sma_3', 'sma_5', 'sma_7', 'volatility_3']
X = df[features]
y = df['direction']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集（不打乱时间顺序）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# 初始化并训练随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 模型预测
y_pred = clf.predict(X_test)

# 输出评估指标
print("🎯 分类报告:")
print(classification_report(y_test, y_pred))
print("📊 混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

上述代码的结果如下：

🎯 分类报告:
              precision    recall  f1-score   support

           0       0.54      0.72      0.62       140
           1       0.54      0.35      0.42       133

    accuracy                           0.54       273
   macro avg       0.54      0.53      0.52       273
weighted avg       0.54      0.54      0.52       273


第六部分代码如下：

# 回填预测结果到原始 dataframe（注意对齐）
df_eval = df.iloc[-len(y_test):].copy()  # 对应测试集部分
df_eval['predicted_direction'] = y_pred

# 根据预测方向生成操作信号
df_eval['signal'] = df_eval['predicted_direction'].map({1: 'buy', 0: 'sell'})

# 保存结果到本地 CSV
output_path = '/workspace/input/007844_with_signals.csv'
df_eval.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 策略信号已生成并保存至：{output_path}")
df_eval[['date', 'unit_nav', 'predicted_direction', 'signal']].head(10)


上述代码的结果如下：

✅ 策略信号已生成并保存至：/workspace/input/007844_with_signals.csv
date	unit_nav	predicted_direction	signal
1092	2024-07-19	0.7939	1	buy
1093	2024-07-22	0.7909	1	buy
1094	2024-07-23	0.7788	0	sell
1095	2024-07-24	0.7743	0	sell
1096	2024-07-25	0.7832	0	sell
1097	2024-07-26	0.7864	0	sell
1098	2024-07-29	0.7776	0	sell
1099	2024-07-30	0.7877	0	sell
1100	2024-07-31	0.7934	0	sell
1101	2024-08-01	0.7707	1	buy



第七部分的代码如下：


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ✅ 使用你实际生成过的特征列
features = ['sma_3', 'sma_5', 'sma_7', 'volatility_3']

# ✅ 打印每一列缺失值数量，方便调试
print("各特征列缺失值统计：")
print(df[features].isna().sum())

# ✅ 丢掉包含 NaN 的行（只考虑特征和标签列）
df_features = df.dropna(subset=features + ['direction']).copy()

# ✅ 再次确认剩余行数
print(f"可用于训练的数据行数：{len(df_features)}")
if df_features.empty:
    raise ValueError("训练数据为空，请检查特征列或缺失处理逻辑。")

# ✅ 特征与标签提取
X_all = df_features[features]
y_all = df_features['direction']

# ✅ 标准化
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# ✅ 模型训练
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_all_scaled, y_all)

# ✅ 使用最新一行数据进行未来预测
latest_row = df_features.iloc[[-1]]
X_future = latest_row[features]
X_future_scaled = scaler.transform(X_future)
future_pred = clf.predict(X_future_scaled)[0]
future_signal = 'buy' if future_pred == 1 else 'sell'

print(f"\n📅 最新数据日期：{latest_row['date'].values[0]}")
print(f"📈 模型预测未来信号：{future_signal.upper()}（预测方向：{future_pred}）")



上述代码的结果如下：

各特征列缺失值统计：
sma_3           0
sma_5           0
sma_7           0
volatility_3    0
dtype: int64
可用于训练的数据行数：1365

📅 最新数据日期：2025-09-01T00:00:00.000000000
📈 模型预测未来信号：SELL（预测方向：0）

```

现在请您按照我的要求优化上述代码，你只需要输出python代码即可，不需要你进行运行，我会在我的电脑上进行运行，你只需要保证代码逻辑正确。你需要分部输出代码




## 0103

下面给出**适合在 Jupyter Notebook 逐格运行**的完整代码（已按你的需求重构与优化）。
每个 Part 是一个单独代码单元，**直接复制到不同的 Notebook 单元**按顺序运行即可。

---

### Part 1 — 抓取并保存原始历史净值（东财）

```python
# Part 1 — 抓取并保存原始历史净值（东财）
import re
import time
import requests
import pandas as pd
from html import unescape
from io import StringIO
from pathlib import Path

# 全局配置（可修改）
FUND_CODE = "007844"
DATA_DIR = "/workspace/input"  # 若不存在此目录，可改为 "."
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

def fetch_fund_history(code: str, per: int = 50, sleep_s: float = 0.6) -> pd.DataFrame:
    page, total_pages = 1, None
    all_dfs = []
    while True:
        url = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page={page}&per={per}"
        r = requests.get(url, timeout=10)
        r.encoding = "utf-8"

        m = re.search(r'content:"(.*?)",records', r.text)
        if not m:
            break
        html_str = unescape(m.group(1))
        dfs = pd.read_html(StringIO(html_str))
        if dfs:
            all_dfs.append(dfs[0])

        if total_pages is None:
            pm = re.search(r'pages:(\d+)', r.text)
            total_pages = int(pm.group(1)) if pm else page

        if page >= total_pages:
            break
        page += 1
        time.sleep(sleep_s)

    if not all_dfs:
        raise ValueError("未抓取到任何数据，请检查基金代码或网络。")

    df = pd.concat(all_dfs, ignore_index=True)
    df.columns = ['date', 'unit_nav', 'cum_nav', 'daily_rate', 'purchase', 'redeem', 'dividend']
    return df

RAW_SAVE = f"{DATA_DIR}/{FUND_CODE}_history_raw.csv"
raw_df = fetch_fund_history(FUND_CODE, per=50)
raw_df.to_csv(RAW_SAVE, index=False, encoding="utf-8-sig")
print(f"[OK] 抓取完成，保存：{RAW_SAVE}（{len(raw_df)} 行）")
raw_df.head()
```

---

### Part 2 — 清洗与基础校验

```python
# Part 2 — 清洗与基础校验
import pandas as pd

RAW_PATH = f"{DATA_DIR}/{FUND_CODE}_history_raw.csv"
CLEAN_PATH = f"{DATA_DIR}/{FUND_CODE}_history_cleaned.csv"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ['date', 'unit_nav', 'cum_nav', 'daily_rate', 'purchase', 'redeem', 'dividend']

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['unit_nav'] = pd.to_numeric(df['unit_nav'], errors='coerce')
    df['cum_nav'] = pd.to_numeric(df['cum_nav'], errors='coerce')
    df['daily_rate'] = pd.to_numeric(df['daily_rate'].astype(str).str.strip('%'), errors='coerce') / 100.0

    # 排序去重与无效列
    df = df.dropna(subset=['date']).sort_values('date').drop_duplicates(subset=['date'])
    df = df.drop(columns=['purchase', 'redeem', 'dividend'], errors='ignore')
    if 'cum_nav' in df.columns:
        df = df.drop(columns=['cum_nav'])

    df = df.dropna(subset=['unit_nav']).reset_index(drop=True)
    return df

df_raw = pd.read_csv(RAW_PATH)
df_clean = clean_data(df_raw)
df_clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
print(f"[OK] 清洗完成：{CLEAN_PATH}（{len(df_clean)} 行）")
df_clean.head()
```

---

### Part 3 — 特征工程与标签构建（避免泄漏）

```python
# Part 3 — 特征工程与标签构建（避免泄漏）
import numpy as np
import pandas as pd

CLEAN_PATH = f"{DATA_DIR}/{FUND_CODE}_history_cleaned.csv"
FEAT_PATH = f"{DATA_DIR}/{FUND_CODE}_features.csv"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret1'] = df['unit_nav'].pct_change()

    # 均线与波动率（仅用历史）
    for w in [3, 5, 7, 10, 20]:
        df[f'sma_{w}'] = df['unit_nav'].rolling(w, min_periods=1).mean()
    for w in [3, 5, 10, 20]:
        df[f'vol_{w}'] = df['ret1'].rolling(w, min_periods=2).std()

    # 动量/RSI
    df['mom_5'] = df['unit_nav'].pct_change(5)
    delta = df['unit_nav'].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['unit_nav'].ewm(span=12, adjust=False).mean()
    ema26 = df['unit_nav'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_sig'] = signal
    df['macd_hist'] = df['macd'] - df['macd_sig']

    # 布林带
    ma20 = df['unit_nav'].rolling(20, min_periods=20).mean()
    std20 = df['unit_nav'].rolling(20, min_periods=20).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    df['bb_pct']   = (df['unit_nav'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # 标签：t+1方向 & 未来5日总体方向
    df['target_1d'] = (df['ret1'].shift(-1) > 0).astype(int)
    df['ret_fwd_5'] = df['unit_nav'].shift(-5) / df['unit_nav'] - 1.0
    df['target_5d'] = (df['ret_fwd_5'] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df

base = pd.read_csv(CLEAN_PATH, parse_dates=['date'])
feat = add_features(base)
feat.to_csv(FEAT_PATH, index=False, encoding="utf-8-sig")
print(f"[OK] 特征完成：{FEAT_PATH}（{len(feat)} 行, 列数={feat.shape[1]}）")
feat.head()
```

---

### Part 4 — 时序验证 + 模型选择 + 测试集评估与图

```python
# Part 4 — 时序验证 + 模型选择 + 测试集评估与图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve,
                             balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

FEAT_PATH = f"{DATA_DIR}/{FUND_CODE}_features.csv"

FEATURES = [
    'sma_3','sma_5','sma_7','sma_10','sma_20',
    'vol_3','vol_5','vol_10','vol_20',
    'mom_5','rsi_14','macd','macd_sig','macd_hist',
    'bb_pct'
]
TARGET = 'target_1d'   # 也可以改为 'target_5d' 评估一周趋势

df = pd.read_csv(FEAT_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)

# 留出法：最后 20% 作为最终测试集
test_size = max(200, int(len(df) * 0.2))
train_df, test_df = df.iloc[:-test_size], df.iloc[-test_size:]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

# 两类模型对比
pipelines = {
    "logreg": Pipeline([("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=500, class_weight='balanced'))]),
    "rf": Pipeline([("scaler", StandardScaler(with_mean=False)),
                    ("clf", RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))])
}
param_grid = {
    "logreg": {"clf__C": [0.1, 0.5, 1.0, 2.0]},
    "rf": {"clf__n_estimators": [200, 400],
           "clf__max_depth": [None, 5, 8, 12],
           "clf__min_samples_leaf": [1, 3, 5]}
}

tscv = TimeSeriesSplit(n_splits=5)
best_model, best_name, best_score = None, None, -np.inf
for name, pipe in pipelines.items():
    gscv = GridSearchCV(pipe, param_grid[name], scoring="balanced_accuracy", cv=tscv, n_jobs=-1, verbose=0)
    gscv.fit(X_train, y_train)
    print(f"[CV] {name} best score={gscv.best_score_:.4f}, params={gscv.best_params_}")
    if gscv.best_score_ > best_score:
        best_model, best_name, best_score = gscv.best_estimator_, name, gscv.best_score_

print(f"[SELECT] 选择模型：{best_name}（CV balanced_acc={best_score:.4f}）")

# 测试集评估
y_pred = best_model.predict(X_test)
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
else:
    dec = best_model.decision_function(X_test)
    y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"\n[TEST] balanced_accuracy={bal_acc:.4f}\n")
print(classification_report(y_test, y_pred, digits=4))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Confusion Matrix (Holdout)")
plt.tight_layout()
plt.show()

# ROC
try:
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Holdout)")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("[WARN] 无法计算ROC：", e)

# 保存测试集预测
pred_out = test_df[['date','unit_nav']].copy()
pred_out['y_true'] = y_test.values
pred_out['y_pred'] = y_pred
pred_out['proba_up'] = y_proba
pred_out['signal'] = np.where(pred_out['y_pred']==1, 'buy', 'sell')
SAVE_PRED = f"{DATA_DIR}/{FUND_CODE}_holdout_predictions.csv"
pred_out.to_csv(SAVE_PRED, index=False, encoding="utf-8-sig")
print(f"[OK] 测试集预测已保存：{SAVE_PRED}")
```

---

### Part 5 — 训练全量模型并输出“下一日”与“一周趋势”预测 + 绘图

```python
# Part 5 — 训练全量模型并输出“下一日”与“一周趋势”预测 + 绘图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

FEAT_PATH = f"{DATA_DIR}/{FUND_CODE}_features.csv"
OUT_SIG = f"{DATA_DIR}/{FUND_CODE}_future_signals.csv"

FEATURES = [
    'sma_3','sma_5','sma_7','sma_10','sma_20',
    'vol_3','vol_5','vol_10','vol_20',
    'mom_5','rsi_14','macd','macd_sig','macd_hist',
    'bb_pct'
]

def train_best(X, y):
    pipelines = {
        "logreg": Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=500, class_weight='balanced'))]),
        "rf": Pipeline([("scaler", StandardScaler(with_mean=False)),
                        ("clf", RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))])
    }
    grid = {
        "logreg": {"clf__C":[0.1,0.5,1.0,2.0]},
        "rf": {"clf__n_estimators":[300,500],
               "clf__max_depth":[None,8,12],
               "clf__min_samples_leaf":[1,3,5]}
    }
    tscv = TimeSeriesSplit(n_splits=5)
    best_est, best_score = None, -np.inf
    for name, pipe in pipelines.items():
        g = GridSearchCV(pipe, grid[name], scoring="balanced_accuracy", cv=tscv, n_jobs=-1, verbose=0)
        g.fit(X, y)
        if g.best_score_ > best_score:
            best_score, best_est = g.best_score_, g.best_estimator_
    return best_est, best_score

df = pd.read_csv(FEAT_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)

# —— 下一日方向（target_1d）
df1 = df.dropna(subset=FEATURES + ['target_1d']).copy()
X1, y1 = df1[FEATURES], df1['target_1d']
model_1d, cv1 = train_best(X1, y1)
last_row = df1.iloc[[-1]][FEATURES]
p1 = model_1d.predict_proba(last_row)[:,1][0] if hasattr(model_1d,"predict_proba") else None
pred1 = model_1d.predict(last_row)[0]
sig1 = 'BUY' if pred1==1 else 'SELL'
print(f"[NEXT-DAY] {sig1} | up_prob={p1:.3f}" if p1 is not None else f"[NEXT-DAY] {sig1}")

# —— 未来5日总体趋势（target_5d）
df5 = df.dropna(subset=FEATURES + ['target_5d']).copy()
X5, y5 = df5[FEATURES], df5['target_5d']
model_5d, cv5 = train_best(X5, y5)
last_row5 = df5.iloc[[-1]][FEATURES]
p5 = model_5d.predict_proba(last_row5)[:,1][0] if hasattr(model_5d,"predict_proba") else None
pred5 = model_5d.predict(last_row5)[0]
sig5 = 'UP_5D' if pred5==1 else 'DOWN_5D'
print(f"[WEEK] {sig5} | up_prob={p5:.3f}" if p5 is not None else f"[WEEK] {sig5}")

# —— 绘图：最近200个交易日 + 未来信号标注
tail_n = 200
plot_df = df.tail(tail_n).copy()
plt.figure(figsize=(10,4))
plt.plot(plot_df['date'], plot_df['unit_nav'], label='Unit NAV')
plt.title(f"{FUND_CODE} — 最近{tail_n}日净值与未来信号")
plt.xlabel("Date"); plt.ylabel("Unit NAV")

x_last = plot_df['date'].iloc[-1]; y_last = plot_df['unit_nav'].iloc[-1]
txt = f"Next: {sig1}"
if p1 is not None: txt += f" ({p1:.2f})"
txt += f" | Week: {sig5}"
if p5 is not None: txt += f" ({p5:.2f})"
plt.scatter([x_last],[y_last], s=60)
plt.annotate(txt, (x_last, y_last), xytext=(15, 15), textcoords="offset points",
             bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray"))
plt.tight_layout()
plt.show()

# —— 保存未来信号
out = {
    "next_day_signal": sig1,
    "next_day_prob_up": float(p1) if p1 is not None else None,
    "week_trend_signal": sig5,
    "week_prob_up": float(p5) if p5 is not None else None,
    "asof": str(df['date'].iloc[-1].date())
}
pd.DataFrame([out]).to_csv(OUT_SIG, index=False, encoding="utf-8-sig")
print(f"[OK] 未来信号已保存：{OUT_SIG}")
```

---

### Part 6 — 可选：滚动回测（Walk-Forward）与 Buy\&Hold 对比

```python
# Part 6 — 可选：滚动回测（Walk-Forward）与 Buy&Hold 对比
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

FEAT_PATH = f"{DATA_DIR}/{FUND_CODE}_features.csv"
OUT_BT = f"{DATA_DIR}/{FUND_CODE}_walkforward_backtest.csv"

FEATURES = [
    'sma_3','sma_5','sma_7','sma_10','sma_20',
    'vol_3','vol_5','vol_10','vol_20',
    'mom_5','rsi_14','macd','macd_sig','macd_hist',
    'bb_pct'
]

df = pd.read_csv(FEAT_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
df = df.dropna(subset=FEATURES + ['target_1d','ret1']).copy()

# 基模型（与 Part 5 风格一致）
base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, class_weight='balanced'))
])

# Walk-forward：前 i 天训练 -> 预测 i+1 天（严格只用历史）
preds = []
min_train = 250  # 约一年起步
for i in range(min_train, len(df)-1):
    train = df.iloc[:i]
    test_row = df.iloc[[i]]

    model = clone(base_pipeline)
    model.fit(train[FEATURES], train['target_1d'])
    proba_up = model.predict_proba(test_row[FEATURES])[:,1][0]
    pred = 1 if proba_up >= 0.5 else 0
    preds.append((df.loc[i,'date'], pred, proba_up))

wf = pd.DataFrame(preds, columns=['date','pred','proba_up'])
merged = df.merge(wf, on='date', how='left')

# 简单策略：pred==1 持有，否则持币（未计交易成本/滑点）
merged['ret_strategy'] = merged['ret1'] * merged['pred'].shift(1)
merged['ret_bh'] = merged['ret1']
merged[['cum_strategy','cum_bh']] = (1+merged[['ret_strategy','ret_bh']]).fillna(1.0).cumprod()

# 画累计收益
plt.figure(figsize=(10,4))
plt.plot(merged['date'], merged['cum_bh'], label='Buy & Hold')
plt.plot(merged['date'], merged['cum_strategy'], label='Strategy (WF)')
plt.title(f"{FUND_CODE} 策略回测（滚动训练，未计费）")
plt.xlabel("Date"); plt.ylabel("Cumulative Return (×)")
plt.legend()
plt.tight_layout()
plt.show()

# 保存
merged[['date','unit_nav','ret1','pred','proba_up','ret_strategy','ret_bh','cum_strategy','cum_bh']].to_csv(
    OUT_BT, index=False, encoding="utf-8-sig"
)
print(f"[OK] 滚动回测结果已保存：{OUT_BT}")
```

---

> 可选小提示：
>
> * 若你的 Notebook 没有 `/workspace/input` 这个目录，把 `DATA_DIR` 改成 `"."` 即可。
> * 如果你希望**只用你最开始的 4 个特征**（`sma_3,sma_5,sma_7,volatility_3`），把 FEATURES 列表改成那 4 个即可（`volatility_3` 对应这里的 `vol_3`）。
> * 本套代码已经：避免了数据泄漏、使用 `TimeSeriesSplit` 做时序交叉验证、以 `balanced_accuracy` 作为评分、更稳健地评估了模型，并附带“下一日 + 一周趋势”预测与图形。



