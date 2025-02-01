# 计划与方案
> 提示词：
> 这份代码，包含了我利用2016年至2023年的数据构建了三个时间序列模型，并且我使用2024年的数据进行测试，用来评价模型的性能。
现在我想基于这份代码所完成的工作，来制作一个app或者小程序，目的是我输入最近一段时间的发病人数，比如最近一个月或是两个月的，它就可以输出接下来一个月的发病人数。我想做一个这样的程序，请问你觉得可行吗？如果可行，请您给出详细的计划和方案，我将会按照你给出的方案和计划进行实施。


## 可行性判断
### 可行性分析

您的需求是完全可行的。基于已有的时间序列模型（如 SARIMA、Prophet 或 LSTM），可以开发一个应用程序或小程序，通过输入近期数据自动预测未来发病人数。以下是详细的计划和方案：

---

### **一、技术方案设计**
#### **1. 技术选型**
- **后端框架**：Python + Flask/FastAPI（轻量级、适合机器学习部署）
- **前端框架**：
  - **Web 应用**：React/Vue.js + HTML/CSS（适合跨平台）
  - **小程序**：微信小程序/UniApp（适合移动端）
- **模型部署**：
  - 使用 `joblib` 或 `pickle` 保存预训练模型。
  - 使用 TensorFlow Serving 或 ONNX 部署深度学习模型（如 LSTM）。
- **数据库**（可选）：SQLite/MySQL（用于存储用户输入的历史数据）。

#### **2. 核心功能**
- **输入**：用户上传或输入最近 1-2 个月的发病人数（支持 CSV 文件或表单输入）。
- **预测**：调用预训练模型输出下个月的预测值。
- **展示**：可视化预测结果（折线图、表格）。
- **历史记录**（可选）：保存用户查询记录。

#### **3. 技术流程**
```plaintext
用户输入数据 → 后端接收并预处理 → 模型预测 → 结果返回前端展示
```

---

### **二、开发步骤**
#### **1. 模型优化与固化**
- **保存模型**：
  ```python
  # 以 SARIMA 为例
  import joblib
  joblib.dump(model, "sarima_model.pkl")
  ```
- **预处理标准化**：确保输入数据与训练数据的分布一致（如归一化、差分）。

#### **2. 后端开发（Python）**
- **API 接口设计**：
  ```python
  # Flask 示例
  from flask import Flask, request, jsonify
  import joblib
  import pandas as pd

  app = Flask(__name__)
  model = joblib.load("sarima_model.pkl")

  @app.route("/predict", methods=["POST"])
  def predict():
      data = request.json["data"]  # 接收输入数据（如 [120, 150]）
      # 预处理数据（如转换为时间序列格式）
      forecast = model.forecast(steps=1)
      return jsonify({"prediction": forecast[0]})

  if __name__ == "__main__":
      app.run(host="0.0.0.0", port=5000)
  ```

#### **3. 前端开发**
- **Web 界面（React 示例）**：
  ```jsx
  import React, { useState } from "react";
  import axios from "axios";

  function App() {
    const [inputData, setInputData] = useState("");
    const [prediction, setPrediction] = useState("");

    const handlePredict = async () => {
      const response = await axios.post("http://localhost:5000/predict", {
        data: inputData.split(",").map(Number),
      });
      setPrediction(response.data.prediction);
    };

    return (
      <div>
        <input
          type="text"
          placeholder="输入近1-2个月数据（逗号分隔，如 100,150）"
          onChange={(e) => setInputData(e.target.value)}
        />
        <button onClick={handlePredict}>预测</button>
        {prediction && <p>下月预测值：{prediction}</p>}
      </div>
    );
  }
  ```

#### **4. 小程序开发（微信小程序）**
- 使用 `wx.request` 调用后端 API：
  ```javascript
  Page({
    data: { prediction: "" },
    predict: function () {
      wx.request({
        url: "http://your-api-domain/predict",
        method: "POST",
        data: { data: "120,150" },
        success: (res) => this.setData({ prediction: res.data.prediction }),
      });
    },
  });
  ```

---

### **三、部署与测试**
#### **1. 本地测试**
- 运行 Flask 后端：
  ```bash
  python app.py
  ```
- 启动前端：
  ```bash
  npm start
  ```

#### **2. 云服务器部署**
- **推荐平台**：AWS EC2、阿里云、腾讯云。
- **步骤**：
  1. 使用 Nginx 反向代理后端 API。
  2. 使用 Docker 容器化部署（保证环境一致性）。
  3. 配置 HTTPS（Let’s Encrypt 免费证书）。

#### **3. 性能优化**
- 缓存模型加载，避免重复初始化。
- 使用异步任务（如 Celery）处理高并发请求。

---

### **四、扩展功能（可选）**
1. **多模型支持**：允许用户选择 SARIMA、Prophet 或 LSTM。
2. **动态训练**：用户上传数据后自动更新模型。
3. **可视化历史数据**：集成 ECharts 或 Plotly 展示趋势图。

---

### **五、开发周期与资源**
- **开发周期**：1-2 周（基础功能）。
- **所需技能**：
  - Python（模型部署、API 开发）
  - JavaScript（前端/小程序）
  - 基础 DevOps（部署、Docker）。

---

### **六、风险与解决方案**
- **模型精度不足**：定期用新数据重新训练模型。
- **输入格式错误**：前端添加数据校验（如数字检查）。
- **高并发性能问题**：使用异步框架（FastAPI）或横向扩展服务器。

---

通过以上方案，您可以逐步实现目标。如需进一步协助（如代码调试或部署指导），请随时告知！

## 有代码后

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 读取数据
df_all_cases = pd.read_excel('2016_2024.xlsx', sheet_name='所有病例')
df = df_all_cases

# 确保就诊时间列是字符串类型
df['就诊时间'] = df['就诊时间'].astype(str)

# 将就诊时间转换为datetime对象，并提取月份
df['就诊月份'] = pd.to_datetime(df['就诊时间'].apply(lambda x: x.split()[0])).dt.to_period('M')

# 计算每个月份的就诊次数
monthly_counts = df['就诊月份'].value_counts().sort_index()

# 将PeriodIndex转换为DatetimeIndex
monthly_counts.index = monthly_counts.index.to_timestamp()

# 划分训练集和测试集
train = monthly_counts['2016':'2023']
test = monthly_counts['2024']

# 使用霍尔特-温特斯季节性模型
model = ExponentialSmoothing(
    train,
    trend='add',  # 使用加性趋势
    seasonal='add',  # 使用加性季节性
    seasonal_periods=12  # 季节性周期为12个月
)

# 拟合模型
model_fit = model.fit()

# 打印模型参数
print("模型参数：")
print(model_fit.params)

# 预测未来12个月（即2024年）
forecast = model_fit.forecast(steps=12)
# 假设Holt-Winters模型的预测结果存储在变量 `forecast_holt_winters`
forecast_holt_winters = model_fit.forecast(steps=12)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt.title('Exponential Smoothing Forecast')
plt.xlabel('Month')
plt.ylabel('Number of Visits')
plt.legend()
plt.grid(True)
plt.show()

# 计算预测性能指标
mse = mean_squared_error(test, forecast)
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test - forecast) / test)) * 100  # 计算平均绝对百分比误差

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

```

> 上述代码是我构建的模型，下一步我应该怎么做呢?

以下是基于你的代码的下一步详细计划，分步骤实现应用部署：

---

### **一、模型固化与优化**（1-2天）
#### 1.1 保存训练好的模型
```python
import joblib

# 保存模型和预处理对象
joblib.dump(model_fit, "exponential_smoothing_model.pkl")

# 保存月度统计数据（用于后续数据对齐）
joblib.dump(monthly_counts, "monthly_stats.pkl")
```

#### 1.2 验证模型加载功能
```python
loaded_model = joblib.load("exponential_smoothing_model.pkl")
test_forecast = loaded_model.forecast(steps=12)
print(test_forecast)  # 确保输出与之前一致
```

#### 1.3 模型接口封装
创建 `predictor.py`：
```python
import joblib
import pandas as pd

class CasePredictor:
    def __init__(self, model_path="exponential_smoothing_model.pkl"):
        self.model = joblib.load(model_path)
        self.last_training_date = pd.Period("2023-12", freq="M")  # 记录最后训练时间
        
    def predict(self, new_data=None, steps=1):
        """支持增量更新（可选）"""
        if new_data:
            # 实现增量训练逻辑（需要修改模型结构）
            pass
        return self.model.forecast(steps=steps)
```

---

### **二、后端API开发**（2-3天）
#### 2.1 使用FastAPI构建服务
安装依赖：
```bash
pip install fastapi uvicorn python-multipart
```

创建 `main.py`：
```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from predictor import CasePredictor
import pandas as pd
import io

app = FastAPI()
predictor = CasePredictor()

class PredictionRequest(BaseModel):
    recent_data: list[float]  # 允许直接输入数值
    months: int = 1  # 预测月数

@app.post("/predict")
async def predict_from_data(request: PredictionRequest):
    try:
        # 这里可以添加数据预处理逻辑（如转换为时间序列）
        forecast = predictor.predict(steps=request.months)
        return {"prediction": forecast.tolist()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_csv")
async def predict_from_csv(file: UploadFile = File(...), months: int = 1):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode()))
    # 添加CSV数据处理逻辑
    return {"prediction": predictor.predict(steps=months).tolist()}
```

#### 2.2 测试API
启动服务：
```bash
uvicorn main:app --reload
```

使用curl测试：
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"recent_data": [120, 150], "months": 1}'
```

---

### **三、前端开发**（3-5天）
#### 3.1 Web版界面（React示例）
创建React项目：
```bash
npx create-react-app case-predictor-frontend
```

修改 `src/App.js`：
```jsx
import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

function App() {
  const [inputData, setInputData] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);

  const predictCases = async () => {
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        recent_data: inputData.split(',').map(Number),
        months: 1
      });
      
      setPrediction(response.data.prediction[0]);
      setHistory([...history, {
        input: inputData,
        prediction: response.data.prediction[0],
        timestamp: new Date().toLocaleString()
      }]);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

  const chartData = {
    labels: history.map((_, i) => `Query ${i+1}`),
    datasets: [{
      label: 'Prediction History',
      data: history.map(h => h.prediction),
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }]
  };

  return (
    <div className="App" style={{ padding: 20 }}>
      <h1>疾病发病人数预测系统</h1>
      <div style={{ marginBottom: 20 }}>
        <input
          type="text"
          placeholder="输入近1-2个月数据（逗号分隔，如 100,150）"
          style={{ width: 300, marginRight: 10 }}
          onChange={(e) => setInputData(e.target.value)}
        />
        <button onClick={predictCases}>开始预测</button>
      </div>

      {prediction && (
        <div style={{ marginBottom: 30 }}>
          <h3>预测结果</h3>
          <div style={{ fontSize: 24, color: 'green' }}>
            下月预计发病人数：{Math.round(prediction)}
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div style={{ width: '80%', margin: '0 auto' }}>
          <h3>预测历史记录</h3>
          <Line data={chartData} />
          <table style={{ width: '100%', marginTop: 20 }}>
            <thead>
              <tr>
                <th>时间</th>
                <th>输入数据</th>
                <th>预测结果</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td>{h.timestamp}</td>
                  <td>{h.input}</td>
                  <td>{h.prediction.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
```

#### 3.2 微信小程序版
```wxml
<!-- pages/index/index.wxml -->
<view class="container">
  <input 
    placeholder="输入近1-2个月数据（逗号分隔）" 
    bindinput="onInput" 
    style="width: 80%; padding: 10rpx;"
  />
  <button bindtap="predict" style="margin-top: 20rpx;">开始预测</button>
  
  <view wx:if="{{prediction}}" class="result-box">
    <text>预测结果：{{prediction}} 人</text>
  </view>

  <canvas canvas-id="chart" style="width: 100%; height: 400rpx; margin-top: 40rpx;"></canvas>
</view>
```

---

### **四、部署方案**（2-3天）
#### 4.1 使用Docker容器化
创建 `Dockerfile`：
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

构建镜像：
```bash
docker build -t case-predictor .
```

#### 4.2 使用Nginx配置
`nginx.conf` 示例：
```nginx
server {
    listen 80;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /app/static_files;
    }
}
```

#### 4.3 云服务部署
推荐使用方案：
```bash
# 1. 启动Docker容器
docker run -d -p 8000:80 --name predictor case-predictor

# 2. 使用PM2保持运行（可选）
npm install -g pm2
pm2 serve build 3000  # React生产版本
```

---

### **五、扩展功能开发**（可选）
#### 5.1 多模型支持
```python
# 修改predictor.py
class MultiModelPredictor:
    def __init__(self):
        self.models = {
            "hw": joblib.load("holt_winters.pkl"),
            "sarima": joblib.load("sarima.pkl")
        }

    def predict(self, model_type="hw", steps=1):
        return self.models[model_type].forecast(steps=steps)
```

#### 5.2 自动数据更新
创建 `data_updater.py`：
```python
import schedule
import time

def update_model():
    # 实现自动获取新数据并重新训练
    print("Daily model update...")

schedule.every().day.at("02:00").do(update_model)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

### **六、实施路线图**
| 阶段 | 时间 | 交付物 |
|-------|-------|---------|
| 模型固化 | 1天 | 可复用的模型文件（.pkl）|
| API开发 | 2天 | 可运行的预测接口 |
| Web界面 | 3天 | 响应式预测页面 |
| 移动端适配 | 3天 | 微信小程序版本 |
| 部署上线 | 2天 | 云端可访问的服务 |
| 监控系统 | 1天 | 性能监控看板 |

---

### **常见问题解决方案**
1. **输入数据格式问题**：
   - 在前端添加正则校验：`/^\d+(,\d+)*$/`
   - 后端添加try-except处理

2. **模型冷启动问题**：
   ```python
   # 在predictor类中添加默认值
   def predict(self, steps=1):
       if not self.model:
           raise ValueError("Model not initialized")
       return self.model.forecast(steps=steps)
   ```

3. **时区处理**：
   ```python
   # 在API层统一转换时区
   from pytz import timezone
   sh_tz = timezone('Asia/Shanghai')
   ```

---

下一步建议从模型部署开始，先确保API服务能稳定运行，再逐步开发前端界面。如果需要具体某个环节的深入指导，请随时告知！

## 实施计划

以下是基于你的代码的下一步详细计划，分步骤实现应用部署：

---

### **一、模型固化与优化**（1-2天）
#### 1.1 保存训练好的模型
```python
import joblib

# 保存模型和预处理对象
joblib.dump(model_fit, "exponential_smoothing_model.pkl")

# 保存月度统计数据（用于后续数据对齐）
joblib.dump(monthly_counts, "monthly_stats.pkl")
```

#### 1.2 验证模型加载功能
```python
loaded_model = joblib.load("exponential_smoothing_model.pkl")
test_forecast = loaded_model.forecast(steps=12)
print(test_forecast)  # 确保输出与之前一致
```

#### 1.3 模型接口封装
创建 `predictor.py`：
```python
import joblib
import pandas as pd

class CasePredictor:
    def __init__(self, model_path="exponential_smoothing_model.pkl"):
        self.model = joblib.load(model_path)
        self.last_training_date = pd.Period("2023-12", freq="M")  # 记录最后训练时间
        
    def predict(self, new_data=None, steps=1):
        """支持增量更新（可选）"""
        if new_data:
            # 实现增量训练逻辑（需要修改模型结构）
            pass
        return self.model.forecast(steps=steps)
```

---

### **二、后端API开发**（2-3天）
#### 2.1 使用FastAPI构建服务
安装依赖：
```bash
pip install fastapi uvicorn python-multipart
```

创建 `main.py`：
```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from predictor import CasePredictor
import pandas as pd
import io

app = FastAPI()
predictor = CasePredictor()

class PredictionRequest(BaseModel):
    recent_data: list[float]  # 允许直接输入数值
    months: int = 1  # 预测月数

@app.post("/predict")
async def predict_from_data(request: PredictionRequest):
    try:
        # 这里可以添加数据预处理逻辑（如转换为时间序列）
        forecast = predictor.predict(steps=request.months)
        return {"prediction": forecast.tolist()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_csv")
async def predict_from_csv(file: UploadFile = File(...), months: int = 1):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode()))
    # 添加CSV数据处理逻辑
    return {"prediction": predictor.predict(steps=months).tolist()}
```

#### 2.2 测试API
启动服务：
```bash
uvicorn main:app --reload
```

使用curl测试：
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"recent_data": [120, 150], "months": 1}'
```

---

### **三、前端开发**（3-5天）
#### 3.1 Web版界面（React示例）
创建React项目：
```bash
npx create-react-app case-predictor-frontend
```

修改 `src/App.js`：
```jsx
import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

function App() {
  const [inputData, setInputData] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);

  const predictCases = async () => {
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        recent_data: inputData.split(',').map(Number),
        months: 1
      });
      
      setPrediction(response.data.prediction[0]);
      setHistory([...history, {
        input: inputData,
        prediction: response.data.prediction[0],
        timestamp: new Date().toLocaleString()
      }]);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

  const chartData = {
    labels: history.map((_, i) => `Query ${i+1}`),
    datasets: [{
      label: 'Prediction History',
      data: history.map(h => h.prediction),
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }]
  };

  return (
    <div className="App" style={{ padding: 20 }}>
      <h1>疾病发病人数预测系统</h1>
      <div style={{ marginBottom: 20 }}>
        <input
          type="text"
          placeholder="输入近1-2个月数据（逗号分隔，如 100,150）"
          style={{ width: 300, marginRight: 10 }}
          onChange={(e) => setInputData(e.target.value)}
        />
        <button onClick={predictCases}>开始预测</button>
      </div>

      {prediction && (
        <div style={{ marginBottom: 30 }}>
          <h3>预测结果</h3>
          <div style={{ fontSize: 24, color: 'green' }}>
            下月预计发病人数：{Math.round(prediction)}
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div style={{ width: '80%', margin: '0 auto' }}>
          <h3>预测历史记录</h3>
          <Line data={chartData} />
          <table style={{ width: '100%', marginTop: 20 }}>
            <thead>
              <tr>
                <th>时间</th>
                <th>输入数据</th>
                <th>预测结果</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td>{h.timestamp}</td>
                  <td>{h.input}</td>
                  <td>{h.prediction.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
```

#### 3.2 微信小程序版
```wxml
<!-- pages/index/index.wxml -->
<view class="container">
  <input 
    placeholder="输入近1-2个月数据（逗号分隔）" 
    bindinput="onInput" 
    style="width: 80%; padding: 10rpx;"
  />
  <button bindtap="predict" style="margin-top: 20rpx;">开始预测</button>
  
  <view wx:if="{{prediction}}" class="result-box">
    <text>预测结果：{{prediction}} 人</text>
  </view>

  <canvas canvas-id="chart" style="width: 100%; height: 400rpx; margin-top: 40rpx;"></canvas>
</view>
```

---

### **四、部署方案**（2-3天）
#### 4.1 使用Docker容器化
创建 `Dockerfile`：
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

构建镜像：
```bash
docker build -t case-predictor .
```

#### 4.2 使用Nginx配置
`nginx.conf` 示例：
```nginx
server {
    listen 80;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /app/static_files;
    }
}
```

#### 4.3 云服务部署
推荐使用方案：
```bash
# 1. 启动Docker容器
docker run -d -p 8000:80 --name predictor case-predictor

# 2. 使用PM2保持运行（可选）
npm install -g pm2
pm2 serve build 3000  # React生产版本
```

---

### **五、扩展功能开发**（可选）
#### 5.1 多模型支持
```python
# 修改predictor.py
class MultiModelPredictor:
    def __init__(self):
        self.models = {
            "hw": joblib.load("holt_winters.pkl"),
            "sarima": joblib.load("sarima.pkl")
        }

    def predict(self, model_type="hw", steps=1):
        return self.models[model_type].forecast(steps=steps)
```

#### 5.2 自动数据更新
创建 `data_updater.py`：
```python
import schedule
import time

def update_model():
    # 实现自动获取新数据并重新训练
    print("Daily model update...")

schedule.every().day.at("02:00").do(update_model)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

### **六、实施路线图**
| 阶段 | 时间 | 交付物 |
|-------|-------|---------|
| 模型固化 | 1天 | 可复用的模型文件（.pkl）|
| API开发 | 2天 | 可运行的预测接口 |
| Web界面 | 3天 | 响应式预测页面 |
| 移动端适配 | 3天 | 微信小程序版本 |
| 部署上线 | 2天 | 云端可访问的服务 |
| 监控系统 | 1天 | 性能监控看板 |

---

### **常见问题解决方案**
1. **输入数据格式问题**：
   - 在前端添加正则校验：`/^\d+(,\d+)*$/`
   - 后端添加try-except处理

2. **模型冷启动问题**：
   ```python
   # 在predictor类中添加默认值
   def predict(self, steps=1):
       if not self.model:
           raise ValueError("Model not initialized")
       return self.model.forecast(steps=steps)
   ```

3. **时区处理**：
   ```python
   # 在API层统一转换时区
   from pytz import timezone
   sh_tz = timezone('Asia/Shanghai')
   ```

---

下一步建议从模型部署开始，先确保API服务能稳定运行，再逐步开发前端界面。如果需要具体某个环节的深入指导，请随时告知！

