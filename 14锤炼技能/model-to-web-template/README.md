# Model-to-Web 模板仓库（Python + TypeScript）

把 **任意 scikit-learn 风格模型** 变成 **可交互网页** 的最小可行脚手架：
- **后端**：FastAPI（/predict、/predict_csv）
- **前端**：Next.js + TypeScript（JSON/CSV 预测演示页）
- **训练脚本**：`scripts/train.py` 生成示例 Iris 模型与特征名
- **容器化**：Dockerfile + docker-compose，一条命令起全栈
- **工程化**：pytest、GitHub Actions、pre-commit

## 目录结构
```
.
├── backend/            # FastAPI 服务
│   ├── app/
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/           # Next.js 前端
│   ├── app/page.tsx
│   ├── lib/api.ts
│   ├── package.json
│   └── Dockerfile
├── scripts/
│   └── train.py        # 训练并保存示例模型到 models/
├── models/             # 模型与元数据（feature_names.json, target_names.json）
├── data/               # 示例 CSV
├── docker-compose.yml
├── Makefile
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
└── README.md
```

## 快速开始（本地）
```bash
# 1) 安装依赖
python -m pip install -r backend/requirements.txt
(cd frontend && npm install)

# 2) 训练示例模型
python scripts/train.py

# 3) 启动后端（端口 8000）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend

# 4) 启动前端（端口 3000）
cd frontend && npm run dev
# 浏览器打开 http://localhost:3000
```

## Docker 一键启动
```bash
docker compose up --build
# 前端：http://localhost:3000
# 后端：http://localhost:8000/docs
```

## 将你的模型接入
1. 用你的数据训练模型，保存到 `models/your_model.pkl`，并写入 `feature_names.json`（列名数组）。
2. 设置环境变量 `MODEL_PATH=models/your_model.pkl`（可在 docker-compose 或部署环境中配置）。
3. 保持输入 JSON 的 `X` 为二维数组，或使用 `/predict_csv` 上传包含同名列的 CSV。

## 常见定制
- **鉴权**：在 FastAPI 加 `OAuth2PasswordBearer` 或前端接入 OAuth（Auth0/Supabase）。
- **可视化**：在前端增加 ECharts 图表展示校准曲线或 SHAP（可后续扩展）。
- **批量/异步**：把推理放到任务队列（Celery/Redis）并返回任务 ID。

## 测试与质量
```bash
pytest -q backend/tests
pre-commit install && pre-commit run -a
```

## 环境变量
- `MODEL_PATH`：模型路径（默认 `models/iris_rf.pkl`）
- `MODEL_DIR`：模型目录（用于寻找特征名/标签名）
- `NEXT_PUBLIC_API_BASE`：前端调用的后端地址
```

