太好了，我已经准备好了！咱们从**第 1 周：TypeScript 核心语法上手（结合你的 Model-to-Web 脚手架）**开始。本周目标是把前端改造成**强类型、零 any、可读性高**的代码，并掌握 TS 的最小必备集合。学习计划依据你上传的 12 周路线（第 1 周部分）。

---

# 第 1 周目标（产出导向）

1. 在模板 `frontend/` 中新增 **`types.ts`**，定义请求/响应的精准类型。
2. 把 **`lib/api.ts`** 与 **`app/page.tsx`** 全部**类型化**，杜绝 `any`。
3. 完成 3 个练习（类型收窄、泛型、错误分支）并通过 TS 编译与手测。

---

## Step 0｜本周只需的环境

```bash
# 进入前端目录
cd frontend
npm install
npm run dev
# 浏览器 http://localhost:3000
# （后端按上周脚手架，已提供 /predict 与 /predict_csv）
```

---

## Step 1｜创建全局类型定义（`frontend/types.ts`）

在 `frontend/` 根目录新增文件 `types.ts`：

```ts
// frontend/types.ts
export type PredictRequest = {
  X: number[][];
  feature_names?: string[];
};

export type PredictResponse = {
  y_pred: number[];
  y_proba?: number[][];
  target_names?: string[];
};

// 用于后续错误统一处理的形状（前后端可逐步对齐）
export type ApiError = {
  message: string;
  details?: unknown;
};

// 运行时类型守卫（可在接口返回后做断言）
export function isPredictResponse(x: unknown): x is PredictResponse {
  if (!x || typeof x !== "object") return false;
  const r = x as Record<string, unknown>;
  return Array.isArray(r.y_pred);
}
```

> 说明：**类型先行**。先把契约（请求/响应）写清楚，后写调用代码。

---

## Step 2｜给 API 封装上类型（`frontend/lib/api.ts`）

把原来的 `api.ts` 改成强类型版本（引入泛型 + 统一实例）：

```ts
// frontend/lib/api.ts
import axios, { AxiosInstance, AxiosError } from "axios";
import type { PredictRequest, PredictResponse, ApiError } from "../types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function createClient(): AxiosInstance {
  const client = axios.create({
    baseURL: API_BASE,
    timeout: 15000,
  });

  // 可选：集中错误拦截，返回统一错误形状
  client.interceptors.response.use(
    (res) => res,
    (err: AxiosError): Promise<never> => {
      const apiErr: ApiError = {
        message: err.message || "Request failed",
        details: err.response?.data,
      };
      return Promise.reject(apiErr);
    }
  );
  return client;
}

const http = createClient();

export async function predictJson(payload: PredictRequest): Promise<PredictResponse> {
  const res = await http.post<PredictResponse>("/predict", payload);
  return res.data;
}

export async function predictCsv(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await http.post<PredictResponse>("/predict_csv", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}
```

要点：

* `http.post<PredictResponse>` 使用**响应泛型**，axios 会推断 `res.data` 的类型。
* 统一错误拦截，返回 `ApiError`，方便页面里**类型收窄**。

---

## Step 3｜页面组件全面类型化（`frontend/app/page.tsx`）

把状态、事件、错误处理改为强类型，利用**类型收窄(narrowing)**：

```tsx
'use client';
import { useState } from 'react';
import { predictJson, predictCsv } from '../lib/api';
import type { PredictRequest, PredictResponse, ApiError } from '../types';
import { isPredictResponse } from '../types';

export default function Page() {
  const [jsonInput, setJsonInput] = useState<string>('[[5.1, 3.5, 1.4, 0.2]]');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<ApiError | null>(null);

  const onPredictJson = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const X = JSON.parse(jsonInput) as PredictRequest['X']; // 受控断言：我们知道目标形状
      const data = await predictJson({ X });
      if (!isPredictResponse(data)) {
        setError({ message: 'Unexpected response shape', details: data });
        return;
      }
      setResult(data);
    } catch (e: unknown) {
      // 类型收窄：先判断是否 ApiError
      if (e && typeof e === 'object' && 'message' in e) {
        setError(e as ApiError);
      } else {
        setError({ message: 'Invalid JSON or unknown error', details: e });
      }
    } finally {
      setLoading(false);
    }
  };

  const onPredictCsv = async () => {
    if (!csvFile) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await predictCsv(csvFile);
      setResult(data);
    } catch (e: unknown) {
      if (e && typeof e === 'object' && 'message' in e) {
        setError(e as ApiError);
      } else {
        setError({ message: 'Upload failed', details: e });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Model-to-Web Demo</h1>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">JSON 输入</h2>
        <textarea
          className="w-full h-32 p-2 border rounded"
          value={jsonInput}
          onChange={(e) => setJsonInput(e.target.value)}
        />
        <button className="mt-2 px-4 py-2 border rounded" onClick={onPredictJson} disabled={loading}>
          {loading ? 'Predicting...' : '预测（JSON）'}
        </button>
      </section>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">CSV 上传</h2>
        <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)} />
        <button className="mt-2 px-4 py-2 border rounded" onClick={onPredictCsv} disabled={loading || !csvFile}>
          {loading ? 'Uploading...' : '预测（CSV）'}
        </button>
      </section>

      {error && (
        <div className="text-red-600 text-sm mb-2">
          <div>错误：{error.message}</div>
          {error.details && <pre className="mt-1 p-2 border rounded overflow-auto">{JSON.stringify(error.details, null, 2)}</pre>}
        </div>
      )}

      {result && (
        <section className="mt-4">
          <h2 className="text-lg font-semibold mb-2">结果</h2>
          <pre className="p-2 border rounded overflow-auto text-sm">{JSON.stringify(result, null, 2)}</pre>
        </section>
      )}
    </main>
  );
}
```

要点：

* `useState<类型>` 显式标注，避免隐式 `any`。
* `JSON.parse` 后做**受控断言**（或进一步写运行时校验）。
* 错误统一为 `ApiError`，分支中用**类型收窄**保证安全展示。

---

## Step 4｜（可选加强）运行时校验

如果你希望进一步稳健：引入 `zod`（或轻量自写）对输入做**运行时**校验，把解析和报错信息更友好地呈现。这个可以放到第 2 周组件化时再做，不急。

---

## 本周练习（建议当天完成）

1. **类型收窄练习**：在 `page.tsx` 里新增一个工具函数：

   ```ts
   function isApiError(x: unknown): x is ApiError {
     return !!x && typeof x === 'object' && 'message' in x;
   }
   ```

   并把 `catch` 中的分支改为用 `isApiError(e)` 做判断。

2. **泛型练习**：在 `lib/api.ts` 封装一个通用 requester：

   ```ts
   async function postJSON<TReq, TRes>(url: string, payload: TReq): Promise<TRes> {
     const res = await http.post<TRes>(url, payload);
     return res.data;
   }
   ```

   用它重写 `predictJson`。

3. **错误分支练习**：在后端（FastAPI）临时制造一个错误（比如故意校验失败），确认前端能在 `error.details` 里看到结构化信息，并且 UI 不崩溃。

---

## 本周自测清单（全部通过算完成）

* [ ] `frontend/types.ts` 存在并被 `api.ts`、`page.tsx` 使用
* [ ] `npm run dev` 正常；控制台无类型报错
* [ ] 页面里不出现 `any`（或能解释为少量受控断言）
* [ ] CSV/JSON 两种路径都能得到预测，错误时 UI 能展示清晰信息

---

## 常见坑位 & 快速修复

* **`any` 滥用**：如果 TS 推断不出来，优先**补类型**或**改代码结构**，不要直接上 `any`。
* **`JSON.parse`**：返回 `any`，务必用类型守卫/断言，或外层运行时校验。
* **Axios 泛型**：`http.post<T>` 给的是**响应体类型**，别写成请求体。
* **环境变量**：确保 `NEXT_PUBLIC_API_BASE` 正确；Next 只会将 `NEXT_PUBLIC_*` 注入到浏览器端。

---

## 你现在可以做两件事

1. 按上面的 Step 1–3 把代码改完，跑一遍页面；
2. 告诉我你在第 1 周练习中遇到的任何报错或类型困惑，我会直接给出**最小可行修复**（含代码片段）。

等你完成本周任务后，我们就进入**第 2 周：App Router 组件化 + 错误/加载边界 + 上传进度**。
