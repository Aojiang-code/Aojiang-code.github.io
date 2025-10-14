'use client';
import { useState } from 'react';
import { predictJson, predictCsv } from '../lib/api';

export default function Page() {
  const [jsonInput, setJsonInput] = useState('[[5.1, 3.5, 1.4, 0.2]]');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onPredictJson = async () => {
    setLoading(true); setError(null);
    try {
      const X = JSON.parse(jsonInput);
      const data = await predictJson({ X });
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const onPredictCsv = async () => {
    if (!csvFile) return;
    setLoading(true); setError(null);
    try {
      const data = await predictCsv(csvFile);
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Model-to-Web Demo</h1>
      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">JSON 输入</h2>
        <p className="text-sm mb-2">二维数组（每行一个样本）。Iris 示例：4 个特征。</p>
        <textarea
          className="w-full h-32 p-2 border rounded"
          value={jsonInput}
          onChange={(e) => setJsonInput(e.target.value)}
        />
        <button
          className="mt-2 px-4 py-2 border rounded"
          onClick={onPredictJson}
          disabled={loading}
        >
          {loading ? 'Predicting...' : '预测（JSON）'}
        </button>
      </section>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-2">CSV 上传</h2>
        <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)} />
        <button
          className="mt-2 px-4 py-2 border rounded"
          onClick={onPredictCsv}
          disabled={loading || !csvFile}
        >
          {loading ? 'Uploading...' : '预测（CSV）'}
        </button>
        <p className="text-sm mt-2">示例 CSV 列：sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)</p>
      </section>

      {error && <div className="text-red-600">Error: {error}</div>}
      {result && (
        <section className="mt-4">
          <h2 className="text-lg font-semibold mb-2">结果</h2>
          <pre className="p-2 border rounded overflow-auto text-sm">{JSON.stringify(result, null, 2)}</pre>
        </section>
      )}
    </main>
  );
}
