import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export async function predictJson(payload: { X: number[][], feature_names?: string[] }) {
  const res = await axios.post(`${API_BASE}/predict`, payload);
  return res.data;
}

export async function predictCsv(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await axios.post(`${API_BASE}/predict_csv`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}
