# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# 加载模型、标准化器和特征列
model_path = "E:/JA/github_blogs/blogs-master/04文献阅读/05肾病/03机器学习预测慢性肾病的终末期肾病/02app/models/best_model.pkl"
scaler_path = "E:/JA/github_blogs/blogs-master/04文献阅读/05肾病/03机器学习预测慢性肾病的终末期肾病/02app/models/scaler.pkl"
features_path = "E:/JA/github_blogs/blogs-master/04文献阅读/05肾病/03机器学习预测慢性肾病的终末期肾病/02app/models/features.pkl"


best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
features = joblib.load(features_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_data_scaled)
    return jsonify({"prediction": "ESKD+" if prediction[0] == 1 else "ESKD-"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)