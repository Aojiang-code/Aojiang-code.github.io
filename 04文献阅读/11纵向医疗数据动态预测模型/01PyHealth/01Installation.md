好的，以下是你提供的“Installation 安装部分”信息整理后的中文学习笔记，可作为你 PyHealth 项目的环境准备部分文档：

---

## 🛠️ PyHealth 安装指南（Installation）

### 📦 一、安装方式

PyHealth 支持两种安装方式：

#### ✅ 方式一：从 PyPI 安装（推荐方式）

```bash
pip install pyhealth
```

#### ✅ 方式二：从 GitHub 源码安装（适合开发者）

```bash
git clone https://github.com/sunlabuiuc/PyHealth.git
cd pyhealth
pip install .
```

---

### 🧾 二、所需依赖（requirements）

| 依赖项          | 版本要求                  |
| ------------ | --------------------- |
| Python       | ≥ 3.8                 |
| PyTorch      | ≥ 1.8.0（需手动安装）        |
| RDKit        | ≥ 2022.03.4（用于药物结构建模） |
| scikit-learn | ≥ 0.24.2              |
| networkx     | ≥ 2.6.3               |
| pandas       | ≥ 1.3.2               |
| tqdm         | 任意版本（用于进度条显示）         |

---

### ⚠️ 三、重要提醒

> ❗ PyHealth 不会自动安装深度学习库（例如 PyTorch），以避免与用户已有版本冲突。

因此，如果你计划使用 **神经网络模型（如 LSTM、GRU、RETAIN、Transformer 等）**，请自行先安装 PyTorch：

```bash
pip install torch torchvision torchaudio
```

或根据显卡选择合适的 CUDA 版本：

```bash
# 以 NVIDIA RTX A6000 显卡 + CUDA 11.3 为例
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

👉 你可以根据以下页面选择适合你设备的安装方式：
🔗 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

### 📁 推荐的虚拟环境配置（完整示例）

```bash
conda create -n pyhealth_env python=3.9 -y
conda activate pyhealth_env

# 安装 PyTorch（根据显卡选合适 CUDA）
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch

# 安装 PyHealth 和其他依赖
pip install pyhealth
pip install rdkit scikit-learn networkx pandas tqdm
```

---

需要我为你生成一份 `.yml` 的 Conda 环境文件，方便一键部署吗？或者生成 Dockerfile？告诉我你的使用环境即可（Windows/macOS/Linux + 是否使用GPU）。
