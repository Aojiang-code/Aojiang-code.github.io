# 使用1-Dimensional Inception Time Convolutional Neural Networks进行心音图分类：George B. Moody PhysioNet Challenge 2022

## 团队名称：Simulab
------------------------------------------------
![alt text](06项目复现\02心脏杂音\01Heart-murmur-detection-2022-Simulab-main\注释版Heart-murmur-detection-2022-Simulab-main\Heart-murmur-detection-2022-Simulab-main\注释集合\CNN_architecture.png)

#### 两个分类器的1-D CNN Inception时间架构

这个仓库与以下论文相关联：
[*Phonocardiogram Classification Using 1-Dimensional Inception Time Convolutional Neural Networks*](https://ieeexplore.ieee.org/document/10081878)

原始论文的完整引用：

> B. -J. Singstad, A. M. Gitau, M. K. Johnsen, J. Ravn, L. A. Bongo 和 H. Schirmer, "Phonocardiogram Classification Using 1-Dimensional Inception Time Convolutional Neural Networks," 2022 Computing in Cardiology (CinC), Tampere, Finland, 2022, pp. 1-4, doi: 10.22489/CinC.2022.108.

## 团队成员：

- Markus Johnsen
- Johan Ravn
- Lars Ailo Bongo
- Henrik Schirmer
- Antony M. Gitau
- Bjørn-Jostein Singstad
----------------------------------------------------------------


## 这个仓库包含什么？

这个仓库包含了我们对George B. Moody PhysioNet Challenge 2022的贡献代码。这个挑战的目标是从胸部多个听诊位置收集的数字听诊器的心音图（PCG）中识别出杂音的存在、不存在或不清晰，并预测正常或异常的临床结果。

我们训练并测试了两个1-D卷积神经网络（CNN）在一个来自1568名儿童的PCG数据集（5272个PCG）上。一个模型预测杂音，另一个模型预测临床结果。两个模型都被训练为给出录音级别的预测，而最终的预测是针对每个病人（病人级别预测）。

我们的团队Simulab训练了一个临床结果分类器，在验证集上取得了8720的挑战成本分数（在305个提交中排名第1），而杂音分类器在验证集上取得了0.585的加权准确率（在305个提交中排名第182）。

## 仓库中的文件和文件夹简介

### Python脚本

这个仓库包含了一些我们可以编辑的脚本，以及George B. Moody PhysioNet Challenge 2022组织者提供的不应该被编辑的脚本。

可以编辑的：

- cross_validate.py
- team_code.py

不应该被编辑的：

- evaluate_model.py
- helper_code.py
- run_model.py
- train_model.py

### 笔记本

仓库还包含了Jupyter Notebooks，这使得实验数据、模型和其他参数更加容易。一些笔记本设计用于在Google Colab中使用，以便获得GPU访问权限，而一些笔记本可以在本地计算机上运行。

Colab:

- `5-fold cross-validation.ipynb` # 包含在训练集上交叉验证模型的代码
- `Train and test model.ipynb` # 包含在完整训练集上训练模型、保存权重，然后在训练集上测试模型并最终评估预测的代码
- `pretrain model on 2015 dataset.ipynb` # 包含在PhysioNet Challenge 2016的心音图数据上预训练模型的代码。返回一个.h5文件

Local:

- `EDA-Phonocardiogram-dataset.ipynb` # 对数据集进行探索性数据分析

### 其他文件

这个仓库还包含了以下文件：
- .gitignore
- LICENSE
- Dockerfile
- requirements.txt

Dockerfile和requirements.txt非常重要，因为它们用于构建提交给挑战组织者的Docker镜像。

## 数据

主要数据来源是[CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.0/)数据集。在使用Colab中的Jupyter Notebooks时，我们每次开始新会话时都需要下载这个数据集。

为了加快下载速度，我们创建了这个[数据集的Kaggle版本](https://www.kaggle.com/datasets/bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2)。

要下载数据集，你需要注册一个Kaggle账户，从你的Kaggle个人资料中获取一个kaggle.json文件（包含API令牌），并将其上传到Colab的临时文件夹的根目录。

为了预训练模型，我们使用[PhysioNet Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/), also available on [Kaggle](https://www.kaggle.com/datasets/bjoernjostein/physionet-challenge-2016)的开放数据集，也在Kaggle上有提供。

## 依赖项

- numpy==1.21.6
- scipy==1.4.1
- scikit-learn==0.23.2
- joblib==0.17.0
- tensorflow == 2.8.2
- Cython==0.29.24
- pandas==1.3.2
- h5py==2.10.0
- tqdm==4.54.0

## 在本地运行代码：

你可以尝试通过在挑战训练集上运行以下命令来尝试。这些命令应该在最近的个人电脑上从开始到结束只需要几分钟或更少的时间。

例如，我们实现了一个带有多个特征的随机森林分类器。你可以为你的参赛作品使用不同的分类器、特征和库。这个简单的例子设计得不好，所以**不应该**作为你模型性能的基准。

这段代码使用了四个主要脚本，下面将进行描述，用于训练和运行2022年挑战的模型。

## 使用Docker运行代码

你可以安装这些脚本的依赖项，通过创建一个Docker镜像（见下文）并运行。

    pip install requirements.txt

你可以训练并运行你的模型，运行以下命令：

    python train_model.py training_data model
    python run_model.py model test_data test_outputs

其中`training_data`是包含训练数据文件的文件夹，`model`是保存你的模型的文件夹，`test_data`是包含测试数据文件的文件夹（你可以使用训练数据进行调试和交叉验证），`test_outputs`是保存你的模型输出的文件夹。[2022年挑战网站](https://physionetchallenges.org/2022/)提供了一个训练数据库，描述了数据文件的内容和结构。

你可以通过运行以下命令来评估你的模型：

    python evaluate_model.py labels outputs scores.csv class_scores.csv

其中`labels`是包含数据标签的文件夹，例如`PhysioNet`网页上的培训数据库；`outputs`是包含你模型输出文件的文件夹；`scores.csv`（可选）是你的模型的分数集合；`class_scores.csv`（可选）是你的模型的每类分数集合。

## 如何在Docker中运行这些脚本？

Docker和类似平台允许你将代码与特定依赖项一起容器化和打包，以便你可以在其他计算环境和操作系统中可靠地运行你的代码。

为了确保我们可以运行你的代码，请[安装](https://docs.docker.com/get-docker/)Docker，从你的代码构建一个Docker镜像，并在训练数据上运行它。为了快速检查你的代码是否有错误，你可能希望在训练数据的小子集上运行它。

如果你在运行代码时遇到问题，那么请尝试以下步骤来运行示例代码。

1. 在你的主目录中创建一个名为`example`的文件夹，并在其中创建几个子文件夹。

    user@computer:~$ cd ~/
    user@computer:~$ mkdir example
    user@computer:~$ cd example
    user@computer:~/example$ mkdir training_data test_data model test_outputs
    
2. 从[挑战网站](https://physionetchallenges.org/2022)下载训练数据。将一些训练数据放入`training_data`和`test_data`。你可以使用一些训练数据来检查你的代码（并且应该在训练数据上执行交叉验证以评估你的算法）。

3. 在你的终端下载或克隆这个仓库。

    user@computer:~/example$ git clone https://github.com/physionetchallenges/python-classifier-2022.git

4. 在你的终端构建一个Docker镜像并在其中运行示例代码。

    user@computer:~/example$ ls
    model  python-classifier-2022  test_data  test_outputs  training_data
    
    user@computer:~/example/python-classifier-2022$ docker build -t image .
    ...
    Successfully tagged image:latest
    
    user@computer:~/example/python-classifier-2022$ docker run -it -v ~/example/model:/physionet/model -v ~/example/test_data:/physionet/test_data -v ~/example/test_outputs:/physionet/test_outputs -v ~/example/training_data:/physionet/training_data image bash
    
    root@[...]:/physionet# ls
    Dockerfile             README.md         test_outputs
    evaluate_model.py      requirements.txt  training_data
    helper_code.py         team_code.py      train_model.py
    LICENSE                run_model.py
    
    root@[...]:/physionet# python train_model.py training_data model
    
    root@[...]:/physionet# python run_model.py model test_data test_outputs
    
    root@[...]:/physionet# python evaluate_model.py model test_data test_outputs
    ...
    
    root@[...]:/physionet# exit
    Exit


上述内容是一个在Linux终端中使用Docker的示例，包括构建Docker镜像和在镜像中运行Python脚本的步骤。下面是逐行的详细解释：

### 1. 查看当前目录内容

```bash
user@computer:~/example$ ls
```

这行命令显示了用户当前所在的目录（`~/example`）中的文件和文件夹列表。在这个目录下，有名为`model`、`python-classifier-2022`、`test_data`、`test_outputs`和`training_data`的文件夹。

### 2. 构建Docker镜像

```bash
user@computer:~/example/python-classifier-2022$ docker build -t image .
```

用户切换到包含Python分类器代码的目录（`python-classifier-2022`），然后使用`docker build`命令来构建一个新的Docker镜像。`-t image`指定了新镜像的名称为`image`，`.`表示Docker会在这个目录下查找名为`Dockerfile`的文件来构建镜像。

### 3. 构建成功

```plaintext
...
Successfully tagged image:latest
```

构建过程结束后，Docker会输出一条消息，表明镜像已经成功创建，并且默认被打上了`latest`标签。

### 4. 运行Docker容器

```bash
user@computer:~/example/python-classifier-2022$ docker run -it -v ~/example/model:/physionet/model -v ~/example/test_data:/physionet/test_data -v ~/example/test_outputs:/physionet/test_outputs -v ~/example/training_data:/physionet/training_data image bash
```

用户使用`docker run`命令来启动一个新的Docker容器。这个命令包含了几个参数：

- `-it`：这个组合参数允许用户以交互模式运行容器，并且分配一个伪终端。
- `-v`：这个参数用于挂载宿主机的目录到容器内部。在这个例子中，宿主机的`model`、`test_data`、`test_outputs`和`training_data`目录被挂载到容器内的相应路径。
- `image`：这是之前构建的Docker镜像的名称。
- `bash`：这是要在容器内部执行的命令，即启动bash shell。

### 5. 查看容器内部内容

```plaintext
root@[...]:/physionet# ls
```

在容器内部，用户使用`ls`命令查看当前目录（`/physionet`）下的文件和文件夹。这些文件和文件夹是从宿主机挂载过来的。

### 6. 在容器内运行Python脚本

用户在容器内部执行了三个Python脚本，这些脚本可能是用于训练模型、运行模型和评估模型的：

```plaintext
python train_model.py training_data model
python run_model.py model test_data test_outputs
python evaluate_model.py model test_data test_outputs
```

这些命令调用了Python脚本，传入了必要的参数，如数据路径和模型输出路径。

### 7. 退出容器

```plaintext
root@[...]:/physionet# exit
Exit
```

用户在容器内部输入`exit`命令退出了bash shell。这会导致容器停止运行。在Docker中，容器的状态会被保存，所以用户可以在以后重新启动容器，继续之前的工作。


而在Windows终端（假设你使用的是Windows 10的PowerShell或者Windows Subsystem for Linux (WSL)）中运行上述Docker命令，步骤和Linux终端中的类似，但有一些小的差异。以下是在Windows环境中运行上述Docker命令的步骤：

1. **打开Windows终端**：
   - 如果你使用的是PowerShell，可以直接打开它。
   - 如果你使用的是WSL，可以通过在开始菜单中搜索并打开它。

2. **导航到项目目录**：
   - 使用`cd`命令切换到包含Python分类器代码的目录。例如：
     ```powershell
     cd C:\path\to\example\python-classifier-2022
     ```

3. **构建Docker镜像**：
   - 在PowerShell中，你可以使用与Linux相同的命令来构建Docker镜像：
     ```powershell
     docker build -t image .
     ```
   - 如果你使用的是WSL，命令也是相同的。

4. **运行Docker容器**：
   - 在PowerShell中，挂载卷的路径需要使用Windows风格的路径，并且可能需要使用`\\`来转义反斜杠。例如：
     ```powershell
     docker run -it -v C:\path\to\example\model:/physionet/model -v C:\path\to\example\test_data:/physionet/test_data -v C:\path\to\example\test_outputs:/physionet/test_outputs -v C:\path\to\example\training_data:/physionet/training_data image bash
     ```
   - 如果你使用的是WSL，路径可以直接使用Linux风格的路径，就像在Linux终端中一样。

5. **在容器内执行命令**：
   - 一旦容器启动，你可以像在Linux终端中一样执行Python脚本。

6. **退出容器**：
   - 输入`exit`命令退出容器。

请注意，如果你在Windows上没有安装Docker，你需要先下载并安装Docker Desktop。Docker Desktop for Windows提供了一个Docker命令行界面，允许你在Windows上运行Docker命令。如果你使用的是WSL，Docker命令的语法和Linux上的基本相同。




## 结果

| 模型 | 最佳参数 | 指标 | 训练 | 验证 | 测试 |
|:-----:|:-----------------:|:---------:|:----------:|:------------:|:----:|
| 杂音 | Adam优化 | 加权准确率 | 0.497 ± 0.083 | 0.585 | 0.593 |
|| 加权分类交叉熵 | 挑战指标 | 13158 ± 1283 | 8866 | 13134 |
|| 20批大小 | 准确率 | 0.446 ± 0.070 | 0.423 | 0.497 |
|| 30个周期 | F测量 | 0.403 ± 0.055 | 0.384 | 0.398 |
| 临床 | Adam优化 | 加权准确率 | 0.713 ± 0.042 | 0.732 | 0.703 |
|| 20批大小 | 挑战指标 | 12315 ± 903 | 8720
| | 20个周期 | 准确率 | 0.51 ± 0.047 | 0.537 | 0.537 |
|| 加权分类交叉熵 | F测量 | 0.465 ± 0.061 | 0.530 | 0.503 |


# Citation
```
@inproceedings{singstad_2022_phono,
  title={Phonocardiogram classification using 1-dimensional inception time convolutional neural networks},
  author={Singstad, Bj{\o}rn-Jostein and Gitau, Antony M and Johnsen, Markus Kreutzer and Ravn, Johan and Bongo, Lars Ailo and Schirmer, Henrik},
  booktitle={2022 Computing in Cardiology (CinC)},
  volume={498},
  pages={1--4},
  year={2022},
  organization={IEEE}
}
```

