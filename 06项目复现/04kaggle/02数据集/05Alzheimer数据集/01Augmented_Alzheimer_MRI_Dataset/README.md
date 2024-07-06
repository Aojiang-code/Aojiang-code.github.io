# 01Augmented_Alzheimer_MRI_Dataset

> 网址：[Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data)


* 2022年创建
* 72人投票
* 5,790 downloads

## 相关代码(按vote排序)

### [01Augmented_Alzheimer_MRI_Dataset_with_93.5%](06项目复现\04kaggle\02数据集\05Alzheimer数据集\01Augmented_Alzheimer_MRI_Dataset\01Augmented_Alzheimer_MRI_Dataset_with_93_5\README.md)

> 网址：[Augmented Alzheimer MRI Dataset with 93.5%](https://www.kaggle.com/code/mohamedgobara/augmented-alzheimer-mri-dataset-with-93-5)

* 8个月前发布(也就是2023年12月发布)
* 2,144次浏览
* 46人认可
* 106次复现


### 





要查看当前显卡的详细信息，你可以使用以下方法：

### 在 Linux 系统中：

1. **使用 `lspci` 命令**：
   你已经使用了这个命令，它提供了显卡的基本信息。为了获取更详细的信息，可以添加 `-vvv` 标志来显示所有可用的详细信息：
   ```bash
   !lspci -vvv | grep -i vga
   ```

2. **使用 `lshw` 命令**（如果已安装）：
   `lshw` 可以提供包括显卡在内的硬件详细信息。使用以下命令：
   ```bash
   !sudo lshw -C display
   ```

3. **查看 Xorg 服务器日志**：
   Xorg 日志文件通常包含显卡的配置信息和状态报告。你可以查看 `/var/log/Xorg.0.log` 文件：
   ```bash
   !cat /var/log/Xorg.0.log
   ```

4. **使用 `inxi` 命令**（可能需要安装）：
   `inxi` 是一个全面的系统信息工具，可以提供详细的硬件信息：
   ```bash
   !sudo inxi -Fxz
   ```

5. **使用 `glxinfo` 命令**（需要 Xorg 和 OpenGL 库）：
   ```bash
   !glxinfo | grep -i direct
   ```

6. **查看显卡制造商的官方网站**：
   对于 ASPEED Graphics Family 显卡，你可以访问 [ASPEED 官方网站](https://www.aspeedtech.com/) 来获取更多关于显卡的信息。

7. **使用 `hwinfo` 命令**（可能需要安装）：
   `hwinfo` 是一个显示硬件信息的工具，类似于 `lshw`：
   ```bash
   !sudo hwinfo --short --category display
   ```

8. **使用 `hardinfo` GUI 工具**（可能需要安装）：
   如果你更喜欢图形界面，`hardinfo` 是一个显示硬件信息的 GUI 工具。

### 在 Jupyter Notebook 中：

在 Jupyter Notebook 中，你可以直接运行上述命令（除了那些明确需要图形界面的工具）。例如：

```python
!lspci -vvv | grep -i vga
```

请记住，在 Jupyter Notebook 中执行系统命令时，需要在命令前加上 `!`。

### 注意：

- 某些命令可能需要管理员权限，因此在命令前使用 `sudo` 可能需要输入密码。
- 如果你的系统是无头服务器，某些依赖于图形界面的工具可能不会安装或不包含图形相关的信息。
- 如果你正在使用的是 Docker 容器或某种特殊的虚拟环境，可能需要确保这些工具已经安装在环境中。















根据您提供的 `lspci` 命令输出，您的系统中安装了一个由 ASPEED Technology, Inc. 生产的 VGA 兼容控制器，属于 ASPEED Graphics Family。这个控制器的修订版本号是 41。

 虽然这个命令提供了显卡的基本识别信息，但如果您想获取更详细的信息，可以尝试以下命令：

1. **使用 `lspci` 命令并增加 `-vvv` 选项**：
   ```bash
   lspci -vvv | grep -i vga
   ```
   这将提供包括设备中断、电源管理、错误报告等在内的更详细的信息。

2. **查看显卡的内存信息**：
   如果您想查看显卡的内存大小，可以使用以下命令：
   ```bash
   grep -i "vga compatible controller" /var/log/Xorg.0.log
   ```
   这个命令可能会在 Xorg 服务器的日志文件中找到显卡的内存信息。

3. **使用 `inxi` 命令**（如果已安装）：
   ```bash
   inxi -Fxz | grep -i "vga"
   ```
   `inxi` 可以提供包括显卡在内的详细硬件信息。

4. **使用 `hwinfo` 命令**（如果已安装）：
   ```bash
   sudo hwinfo --short --class display
   ```
   `hwinfo` 可以提供类似于 `lshw` 的硬件信息。

5. **查看显卡驱动模块**：
   显卡驱动可能作为一个内核模块加载，你可以使用 `lsmod` 命令查看：
   ```bash
   lsmod | grep -i aspeed
   ```

6. **查看 BIOS/UEFI 信息**：
   有时 BIOS/UEFI 设置中会显示显卡的详细信息，你可以在启动时进入 BIOS/UEFI 来查看。

7. **查看系统制造商网站**：
   访问 ASPEED 官方网站或你的系统制造商的官方网站，搜索你的显卡型号以获取更多信息。

请注意，某些命令可能需要管理员权限，因此在命令前使用 `sudo` 可能需要输入密码。此外，如果你的系统是无头服务器（没有图形界面），某些依赖于图形界面的工具可能不会安装或不包含图形相关的信息。





















要生成包含代码依赖包的文档，你可以采取以下几种方法：

1. **使用 `pip freeze` 命令**：
   这个命令会列出你当前环境中安装的所有 Python 包及其版本。将这个命令的输出保存到一个文本文件中，可以作为依赖列表。
   ```bash
   !pip freeze > requirements.txt
   ```

2. **创建 `requirements.txt` 文件**：
   如果你已经有了一个 `requirements.txt` 文件，你可以将所有需要的包添加到这个文件中。确保每个包后面都跟着 `==` 和版本号，以确保精确复制环境。

3. **使用 `pipreqs`**：
   `pipreqs` 是一个可以扫描你的项目并自动创建 `requirements.txt` 文件的工具。安装 `pipreqs` 后，你可以使用以下命令：
   ```bash
   pip install pipreqs
   pipreqs /path/to/your/project
   ```
   这将在你的项目目录中生成一个 `requirements.txt` 文件。

4. **手动列出依赖**：
   如果你知道代码依赖哪些包，你可以直接手动创建一个 `requirements.txt` 文件，并列出所有必要的包及其版本。

5. **使用 Docker 创建环境**：
   如果你想要更完整的环境复制，可以考虑使用 Docker。创建一个 `Dockerfile`，列出所有需要的步骤来安装依赖和设置环境。然后，你可以在任何系统上使用这个 `Dockerfile` 来构建镜像并运行你的代码。

6. **使用 Conda 环境**（如果你使用 Anaconda）：
   如果你使用的是 Anaconda，你可以列出环境中的所有包，并将它们保存到一个文件中：
   ```bash
   conda list --export > environment.yml
   ```
   然后，你可以将这个 `environment.yml` 文件分享给他人，他们可以使用它来创建相同环境：
   ```bash
   conda env create -f environment.yml
   ```

7. **文档化依赖**：
   在文档中详细记录所有依赖的包及其版本，以及安装这些包的步骤。这可以是一个简单的文本文件或 Markdown 文件。

8. **使用虚拟环境**：
   使用 Python 的虚拟环境（venv）或虚拟环境包装器（virtualenv）来创建一个隔离的环境，并记录下该环境中安装的所有包。

选择适合你需求的方法，并确保在不同设备上复制环境时遵循相同的步骤。这样，你就可以确保代码在不同的环境中都能正常运行。
















Index(['Age', 'Gender', 'Hypertension', 'Diabetes', 'CHF', 'COPD', 'Self-Care',
       '5-MFI', 'Osteoporosis', 'BMI', 'PNI', 'Albumin',
       'Absolute Lymphocyte Count', 'CRP', 'Hb', 'Lactate', 'Surgery Duration',
       'Anesthesia Type', 'ASA', 'Hospital Stay Duration',
       'Perioperative Blood Product Preparation', 'Hospitalization Cost',
       'ADL Preoperative', 'ADL Postoperative',
       'Lower Extremity Deep Venous Thrombosis', 'Delirium', 'Heart Failure',
       'Respiratory Failure', 'Gastrointestinal Hemorrhage',
       'Renal Insufficiency', 'Urinary Tract Infection',
       'Wound Redness and Swelling', 'Respiratory System Infection',
       'Total Complications', 'Mortality', 'ICU', 'Unplanned Readmission',
       'class'],
      dtype='object')





# Categorical columns (Variables that can be categorized into groups)
cat_cols = [
    'Gender',          # Gender (Male 1, Female 2)
    'Hypertension',     # Hypertension (Binary: 0 or 1)
    'Diabetes',         # Diabetes (Binary: 0 or 1)
    'CHF',              # Congestive Heart Failure (Binary: 0 or 1)
    'COPD',             # Chronic Obstructive Pulmonary Disease (Binary: 0 or 1)
    'Self-Care',        # Self-Care (Binary: 0 or 1)
    'Osteoporosis',     # Osteoporosis (Binary: 0 or 1)
    'Anesthesia Type',  # Anesthesia Type (General 1, Regional 2)
    'Perioperative Blood Product Preparation', # Perioperative Blood Product Use (Binary: 0 or 1)
    'Lower Extremity Deep Venous Thrombosis', # Binary: 0 or 1
    'Delirium',         # Binary: 0 or 1
    'Heart Failure',     # Binary: 0 or 1
    'Respiratory Failure', # Binary: 0 or 1
    'Gastrointestinal Hemorrhage', # Binary: 0 or 1
    'Renal Insufficiency', # Binary: 0 or 1
    'Urinary Tract Infection', # Binary: 0 or 1
    'Wound Redness and Swelling', # Binary: 0 or 1
    'Respiratory System Infection', # Binary: 0 or 1
    'Total Complications', # Binary: 0 or 1
    'Mortality',         # Binary: 0 or 1
    'ICU',              # Binary: 0 or 1
    'Unplanned Readmission' # Binary: 0 or 1
]

# Numerical columns (Variables that represent quantities or measurements)
num_cols = [
    'Age',
    '5-MFI',            # A specific measure of physical function
    'BMI',              # Body Mass Index
    'PNI',              # Prognostic Nutritional Index
    'Albumin',
    'Absolute Lymphocyte Count',
    'CRP',              # C-Reactive Protein
    'Hb',        # Hemoglobin
    'Lactate',
    'Surgery Duration',
    'ASA',        # American Society of Anesthesiologists Score
    'Hospital Stay Duration',
    'Hospitalization Cost',
    'ADL Preoperative', # Activities of Daily Living
    'ADL Postoperative'  # Activities of Daily Living
]