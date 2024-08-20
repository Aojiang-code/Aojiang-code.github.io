# 使用kaggle下载数据集

## 需求：从kaggle下载数据集
我希望在终端中下载kaggle上的一个数据集到当前工作目录，这个数据集的网址为https://www.kaggle.com/competitions/aptos2019-blindness-detection/data?select=train_images

在上述网址中包含下载命令：

```bash
kaggle competitions download -c aptos2019-blindness-detection
```

## 一、设置 API 密钥

要在终端中下载 Kaggle 的数据集，首先需要确保已经安装了 Kaggle API 并正确设置了 API 密钥。

### 步骤 1: 安装 Kaggle API
如果你还没有安装 Kaggle API，可以通过以下命令安装：
```bash
pip install kaggle
```

### 步骤 2: 配置 Kaggle API 密钥
1. 在 Kaggle 网站上，登录你的账户。
2. 转到你的账户设置页面（点击你的头像，然后选择 "Account"）。
3. 在 "API" 部分中，点击 "Create New API Token"。这将下载一个 `kaggle.json` 文件到你的电脑。
4. 将下载的 `kaggle.json` 文件放到你的用户目录下的 `.kaggle` 文件夹中。如果这个文件夹不存在，你需要手动创建它。路径应该是这样的：
   - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`

确保这个文件的权限是安全的，特别是在 Linux 或 Mac 上，你可以通过运行以下命令设置权限：
```bash
chmod 600 ~/.kaggle/kaggle.json
```



## 删除 `.kaggle` 文件夹
现在我想通过终端删除 `.kaggle` 文件夹，请问应该在终端中输入什么命令？


要通过终端删除 `.kaggle` 文件夹，你可以使用以下命令。请确保你确实想删除此文件夹，因为这将删除所有相关的 Kaggle API 配置和凭证。

如果你使用的是 Linux 或 Mac，可以在终端中输入以下命令：

```bash
rm -rf ~/.kaggle
```

这个命令的含义是：
- `rm`: 删除文件或目录的命令。
- `-rf`: 参数 `r` 表示递归删除，用于删除目录及其内容；`f` 表示强制执行，不会询问确认。


## 二、添加环境变量，以解决`kaggle: command not found`

```bash
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ kaggle competitions download -c aptos2019-blindness-detection
bash: kaggle: command not found
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ 
请帮助我解决上述问题
```


由于系统无法找到 `kaggle` 命令，我们需要确保 `kaggle` API 工具已经安装，并且其可执行文件路径被正确添加到你的 `PATH` 环境变量中。下面是一步一步的解决方案：

### 确认 Kaggle 工具已安装

1. **检查 Kaggle 是否已安装**：
   在终端运行以下命令来确认 Kaggle CLI 是否已经安装：
   ```bash
   pip show kaggle
   ```

   如果这个命令返回了 Kaggle 的安装信息，那么 Kaggle API 是安装了的。如果没有，你需要运行以下命令来安装：
   ```bash
   pip install kaggle
   ```



```bash
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ pip show kaggle

Name: kaggle
Version: 1.6.17
Summary: Kaggle API
Home-page: https://github.com/Kaggle/kaggle-api
Author: Kaggle
Author-email: support@kaggle.com
License: Apache 2.0
Location: /work/home/aojiang/.local/lib/python3.10/site-packages
Requires: bleach, certifi, python-dateutil, python-slugify, requests, six, tqdm, urllib3
Required-by: 
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ 
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$
```







从你提供的信息来看，`kaggle` 命令的安装位置在 `/work/home/aojiang/.local/lib/python3.10/site-packages`。通常，可执行文件应该在 `~/.local/bin` 下面。我们需要确认 `kaggle` 可执行文件是否在这个路径下，然后将其添加到你的 `PATH` 环境变量中。

### 配置环境变量：

1. **确认 `kaggle` 可执行文件的位置**：
   在你的终端中运行下面的命令来查找 `kaggle` 可执行文件：
   ```bash
   find ~/.local -name kaggle
   ```



```bash
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ find ~/.local -name kaggle


/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle
/work/home/aojiang/.local/bin/kaggle
```






上述信息显示了 `kaggle` 可执行文件的确切位置，即 `/work/home/aojiang/.local/bin/kaggle`。现在，需要确保这个路径被添加到 `PATH` 环境变量中。



### 使用 `vi` 或 `vim` 编辑 `.bashrc`

1. **打开 `.bashrc` 文件**：
   在终端中输入以下命令来使用 `vi` 编辑器打开 `.bashrc` 文件：
   ```bash
   vi ~/.bashrc
   ```

   如果你更喜欢使用 `vim`（如果已安装），可以用 `vim` 替换 `vi` 命令。

2. **编辑文件**：
   - 在 `vi` 或 `vim` 中，首先按 `i` 进入插入模式。
   - 然后滚动到文件的底部并添加以下行：
     ```bash
    export PATH="/work/home/aojiang/.local/bin:$PATH"
     ```
   - 添加完成后，按 `Esc` 退出插入模式。

3. **保存并退出编辑器**：
   - 输入 `:wq`（意味着写入并退出）然后按 `Enter`。

4. **使更改生效**：
   - 回到终端，输入以下命令使更改立即生效：
     ```bash
     source ~/.bashrc
     ```

现在，已经成功将 `kaggle` 的路径添加到了 `PATH` 环境变量中，且可以在任何地方运行 `kaggle` 命令。


在 `vi` 或 `vim` 编辑器中保存并退出可以通过以下步骤完成：

1. **退出插入模式**：
   如果你正在编辑（插入模式），首先需要按 `Esc` 键退出编辑模式，返回到命令模式。

2. **保存文件**：
   在命令模式中，输入 `:w` （意味着 "write"）命令来保存文件。

3. **退出编辑器**：
   继续输入 `:q` （意味着 "quit"）命令来退出 `vi` 或 `vim`。

4. **组合命令**：
   你可以将保存和退出的命令组合为一个步骤，输入 `:wq` 命令，然后按 `Enter`。这条命令同时保存更改并退出编辑器。

如果你想放弃所有更改并退出，可以使用 `:q!` 命令，其中 `!` 表示强制执行而不保存更改。

简单来说，只需要这样做：
- 按 `Esc` 退出编辑模式。
- 输入 `:wq` 然后按 `Enter` 键来保存并退出。

这样就可以保存在 `.bashrc` 文件中所做的更改并退出编辑器了。




## 三、再次尝试下载数据集
   

1. **验证 `kaggle` 命令**：
   现在尝试运行：
   ```bash
   kaggle --version
   ```


```bash
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ kaggle --version
Kaggle API 1.6.17
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ 
```


   如果这个命令输出了 `kaggle` 的版本信息，那么你已经成功地将 `kaggle` 配置为了全局命令。

2. **再次尝试下载数据集**：
   使用以下命令尝试下载数据集：
   

```bash
   kaggle competitions download -c aptos2019-blindness-detection
   ```
这个命令会将数据集下载到当前工作目录下的 `aptos2019-blindness-detection.zip` 文件中。

### 步骤 4: 解压数据集
下载完成后，你可能需要解压数据集：
```bash
unzip aptos2019-blindness-detection.zip
```








