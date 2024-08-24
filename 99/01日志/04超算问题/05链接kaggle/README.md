


## 目的：使用kaggle下载数据集

```bash
#进入指定的文件夹
cd /work/home/aojiang/06项目复现/08Kansformer/code/data
```



### 需求：从kaggle下载数据集
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


## 报错

```bash
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ kaggle competitions download -c aptos2019-blindness-detection
2024-08-20 17:19:24,513 WARNING Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f3dc3eabd90>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /api/v1/competitions/data/download-all/aptos2019-blindness-detection
2024-08-20 17:19:29,516 WARNING Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f3dc3eab7c0>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /api/v1/competitions/data/download-all/aptos2019-blindness-detection
2024-08-20 17:19:39,526 WARNING Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f3dc3eab520>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /api/v1/competitions/data/download-all/aptos2019-blindness-detection
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connection.py", line 174, in _new_conn
    conn = connection.create_connection(
  File "/opt/conda/lib/python3.10/site-packages/urllib3/util/connection.py", line 72, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
  File "/opt/conda/lib/python3.10/socket.py", line 955, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connectionpool.py", line 703, in urlopen
    httplib_response = self._make_request(
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connectionpool.py", line 386, in _make_request
    self._validate_conn(conn)
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connectionpool.py", line 1042, in _validate_conn
    conn.connect()
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connection.py", line 358, in connect
    self.sock = conn = self._new_conn()
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connection.py", line 186, in _new_conn
    raise NewConnectionError(
urllib3.exceptions.NewConnectionError: <urllib3.connection.HTTPSConnection object at 0x7f3dc3eab130>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/work/home/aojiang/.local/bin/kaggle", line 8, in <module>
    sys.exit(main())
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/cli.py", line 63, in main
    out = args.func(**command_args)
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/api/kaggle_api_extended.py", line 1037, in competition_download_cli
    self.competition_download_files(competition, path, force,
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/api/kaggle_api_extended.py", line 998, in competition_download_files
    self.competitions_data_download_files_with_http_info(
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/api/kaggle_api.py", line 384, in competitions_data_download_files_with_http_info
    return self.api_client.call_api(
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/api_client.py", line 313, in call_api
    return self.__call_api(resource_path, method,
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/api_client.py", line 145, in __call_api
    response_data = self.request(
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/api_client.py", line 335, in request
    return self.rest_client.GET(url,
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/rest.py", line 231, in GET
    return self.request("GET", url,
  File "/work/home/aojiang/.local/lib/python3.10/site-packages/kaggle/rest.py", line 204, in request
    r = self.pool_manager.request(method, url,
  File "/opt/conda/lib/python3.10/site-packages/urllib3/request.py", line 74, in request
    return self.request_encode_url(
  File "/opt/conda/lib/python3.10/site-packages/urllib3/request.py", line 96, in request_encode_url
    return self.urlopen(method, url, **extra_kw)
  File "/opt/conda/lib/python3.10/site-packages/urllib3/poolmanager.py", line 376, in urlopen
    response = conn.urlopen(method, u.request_uri, **kw)
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connectionpool.py", line 815, in urlopen
    return self.urlopen(
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connectionpool.py", line 815, in urlopen
    return self.urlopen(
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connectionpool.py", line 815, in urlopen
    return self.urlopen(
  File "/opt/conda/lib/python3.10/site-packages/urllib3/connectionpool.py", line 787, in urlopen
    retries = retries.increment(
  File "/opt/conda/lib/python3.10/site-packages/urllib3/util/retry.py", line 592, in increment
    raise MaxRetryError(_pool, url, error or ResponseError(cause))
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='www.kaggle.com', port=443): Max retries exceeded with url: /api/v1/competitions/data/download-all/aptos2019-blindness-detection (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f3dc3eab130>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution'))
(base) aojiang@e3654158688d:~/06项目复现/08Kansformer/code/data$ 
```
### 解决方法
错误消息表明系统无法解析域名（"Temporary failure in name resolution"），这通常是因为网络连接问题或DNS配置问题。这可能是由于以下几个原因造成的：

1. **网络连接不稳定或不可用**：确保你的设备已连接到互联网。
2. **代理或VPN**：如果你通过代理或VPN连接到互联网，可能需要正确配置这些服务以确保可以访问外部网站。
3. **DNS服务器问题**：DNS服务器可能无法解析域名。尝试更改你的DNS服务器到Google的DNS（8.8.8.8和8.8.4.4）或其他公共DNS服务。

### 解决步骤

1. **检查网络连接**：
   - 确保你的网络连接正常。尝试访问其他网站或服务以验证网络连接。

2. **检查代理设置**：
   - 如果你使用代理，检查代理设置是否正确。确保命令行工具配置为使用这些代理设置。
   - 在终端运行以下命令查看当前的代理设置：
     ```bash
     echo $http_proxy
     echo $https_proxy
     ```

3. **更改DNS服务器**：
   - 更改DNS设置，使用如Google DNS（8.8.8.8和8.8.4.4）或Cloudflare DNS（1.1.1.1）。
   - 在Linux系统中，你可以通过编辑 `/etc/resolv.conf` 文件来更改DNS服务器：
     ```bash
     sudo nano /etc/resolv.conf
     ```
     添加或修改行：
     ```
     nameserver 8.8.8.8
     nameserver 8.8.4.4
     ```
     保存并退出编辑器。

4. **重新尝试下载**：
   - 在确认网络连接和DNS配置无误后，再次尝试下载数据集：
     ```bash
     kaggle competitions download -c aptos2019-blindness-detection
     ```




### 步骤 4: 解压数据集
下载完成后，需要解压数据集：
```bash
unzip aptos2019-blindness-detection.zip
```








