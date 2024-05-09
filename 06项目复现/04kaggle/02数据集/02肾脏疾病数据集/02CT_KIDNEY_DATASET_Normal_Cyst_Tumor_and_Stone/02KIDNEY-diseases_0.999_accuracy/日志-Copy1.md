# 日志
## 第1次2024年5月6日
### 任务
1. 熟悉云服务器的基本使用方法
2. 安装相关的库
```python
pip show numpy || sudo pip install numpy
pip show pandas || sudo pip install pandas
pip show opencv-python || sudo pip install opencv-python
pip show seaborn || sudo pip install seaborn
pip show matplotlib || sudo pip install matplotlib
pip show scikit-image || sudo pip install scikit-image
```

### 详情
由于超算中心未给我的的账户开通网络代理，所以安装失败。
刘老师已帮我联系审核员解决。
如果您需要使用清华大学的镜像源来检查并安装这些库，您可以将每个库的检查和安装命令修改为包含`-i https://pypi.tuna.tsinghua.edu.cn/simple`选项，以指定使用清华镜像。以下是针对您列出的每个库的命令：

1. **NumPy**:
   ```bash
   pip show numpy || sudo pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **Pandas**:
   ```bash
   pip show pandas || sudo pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **OpenCV** (注意：OpenCV的pip包名为`opencv-python`):
   ```bash
   pip show opencv-python || sudo pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **Seaborn**:
   ```bash
   pip show seaborn || sudo pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

5. **Matplotlib**:
   ```bash
   pip show matplotlib || sudo pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

6. **scikit-image** (包含`skimage.io.imread`):
   ```bash
   pip show scikit-image || sudo pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
7. **imbalanced-learn**
    ```bash
    sudo pip install -U imbalanced-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
    
请注意，如果您的网络环境存在大量下载某些较大二进制文件的行为，可能会导致请求被镜像站阻断。如果您遇到此类问题，您可以尝试更换网络环境或客户端，或者联系镜像站的技术支持。

此外，如果您使用的是Python 3.x，并且系统中同时安装了Python 2.x，您可能需要使用`pip3`代替`pip`：

```bash
pip3 show numpy || sudo pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果您不希望使用`sudo`或没有管理员权限，可以使用`--user`选项来在用户目录下安装库：

```bash
pip install --user numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

请根据您的具体情况选择使用`sudo`还是`--user`选项。


### 注意事项
安装包时，一定要在终端中安装！
安装时，使用的命令是`sudo pip install xxx `

例如：`pip show opencv-python || sudo pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple`
## 第2次2024年5月7日
### 任务
1. 安装相关的库
2. 加载数据集
### 详情
白天在上课，又因为网络连接失败的问题耽误了一点时间。解决网络连接失败的方法是结束作业，重新运行。
晚上完成安装相关的库。
### 注意事项
```python
#查看当前工作目录
%pwd
```
结果为`'/work/home/aojiang'`


`image_path = '../aojiang/06test2/ct1/ct2/Normal/Normal.jpg'`
添加相对路径时，一定要把当前工作夹加上去，也就是`aojiang`

如果使用绝对路径，则为：
`image_path = '/work/home/aojiang/06test2/ct1/ct2/Normal/Normal.jpg'`
## 第3次2024年5月9日
### 问题
由于大多数云服务为了节省资源，会将实例的文件系统重置为初始状态。
#### 解决方案一
所以，在某些情况下（例如这份代码运行在宁波超算中心的云服务器上），需要手动保存环境。例如，在Anaconda环境中，您可以使用`conda env export > environment.yml`来保存当前环境的依赖，并使用`conda env create -f environment.yml`来恢复它。
#### 解决方案二
在容器中运行，

#### 解决方案
宁波超算中心的老师，已经帮忙在后台配置好了环境和依赖的包

直接在超算环境下运行的环境也配置好了，需要您把ipynb文件转成py文件就可以提交到后台运行了

加载环境：source ~/software/env.sh
可以通过脚本提交

### 尝试方案一
