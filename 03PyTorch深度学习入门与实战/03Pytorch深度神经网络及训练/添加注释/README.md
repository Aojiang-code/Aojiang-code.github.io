# Pytorch深度神经网络及训练(有注释)

```python
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```

## 3.2 Pytorch中的优化器

介绍优化器的常用使用方法

```python
import torch
import torch.nn as nn
from torch.optim import Adam
```


```python
## 建立一个测试网络
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        ## 定义隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(13,10),
            nn.ReLU(),
        )
        ## 定义预测回归层
        self.regression = nn.Linear(10,1)
    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        ## 输出为output
        return output
        
## 输出我们的网络结构
testnet = TestNet()
print(testnet)
```


```python
## 使用方式1
optimizer = Adam(testnet.parameters(),lr=0.001)  
```


```python
## 使用方式2：为不同的层定义不同的学习率
optimizer = Adam(
    [{"params":testnet.hidden.parameters(),"lr":0.0001},
    {"params":testnet.regression.parameters(),"lr": 0.01}],
    lr=1e-2)

## 这意味着testnet.hidden的参数将会使用0.0001的学习率，
## testnet.regression的参数将会使用0.01的学习率，
## 而且lr=1e-2将作用于其它没有特殊指定的所有参数。
```


```python
# ## 注意该段程序并不能运行成功，作为示例使用
# ## 对目标函数进行优化时通常的格式
# for input, target in dataset:
#     optimizer.zero_grad()        ## 梯度清零
#     output = testnetst(input)    ## 计算预测值
#     loss = loss_fn(output, target)## 计算损失
#     loss.backward()               ## 损失后向传播
#     optimizer.step()              # 更新网络参数
```

## 3.5 参数初始化方法

**针对一个层的权重初始化方法**

**针对一个网络的权重初始化方法**

```python
## 针对一个层的权重初始化方法
conv1 = torch.nn.Conv2d(3,16,3)
## 使用标准正态分布分布初始化权重
torch.manual_seed(12)  ## 随机数初始化种子
torch.nn.init.normal(conv1.weight,mean=0,std=1)
```


```python
## 使用直方图可视化conv1.weight的分布情况
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.hist(conv1.weight.data.numpy().reshape((-1,1)),bins = 30)
plt.show()
```


```python
## 使用指定值初始化偏置
torch.nn.init.constant(conv1.bias,val=0.1)
```
####  针对一个网络的权重初始化方法

定义一个TestNet()网络类为例

```python
## 建立一个测试网络
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.hidden = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
        )
        self.cla = nn.Linear(50,10)
    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0],-1)
        x = self.hidden(x)
        output = self.cla(x)
        return output
        
## 输出我们的网络结构
testnet = TestNet()
print(testnet)

```


```python
## 定义为网络中的没个层进行权重初始化的函数
def init_weights(m):
    ## 如果是卷积层
    if type(m) == nn.Conv2d:
        torch.nn.init.normal(m.weight,mean=0,std=0.5)
    ## 如果是全连接层
    if type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight,a=-0.1,b=0.1)
        m.bias.data.fill_(0.01)
        

## 使用网络的apply方法进行权重初始化
torch.manual_seed(13)  ## 随机数初始化种子
testnet.apply(init_weights)
testnet.cla.weight.data
```
## 3.6 Pytorch中定义网络的方式
### 数据准备

```python
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

```


```python
## 读取数据
boston_X,boston_y = load_boston(return_X_y=True)
print("boston_X.shape:",boston_X.shape)
```


```python
plt.figure()
plt.hist(boston_y,bins=20)
plt.show()
```


```python
## 数据标准化处理
ss = StandardScaler(with_mean=True,with_std=True)
boston_Xs = ss.fit_transform(boston_X)
# boston_ys = ss.fit_transform(boston_y)
np.mean(boston_Xs,axis=0)
np.std(boston_Xs,axis=0)
```


```python
## 将数据预处理为可以使用pytorch进行批量训练的形式
## 训练集X转化为张量
train_xt = torch.from_numpy(boston_Xs.astype(np.float32))
## 训练集y转化为张量
train_yt = torch.from_numpy(boston_y.astype(np.float32))
## 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(train_xt,train_yt)
## 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=128, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 1, # 使用两个进程 
)

# ##  检查训练数据集的一个batch的样本的维度是否正确
# for step, (b_x, b_y) in enumerate(train_loader):  
#     if step > 0:
#         break
# ## 输出训练图像的尺寸和标签的尺寸，都是torch格式的数据
# print(b_x.shape)
# print(b_y.shape)
```
### 使用继承Module

```python
## 使用继承Module的形式定义全连接神经网络
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel,self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(
            in_features = 13, ## 第一个隐藏层的输入，数据的特征数
            out_features = 10,## 第一个隐藏层的输出，神经元的数量
            bias=True, ## 默认会有偏置
        )
        self.active1 = nn.ReLU()
        ## 定义第一个隐藏层
        self.hidden2 = nn.Linear(10,10)
        self.active2 = nn.ReLU()
        ## 定义预测回归层
        self.regression = nn.Linear(10,1)

    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        output = self.regression(x)
        ## 输出为output
        return output
        
## 输出我们的网络结构
mlp1 = MLPmodel()
print(mlp1)
```


```python
## 对回归模型mlp1进行训练并输出损失函数的变化情况
# 定义优化器和损失函数
optimizer = SGD(mlp1.parameters(),lr=0.001)  
loss_func = nn.MSELoss()  # 最小平方根误差
train_loss_all = [] ## 输出每个批次训练的损失函数
## 进行训练，并输出每次迭代的损失函数
for epoch in range(30):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):  
        output = mlp1(b_x).flatten()      # MLP在训练batch上的输出
        train_loss = loss_func(output,b_y) # 平方根误差
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        train_loss.backward()           # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        train_loss_all.append(train_loss.item())
```


```python
plt.figure()
plt.plot(train_loss_all,"r-")
plt.title("Train loss per iteration")
plt.show()
```
### 使用nn.Sequential

```python
## 使用定义网络时使用nn.Sequential的形式
class MLPmodel2(nn.Module):
    def __init__(self):
        super(MLPmodel2,self).__init__()
        ## 定义隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(13, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
        )
        ## 预测回归层
        self.regression = nn.Linear(10,1)

    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output
        
## 输出我们的网络结构
mlp2 = MLPmodel2()
print(mlp2)
```



```python
## 对回归模型mlp2进行训练并输出损失函数的变化情况
# 定义优化器和损失函数
optimizer = SGD(mlp2.parameters(),lr=0.001)  
loss_func = nn.MSELoss()  # 最小平方根误差
train_loss_all = [] ## 输出每个批次训练的损失函数
## 进行训练，并输出每次迭代的损失函数
for epoch in range(30):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):  
        output = mlp2(b_x).flatten()               # MLP在训练batch上的输出
        train_loss = loss_func(output,b_y) # 平方根误差
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        train_loss.backward()           # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        train_loss_all.append(train_loss.item())
```


```python
plt.figure()
plt.plot(train_loss_all,"r-")
plt.title("Train loss per iteration")
plt.show()
```
## Pytorch模型保存和加载方法
### 方法1:保存整个模型
```python
## 保存整个模型
torch.save(mlp2,"data/chap3/mlp2.pkl")
```


```python
## 导入保存的模型
mlp2load = torch.load("data/chap3/mlp2.pkl")
mlp2load
```
### 方法2:只保存模型的参数

```python
torch.save(mlp2.state_dict(),"data/chap3/mlp2_param.pkl")
```


```python
## 导入保存的模型的参数
mlp2param = torch.load("data/chap3/mlp2_param.pkl")
mlp2param
```

