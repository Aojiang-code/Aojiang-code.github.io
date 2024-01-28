## 第五章 全连接神经网络 分类


```python
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```


```python
## 导入本章所需要的模块
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.manifold import TSNE


import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data

import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from torchviz import make_dot
```
## 垃圾邮件数据分类


| SPAM E-MAIL DATABASE ATTRIBUTES (in .names format)

| 48 continuous real [0,100] attributes of type word_freq_WORD 


| 6 continuous real [0,100] attributes of type char_freq_CHAR

| 1 continuous real [1,...] attribute of type capital_run_length_average
| = average length of uninterrupted sequences of capital letters

| 1 continuous integer [1,...] attribute of type capital_run_length_longest
| = length of longest uninterrupted sequence of capital letters

| 1 continuous integer [1,...] attribute of type capital_run_length_total
| = sum of length of uninterrupted sequences of capital letters
| = total number of capital letters in the e-mail

| 1 nominal {0,1} class attribute of type spam
| = denotes whether the e-mail was considered spam (1) or not (0), 


### 数据准备
```python
## 读取数据显示数据的前几行
spam = pd.read_csv("data/chap5/spambase.csv")
spam.head()
```


```python
## 计算垃圾邮件和非垃圾邮件的数量
pd.value_counts(spam.label)

## 垃圾邮件有1813个样本，非垃圾邮件有2788个样本
```


```python
## 将数据随机切分为训练集和测试集
X = spam.iloc[:,0:57].values   
y = spam.label.values
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.25, random_state=123
)

## 对数据的前57列特征进行数据标准化处理
scales = MinMaxScaler(feature_range=(0, 1))
X_train_s = scales.fit_transform(X_train)
X_test_s = scales.transform(X_test)

```


```python
## 使用训练数据集对数据特征进行可视化
## 使用密度曲线对比不同类别在每个特征上的数据分布情况
colname = spam.columns.values[:-1]
plt.figure(figsize=(20,14))
for ii in range(len(colname)):
    plt.subplot(7,9,ii+1)
    sns.kdeplot(X_train_s[y_train == 0,ii], bw=0.05)
    sns.kdeplot(X_train_s[y_train == 1,ii], bw=0.05)
    plt.title(colname[ii])
plt.subplots_adjust(hspace=0.4)
plt.show()
```


```python
## 使用训练数据集对数据特征进行可视化
## 使用箱线对比不同类别在每个特征上的数据分布情况
colname = spam.columns.values[:-1]
plt.figure(figsize=(20,14))
for ii in range(len(colname)):
    plt.subplot(7,9,ii+1)
    sns.boxplot(x = y_train,y = X_train_s[:,ii])
    plt.title(colname[ii])
plt.subplots_adjust(hspace=0.4)
plt.show()
```


```python
## 使用训练数据集对数据特征进行可视化
## 使用箱线对比不同类别在每个特征上的数据分布情况
colname = spam.columns.values[:-1]
plt.figure(figsize=(30,20))
for ii in range(len(colname)):
    plt.subplot(7,9,ii+1)
    sns.boxplot(x = y_train,y = X_train_s[:,ii])
    plt.title(colname[ii])
plt.subplots_adjust(hspace=0.4)
plt.show()
```


```python
## 将数据转化为张量
X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(X_test_s.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))
## 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(X_train_t,y_train_t)
## 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=64, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 1, # 使用两个进程 
)
```


```python
len(train_loader)
```
### 搭建一个全连接神经网络

```python
## 全连接网络
class MLPclassifica(nn.Module):
    def __init__(self):
        super(MLPclassifica,self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features = 57, ## 第一个隐藏层的输入，数据的特征数
                out_features = 30,## 第一个隐藏层的输出，神经元的数量
                bias=True, ## 默认会有偏置
            ),
            nn.ReLU()
        )
        ## 定义第二个隐藏层
        self.hidden2 = nn.Sequential(
            nn.Linear(30,10),
            nn.ReLU()
        )
        ## 分类层
        self.classifica = nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid()
        )

    ## 定义网络的向前传播路径   
    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)
        ## 输出为两个隐藏层和输出层的输出
        return fc1,fc2,output
        
## 输出我们的网络结构
mlpc = MLPclassifica()
print(mlpc)
```
### 可视化网络的结果

```python
## 使用make_dot可视化网络
x = torch.randn(1,57).requires_grad_(True)
y = mlpc(x)
Mymlpcvis = make_dot(y, params=dict(list(mlpc.named_parameters()) + [('x', x)]))
Mymlpcvis
```


```python
## 将mlpvis保存为图片
# Mymlpcvis.format = "png" ## 形式转化为png,默认pdf
# ## 指定文件保存位置
# Mymlpcvis.directory = "data/chap5/Mymlpc_vis"
# Mymlpcvis.view() ## 会自动在当前文件夹生成文件
```
### 使用未预处理的数据训练模型

```python
## 将数据转化为张量
X_train_nots = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_nots = torch.from_numpy(X_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))
## 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data_nots = Data.TensorDataset(X_train_nots,y_train_t)
## 定义一个数据加载器，将训练数据集进行批量处理
train_nots_loader = Data.DataLoader(
    dataset = train_data_nots, ## 使用的数据集
    batch_size=64, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 1, # 使用1个进程 
)
```


```python
## 输出我们的网络结构
mlpc = MLPclassifica()
print(mlpc)
```


```python
# 定义优化器
optimizer = torch.optim.Adam(mlpc.parameters(),lr=0.01)  
loss_func = nn.CrossEntropyLoss()   # 二分类损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(15):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_nots_loader):  
        ## 计算每个batch的
        _,_,output = mlpc(b_x)               # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)   # 二分类交叉熵损失函数
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        train_loss.backward()           # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        niter = epoch*len(train_loader)+step+1
        
    ## 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            _,_,output = mlpc(X_test_nots)
            _,pre_lab = torch.max(output,1)
            test_accuracy = accuracy_score(y_test_t,pre_lab)
            # 为history添加epoch，损失和精度
            history1.log(niter, train_loss=train_loss, 
                         test_accuracy=test_accuracy)
            # 使用两个图像可视化损失函数和精度
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_accuracy"])

            
```
可以发现，损失函数很难迭代稳定







### 使用hiddenlayer包可视化网络训练过程

可视化网络训练过程，您需要使用两个类：history记录存储指标，而Canvas来绘制它们

```python
## 输出我们的网络结构
mlpc = MLPclassifica()
print(mlpc)
```
### 使用预处理后的数据的网络训练过程

```python
# 定义优化器
optimizer = torch.optim.Adam(mlpc.parameters(),lr=0.01)  
loss_func = nn.CrossEntropyLoss()   # 二分类损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(15):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):  
        ## 计算每个batch的
        _,_,output = mlpc(b_x)               # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)   # 二分类交叉熵损失函数
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        train_loss.backward()           # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        niter = epoch*len(train_loader)+step+1
        
    ## 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            _,_,output = mlpc(X_test_t)
            _,pre_lab = torch.max(output,1)
            test_accuracy = accuracy_score(y_test_t,pre_lab)
            # 为history添加epoch，损失和精度
            history1.log(niter, train_loss=train_loss, 
                         test_accuracy=test_accuracy)
            # 使用两个图像可视化损失函数和精度
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_accuracy"])

            
```


```python
## 计算最终模型在测试集上的精度
_,_,output = mlpc(X_test_t)
_,pre_lab = torch.max(output,1)
test_accuracy = accuracy_score(y_test_t,pre_lab)
print("test_accuracy:",test_accuracy)
print(classification_report(y_test_t,pre_lab))
print(confusion_matrix(y_test_t,pre_lab))

```
### 损失使用平均值的输出

```python
mlpc = MLPclassifica()
print(mlpc)
```


```python
# 定义优化器
optimizer = torch.optim.Adam(mlpc.parameters(),lr=0.01)  
loss_func = nn.CrossEntropyLoss()   # 二分类损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25
train_loss_all = 0
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(15):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):  
        ## 计算每个batch的
        _,_,output = mlpc(b_x)               # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)   # 二分类交叉熵损失函数
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        train_loss.backward()           # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        niter = epoch*len(train_loader)+step+1
        train_loss_all += train_loss
        
    ## 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            _,_,output = mlpc(X_test_t)
            _,pre_lab = torch.max(output,1)
            test_accuracy = accuracy_score(y_test_t,pre_lab)
            # 为history添加epoch，损失和精度
            history1.log(niter, train_loss=train_loss / niter, 
                         test_accuracy=test_accuracy)
            # 使用两个图像可视化损失函数和精度
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_accuracy"])

            
```


```python
## 计算最终模型在测试集上的精度
_,_,output = mlpc(X_test_t)
_,pre_lab = torch.max(output,1)
test_accuracy = accuracy_score(y_test_t,pre_lab)
print("test_accuracy:",test_accuracy)
print(classification_report(y_test_t,pre_lab))
print(confusion_matrix(y_test_t,pre_lab))

```


```python
## 计算最终模型在测试集上的精度
mlpc.eval()
_,_,output = mlpc(X_test_t)
pre_lab = torch.argmax(output,1)
test_accuracy = accuracy_score(y_test_t,pre_lab)
print("test_accuracy:",test_accuracy)
print(classification_report(y_test_t,pre_lab))
print(confusion_matrix(y_test_t,pre_lab))

```


```python
torch.argmax(output,1)

```
### 获取中间层的输出，并可视化

1:使用中间层的输出

2:使用钩子获取中间层的输出

```python
mlpc = MLPclassifica()
print(mlpc)
```


```python
## 计算最终模型在测试集上的第二个隐藏层的输出
_,test_fc2,_ = mlpc(X_test_t)
print("test_fc2.shape:",test_fc2.shape)
## 使用散点图进行可视化
## 对输出进行降维并可视化
test_fc2_tsne = TSNE(n_components=2).fit_transform(test_fc2.data.numpy())
```


```python
## 将特征进行可视化
plt.figure(figsize=(8,6))
# 可视化前设置坐标系的取值范围
plt.xlim([min(test_fc2_tsne[:,0]-1),max(test_fc2_tsne[:,0])+1])
plt.ylim([min(test_fc2_tsne[:,1]-1),max(test_fc2_tsne[:,1])+1])
plt.plot(test_fc2_tsne[y_test==0,0],test_fc2_tsne[y_test==0,1],
         "bo",label = "0")
plt.plot(test_fc2_tsne[y_test==1,0],test_fc2_tsne[y_test==1,1],
         "rd",label = "1")
plt.legend()
plt.title("test_fc2_tsne")
plt.show()
```


```python
## 使用钩子获取分类层的2个特征
## 定义一个辅助函数，来获取指定层名称的特征
activation = {} ## 保存不同层的输出
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
```


```python
## 全连接网络获取第一个全连接层的输出
mlpc.classifica.register_forward_hook(get_activation("classifica"))
_,_,_ = mlpc(X_test_t)
classifica = activation["classifica"].data.numpy()
print("classifica.shape:",classifica.shape)
```


```python
## 将特征进行可视化
plt.figure(figsize=(8,6))
# 可视化前设置坐标系的取值范围
plt.plot(classifica[y_test==0,0],classifica[y_test==0,1],
         "bo",label = "0")
plt.plot(classifica[y_test==1,0],classifica[y_test==1,1],
         "rd",label = "1")
plt.legend()
plt.title("classifica")
plt.show()
```

