# Pytorch深度神经网络及训练(有注释)

```python
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```
当在Jupyter Notebook中逐行执行上述代码时，会进行以下操作：

```python
%config InlineBackend.figure_format = 'retina'
```

这行代码设置了图形的显示格式为'retina'，其中`%config`表示配置命令，`InlineBackend`是IPython的一个后端模块，而`figure_format`参数用于指定图形格式。设置为'retina'表示使用高分辨率的Retina显示。

```python
%matplotlib inline
```

这行代码将Matplotlib绘图引擎的后端设置为'inline'，即内联模式。内联模式会直接在Notebook中显示图像，而不需要在外部窗口中打开。这样可以方便地将图像和代码显示在同一个Notebook单元格内。

通过上述两行代码的组合，我们完成了如下操作：首先，将图形的显示格式配置为Retina，以提高图像的清晰度和质量；然后，将Matplotlib绘图引擎的后端设置为内联模式，以便在Notebook中直接显示生成的图像。这样能够在Notebook中获得更高质量的图像展示效果，并且方便将图像与代码集成在一起进行工作和分享。

## 3.2 Pytorch中的优化器

介绍优化器的常用使用方法

```python
import torch
import torch.nn as nn
from torch.optim import Adam
```
解释如下：

```python
import torch
```

这行代码导入了PyTorch库。PyTorch是一个用于构建深度学习模型的开源机器学习框架。

```python
import torch.nn as nn
```

这行代码导入了PyTorch中的`nn`模块，其中包含了构建神经网络模型所需的各种类和函数。`nn`模块是PyTorch中用于构建神经网络层的基础模块。

```python
from torch.optim import Adam
```

这行代码从`torch.optim`模块中导入Adam优化器。优化器用于在训练神经网络时更新模型的权重，并进行梯度下降等优化操作。Adam是一种常用的优化算法，在更新参数时结合了动量方法和自适应学习率思想。

通过以上代码，我们导入了PyTorch库以及其中的`nn`模块和Adam优化器。这些工具将在搭建和训练神经网络模型时使用。

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
逐行解释上述代码如下：

```python
class TestNet(nn.Module):
```

定义了一个名为TestNet的类，该类继承自`nn.Module`，表示这是一个PyTorch神经网络模型的子类。

```python
    def __init__(self):
        super(TestNet,self).__init__()
```

在类的构造函数中，调用父类`nn.Module`的构造函数初始化网络模型。

```python
        self.hidden = nn.Sequential(
            nn.Linear(13,10),
            nn.ReLU(),
        )
```

定义了一个名为hidden的序列模块(`nn.Sequential`)，其中包含了两个层：一个线性层(`nn.Linear`)和一个ReLU激活函数层(`nn.ReLU`)。
- `nn.Linear(13, 10)` 定义了一个线性层，输入尺寸为13，输出尺寸为10。此线性层将输入特征映射到隐藏层的空间。
- `nn.ReLU()` 定义了一个ReLU激活函数层，用于引入非线性映射。ReLU函数通过保留正值并将负值清零来激活（激发）神经元。

```python
        self.regression = nn.Linear(10,1)
```

定义了一个线性层，作为预测回归层，其输入尺寸为10（前一隐藏层的输出特征数），输出尺寸为1。该层将隐藏层的特征映射到输出结果。

```python
    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output
```

定义了网络的向前传播路径。`forward`方法接受输入x作为参数，将它传递给隐藏层进行计算，然后将计算结果传递给预测回归层，最终返回输出output。

```python
testnet = TestNet()
```

创建一个TestNet类的实例，即创建了一个测试网络对象。

```python
print(testnet)
```

打印出我们定义的测试网络的结构。其中包含两个线性层和一个ReLU激活层，以及一个线性回归层，组成了一个简单的神经网络结构。


```python
## 使用方式1
optimizer = Adam(testnet.parameters(),lr=0.001)  
```
逐行解释上述代码如下：

```python
optimizer = Adam(testnet.parameters(),lr=0.001)
```

创建了一个Adam优化器对象(optimizer)。Adam是一种常用的优化算法，用于调整神经网络模型的参数。

- `Adam`：使用torch.optim中的Adam类来创建优化器对象。
- `testnet.parameters()`：通过调用`testnet`对象的`.parameters()`方法获取模型中的所有可学习参数。这些参数在神经网络的层中定义，并且需要在训练过程中更新。
- `lr=0.001`：设置学习率(learning rate)为0.001。学习率决定了每次参数更新的步长大小。在训练过程中，优化器根据损失函数的梯度和学习率来更新参数值。

通过以上代码，我们创建了一个Adam优化器对象，并将它与我们之前定义的测试网络模型(`testnet`)的参数相关联。这样，在训练过程中我们可以使用这个优化器对象来自动计算并更新模型的参数，以便尽量减小模型的损失函数。

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
逐行解释上述代码如下：

```python
optimizer = Adam(
    [{"params":testnet.hidden.parameters(),"lr":0.0001},
    {"params":testnet.regression.parameters(),"lr": 0.01}],
    lr=1e-2)
```

创建了一个Adam优化器对象(optimizer)。这次我们为不同的网络层定义了不同的学习率。

- `Adam`：使用torch.optim中的Adam类来创建优化器对象。
- `[{"params":testnet.hidden.parameters(),"lr":0.0001}, {"params":testnet.regression.parameters(),"lr": 0.01}]`：传递了一个包含两个字典的列表，每个字典对应一个不同的网络层。
  - `{"params": testnet.hidden.parameters(), "lr": 0.0001}`：指定了hidden层的参数(`testnet.hidden.parameters()`)以及其对应的学习率为0.0001。
  - `{"params": testnet.regression.parameters(), "lr": 0.01}`：指定了regression层的参数(`testnet.regression.parameters()`)以及其对应的学习率为0.01。
- `lr=1e-2`：设置了整体的学习率为1e-2。这个学习率将应用于其他没有特别指定学习率的所有参数。

通过以上代码，我们创建了一个Adam优化器对象，并为测试网络的不同层设置了不同的学习率。这样在训练过程中，优化器将根据各自的学习率来更新相应层的参数。除了特定指定的学习率外，其他参数都使用整体学习率进行更新。这种方式允许我们对不同层的参数设置不同的学习率，以更好地控制模型的训练过程。

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
逐行解释上述代码如下：

```python
for input, target in dataset:
    optimizer.zero_grad()      
    output = testnetst(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

这段代码展示了一个优化目标函数的典型训练循环。它包含以下步骤：

- `for input, target in dataset:`：遍历数据集，每次迭代都获取输入(input)和对应的目标(target)。
- `optimizer.zero_grad()`：梯度清零。在每个样本的训练之前，需要将之前计算得到的梯度清零，以避免梯度叠加。
- `output = testnetst(input)`：通过输入(input)将数据传递给测试网络(testnetst)，计算得到预测值(output)。
- `loss = loss_fn(output, target)`：使用损失函数(loss_fn)计算预测值(output)与目标(target)之间的损失。
- `loss.backward()`：反向传播。自动计算损失函数关于模型参数的导数（梯度），从而实现误差的后向传播。
- `optimizer.step()`：更新网络参数。根据计算得到的梯度进行参数更新，使得模型参数朝着减少损失的方向移动。

这段代码描述了一次完整的训练迭代过程，其中模型参数通过优化器(optimizer)来进行更新。通过反复迭代该过程，我们可以逐渐优化目标函数，并使模型更好地拟合训练数据。请注意，这是一个示例代码块，并不能直接运行，实际运行时还需要对其中的变量和函数进行定义和配置。

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
逐行解释上述代码如下：

```python
conv1 = torch.nn.Conv2d(3,16,3)
```

这行代码创建了一个卷积层(conv1)对象。`torch.nn.Conv2d`是PyTorch中用于定义二维卷积的类。它接受三个参数：输入通道数(3)，输出通道数(16)，卷积核尺寸(3)。该卷积层可用于图像数据的特征提取。

```python
torch.manual_seed(12)
```

这行代码设置了随机数生成的种子(seed)为12。该种子用于生成随机数，通过设置相同的种子可以得到重现性结果，即每次运行生成的随机数序列将一致。

```python
torch.nn.init.normal(conv1.weight, mean=0, std=1)
```

这行代码使用标准正态分布初始化权重。`torch.nn.init`模块提供了各种权重初始化方法。对于卷积层对象`conv1.weight`，我们调用`torch.nn.init.normal()`方法，以标准正态分布(N(0, 1))来初始化权重。其中：

- `conv1.weight`：表示需要初始化的权重。`conv1.weight`是卷积层(conv1)的权重属性。
- `mean=0`：指定正态分布的均值(mean)为0，即N(0, 1)。
- `std=1`：指定正态分布的标准差(std)为1，即N(0, 1)。

通过以上代码，我们创建了一个二维卷积层(conv1)，并使用标准正态分布初始化了它的权重。这样可以使权重具有随机性，并有助于模型在训练过程中学习到适合任务需求的特征表示。

```python
## 使用直方图可视化conv1.weight的分布情况
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.hist(conv1.weight.data.numpy().reshape((-1,1)),bins = 30)
plt.show()
```
逐行解释上述代码如下：

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
```

这两行代码导入了matplotlib库，并创建了一个大小为8x6的画布(figure)。

```python
plt.hist(conv1.weight.data.numpy().reshape((-1,1)), bins=30)
```

这行代码使用`plt.hist()`函数绘制直方图。`conv1.weight.data.numpy().reshape((-1,1))`将卷积层权重转换为NumPy数组，并将其形状调整为(-1, 1)形式，以便适配于直方图的输入格式。`bins=30`指定了直方图所使用的箱(bin)数量。

```python
plt.show()
```

这行代码显示了绘制的直方图。

通过以上代码，我们导入了matplotlib库，并使用该库绘制了卷积层权重(conv1.weight)的分布直方图。直方图可以提供权重值的分布情况，有助于我们了解权重初始化是否合理以及模型在训练过程中是否发生了梯度消失、梯度弥散等问题。

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
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        
        # 建立卷积层conv1，输入通道为3，输出通道为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(3, 16, 3)
        
        # 建立一个包含多个线性层和ReLU激活函数的隐藏层部分
        self.hidden = nn.Sequential(
            nn.Linear(100, 100),  # 输入大小为100，输出大小为100的线性层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(100, 50),  # 输入大小为100，输出大小为50的线性层
            nn.ReLU()  # ReLU激活函数
        )
        
        # 定义最后一层全连接层，将隐藏层的输出映射到10个类别的输出
        self.cla = nn.Linear(50, 10)
    
    def forward(self, x):
        # 网络的向前传播路径
        
        # 对输入图像进行卷积操作
        x = self.conv1(x)
        
        # 将卷积层的输出展平成一个一维向量
        x = x.view(x.shape[0], -1)
        
        # 经过隐藏层部分的线性层和激活函数处理
        x = self.hidden(x)
        
        # 最后通过全连接层进行分类预测
        output = self.cla(x)
        
        return output


# 创建一个TestNet的实例
testnet = TestNet()

# 打印网络结构
print(testnet)
```

逐行注释解释：

- `class TestNet(nn.Module):`：定义了一个名为TestNet的类，继承自`nn.Module`。
- `def __init__(self):`：该方法是TestNet类的初始化方法，在创建TestNet类的实例时被调用。
- `super(TestNet, self).__init__():`：调用父类`nn.Module`的初始化方法，以确保正确地初始化TestNet类。
- `self.conv1 = nn.Conv2d(3, 16, 3):`：建立一个名为conv1的卷积层。输入通道数为3（RGB图像），输出通道数为16，卷积核大小为3x3。
- `self.hidden = nn.Sequential(...)`：建立一个包含多个线性层和ReLU激活函数的隐藏层部分。使用`nn.Sequential()`容器按顺序组合这些层。
- `self.cla = nn.Linear(50, 10):`：建立最后一层全连接层，将隐藏层的输出映射到10个输出类别。
- `def forward(self, x):`：定义了网络的向前传播路径。在PyTorch中，所有自定义的模型都需要实现forward方法。
- `x = self.conv1(x):`：对输入x进行卷积操作，通过conv1卷积层的处理。
- `x = x.view(x.shape[0], -1):`：将卷积层的输出x展平成一个一维向量，保持批量大小不变。
- `x = self.hidden(x)`：通过hidden部分的线性层和ReLU激活函数处理x。
- `output = self.cla(x)`：通过全连接层cla对处理后的x进行分类预测，得到最终输出。
- `testnet = TestNet()`: 创建一个TestNet类的实例，即创建一个测试网络对象。
- `print(testnet)`: 打印测试网络的结构，包括卷积层、隐藏层和全连接层。

```python
## 定义为网络中的每个层进行权重初始化的函数
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

```python
def init_weights(m):
    ## 如果是卷积层
    if type(m) == nn.Conv2d:
        # 对卷积层的权重进行初始化，从均值为0、标准差为0.5的正态分布中随机采样赋值给权重
        torch.nn.init.normal_(m.weight, mean=0, std=0.5)
    
    ## 如果是全连接层
    if type(m) == nn.Linear:
        # 对全连接层的权重进行初始化，使用在区间[-0.1, 0.1]上均匀分布的随机值赋值给权重
        torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        
        # 对全连接层的偏置进行初始化，将偏置值设为0.01
        m.bias.data.fill_(0.01)


## 使用网络的apply方法进行权重初始化
torch.manual_seed(13)  ## 随机数初始化种子
testnet.apply(init_weights)

# 输出全连接层cla的权重数据
testnet.cla.weight.data
```
逐行注释解释：

- `def init_weights(m):`：定义了一个自定义函数init_weights，用于对网络中的每个层进行权重初始化。该函数会作为参数传递给`apply()`方法。
- `if type(m) == nn.Conv2d:`：如果当前层是卷积层，则执行下面的代码块。
- `torch.nn.init.normal_(m.weight, mean=0, std=0.5)`：使用正态分布从均值为0、标准差为0.5的分布中随机初始化卷积层m的权重。
- `if type(m) == nn.Linear:`：如果当前层是全连接层，则执行下面的代码块。
- `torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)`：使用在区间[-0.1, 0.1]上均匀分布的随机值初始化全连接层m的权重。
- `m.bias.data.fill_(0.01)`：将全连接层m的偏置初始化为常数值0.01。
- `testnet.apply(init_weights)`：对测试网络testnet中的每个层应用init_weights函数进行权重初始化。
- `torch.manual_seed(13)`：设置随机数生成器的种子，以确保每次初始化时得到相同的随机结果。
- `testnet.cla.weight.data`：输出全连接层cla的权重数据。


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

逐行注释解释：

- `import torch`：导入PyTorch库，用于构建和训练深度学习模型。
- `import torch.nn as nn`：导入torch.nn模块，包含了定义神经网络层的相关类和函数。
- `from torch.optim import SGD`：从torch.optim模块中导入随机梯度下降优化器SGD。
- `import torch.utils.data as Data`：导入torch.utils.data模块，该模块提供了用于数据加载和处理的工具。
- `from sklearn.datasets import load_boston`：从sklearn.datasets模块中导入load_boston，用于加载波士顿房价数据集。
- `from sklearn.preprocessing import StandardScaler`：从sklearn.preprocessing模块中导入StandardScaler，用于数据的标准化处理。
- `import pandas as pd`：导入pandas库，用于数据处理和分析。
- `import numpy as np`：导入numpy库，用于数值计算和数组操作。
- `import matplotlib.pyplot as plt`：导入matplotlib.pyplot库并命名为plt，用于数据可视化。

```python
## 读取数据
boston_X,boston_y = load_boston(return_X_y=True)
print("boston_X.shape:",boston_X.shape)
```

结果:
```
boston_X.shape: (506, 13)
```

逐行注释解释：
```python
boston_X, boston_y = load_boston(return_X_y=True)
```
- `boston_X, boston_y = load_boston(return_X_y=True)`: 使用`load_boston()`函数加载波士顿房价数据集，并将返回的数据特征矩阵赋值给`boston_X`，目标变量赋值给`boston_y`。`return_X_y=True`表示返回特征矩阵和目标变量。

```python
print("boston_X.shape:", boston_X.shape)
```
- `print("boston_X.shape:", boston_X.shape)`: 打印输出字符串`"boston_X.shape:"`和`boston_X`的形状，即数据特征矩阵`boston_X`的维度信息。使用`.shape`属性可以得到矩阵的形状。在此例中，输出的结果是`(506, 13)`，表示数据特征矩阵`boston_X`有506个样本和13个特征。


```python
plt.figure()
plt.hist(boston_y,bins=20)
plt.show()
```

逐行注释解释：
```python
plt.figure()
```
- `plt.figure()`: 创建一个新的图形窗口或画布。

```python
plt.hist(boston_y, bins=20)
```
- `plt.hist(boston_y, bins=20)`: 绘制直方图。`boston_y`是待绘制的数据，`bins=20`表示将数据分成20个区间进行统计。

```python
plt.show()
```
- `plt.show()`: 显示绘制的图形。将之前创建的图形窗口或画布中的内容显示出来。

上述代码首先创建一个图形窗口或画布，然后使用`plt.hist()`函数绘制了目标变量`boston_y`的直方图，并指定了将数据分成20个区间进行统计。最后通过`plt.show()`方法显示绘制的直方图。直方图展示了目标变量`boston_y`的分布情况，及其在不同区间的频率或样本数量。

```python
## 数据标准化处理
ss = StandardScaler(with_mean=True,with_std=True)
boston_Xs = ss.fit_transform(boston_X)
# boston_ys = ss.fit_transform(boston_y)
np.mean(boston_Xs,axis=0)
np.std(boston_Xs,axis=0)
```
结果:
```python
array([-8.78743718e-17,  1.37029363e-16, -9.89668720e-16, -5.19566823e-16,
        6.34668220e-15, -2.18569614e-14,  2.10635299e-15, -3.51004018e-16,
       -8.54560355e-18,  8.16310129e-16, -9.93416097e-15,  8.88178420e-17,
        1.65545459e-14])
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

计算过程解释：

逐行注释解释：
```python
ss = StandardScaler(with_mean=True,with_std=True)
```
- `ss = StandardScaler(with_mean=True, with_std=True)`: 创建一个`StandardScaler`对象，用于对数据进行标准化处理。`with_mean=True`表示将数据按列均值中心化，`with_std=True`表示将数据按列标准差缩放。

```python
boston_Xs = ss.fit_transform(boston_X)
```
- `boston_Xs = ss.fit_transform(boston_X)`: 使用`fit_transform()`方法对`boston_X`进行标准化处理，得到标准化后的特征矩阵`boston_Xs`。`fit_transform()`方法会使用`StandardScaler`对象的参数（如列均值、列标准差）对数据进行标准化，并返回标准化后的结果。

```python
np.mean(boston_Xs, axis=0)
```
- `np.mean(boston_Xs, axis=0)`: 计算标准化后的特征矩阵`boston_Xs`在每列上的均值，`axis=0`表示沿着列的方向进行计算。返回的结果是一个一维数组，包含每列均值的值。

```python
np.std(boston_Xs, axis=0)
```
- `np.std(boston_Xs, axis=0)`: 计算标准化后的特征矩阵`boston_Xs`在每列上的标准差，`axis=0`表示沿着列的方向进行计算。返回的结果是一个一维数组，包含每列标准差的值。

解释代码：

首先，创建一个StandardScaler对象`ss`，然后调用`fit_transform()`方法将`boston_X`传递给`ss`进行标准化处理。标准化处理会根据列均值和列标准差对数据进行中心化和缩放操作，返回标准化后的特征矩阵`boston_Xs`。

接下来，使用`np.mean()`和`np.std()`函数分别计算了`boston_Xs`在每列上的均值和标准差。方法`np.mean()`用于计算平均值，参数`axis=0`表示沿着列的方向计算，返回结果为一个一维数组，包含了每列的均值。方法`np.std()`用于计算标准差，参数`axis=0`表示沿着列的方向计算，返回结果为一个一维数组，包含了每列的标准差。

结果显示均值数组接近零，标准差数组接近1，说明数据已按列进行中心化和缩放处理。

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

首先，给出结果的代码部分：
```python
print(b_x.shape)
print(b_y.shape)
```

结果：
```python
torch.Size([128, 13])
torch.Size([128])
```

逐行注释解释:
```python
train_xt = torch.from_numpy(boston_Xs.astype(np.float32))
```
- 将标准化后的特征矩阵`boston_Xs`转换为张量并将其存储在`train_xt`中。使用`torch.from_numpy()`将NumPy数组转换为PyTorch张量，并且在转换过程中指定数据类型为`np.float32`。

```python
train_yt = torch.from_numpy(boston_y.astype(np.float32))
```
- 将目标变量`boston_y`转换为张量并将其存储在`train_yt`中。同样，使用`torch.from_numpy()`将NumPy数组转换为PyTorch张量，并且将数据类型指定为`np.float32`。

```python
train_data = Data.TensorDataset(train_xt, train_yt)
```
- 使用`Data.TensorDataset`将训练集特征矩阵`train_xt`和目标变量`train_yt`整合到一起，形成一个用于批量训练的数据集对象`train_data`。

```python
train_loader = Data.DataLoader(
    dataset=train_data, 
    batch_size=128,
    shuffle=True,
    num_workers=1
)
```
- 定义一个数据加载器`train_loader`，用于按批次加载训练数据集。
- `dataset=train_data`：指定要使用的训练数据集。
- `batch_size=128`：每个批次加载的样本数量为128。
- `shuffle=True`：在每次迭代之前随机打乱数据集。
- `num_workers=1`：使用一个进程来加载数据。

注释：
```python
print(b_x.shape)
print(b_y.shape)
```
- 打印输出一个批次样本的维度信息。在这个例子中，输出结果的`b_x`表示特征矩阵的形状，大小为`(128, 13)`，表示每个批次包含128个样本，每个样本有13个特征。
- `b_y`表示目标变量的形状，大小为`(128,)`，表示每个批次包含128个目标变量值。


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

