# 基于Pytorch的相关可视化工具(有注释)

## 网络结构的可视化

```python
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```
这段代码是Jupyter Notebook中的魔术命令，用于配置图形输出的格式和将图形直接显示在Notebook中。让我们逐行解释：

```python
%config InlineBackend.figure_format = 'retina'
```

该命令设置图形输出的分辨率为Retina级别。在高DPI（像素密度）的屏幕上，使用'Retina'选项可以提供更清晰的图像。

```python
%matplotlib inline
```

该命令启用内联绘图，并且在Notebook中自动显示绘制的图形。这使得我们可以在Notebook中直接看到图形结果，而无需调用`plt.show()`来显示图形。

这两行代码的目的是为了优化图形的显示效果以及直接在Notebook中展示绘制的图形。

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.optim import SGD
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```
这段代码是一个典型的PyTorch的导入代码块，它导入所需的库和模块。让我们逐行解释：

```python
import torch
```
`torch`是PyTorch库的主要入口。它提供了张量操作、神经网络构建、自动求导等功能。

```python
import torch.nn as nn
```
`torch.nn`模块提供了构建神经网络所需的各种工具和类。

```python
import torchvision
```
`torchvision`库提供了常用的计算机视觉任务相关的数据集、模型架构和图像处理函数等功能。

```python
import torchvision.utils as vutils
```
`torchvision.utils`中的`vutils`模块提供了在PyTorch中方便地可视化和处理图像的实用函数。

```python
from torch.optim import SGD
```
`torch.optim`模块提供了优化算法的实现，`SGD`是其中的一种梯度下降优化器。

```python
import torch.utils.data as Data
```
`torch.utils.data`模块提供了用于处理和加载数据的工具类，比如`DataLoader`。

```python
from sklearn.metrics import accuracy_score
```
`sklearn.metrics`模块提供了各种评估指标的计算，`accuracy_score`是其中用于计算准确率的函数。

```python
import matplotlib.pyplot as plt
```
`matplotlib.pyplot`模块是Matplotlib库的子模块，用于绘制图形和可视化数据。这里我们将其导入为`plt`以方便使用。

该代码段的目的是导入PyTorch及相关的库和模块以供后续使用，包括构建神经网络、数据操作、优化器、评估指标和图形绘制等。

```python
## 使用手写字体数据
## 准备训练数据集
train_data  = torchvision.datasets.MNIST(
    root = "./data/MNIST", # 数据的路径
    train = True, # 只使用训练数据集
    # 将数据转化为torch使用的张量,取值范围为［0，1］
    transform  = torchvision.transforms.ToTensor(),
    download= False # 因为数据已经下载过，所以这里不再下载
)
## 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset = train_data, ## 使用的数据集
    batch_size=128, # 批处理样本大小
    shuffle = True, # 每次迭代前打乱数据
    num_workers = 2, # 使用两个进程 
)

##  获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):  
    if step > 0:
        break

## 输出训练图像的尺寸和标签的尺寸
print(b_x.shape)
print(b_y.shape)


## 准备需要使用的测试数据集
test_data  = torchvision.datasets.MNIST(
    root = "./data/MNIST", # 数据的路径
    train = False, # 不使用训练数据集
    download= False # 因为数据已经下载过，所以这里不再下载
)
## 为数据添加一个通道纬度,并且取值范围缩放到0～1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x,dim = 1)
test_data_y = test_data.targets  ## 测试集的标签

print("test_data_x.shape:",test_data_x.shape)
print("test_data_y.shape:",test_data_y.shape)

```
这段代码是用于准备手写数字数据集（MNIST）并加载训练和测试数据的过程。让我们逐行解释：
```python
train_data = torchvision.datasets.MNIST(
    root="./data/MNIST",  # 数据的路径
    train=True,  # 只使用训练数据集
    transform=torchvision.transforms.ToTensor(),  # 将数据转化为torch使用的张量，取值范围为[0,1]
    download=False  # 因为数据已经下载过，所以这里不再下载
)
```

这行代码创建了一个名为`train_data`的`MNIST`数据集实例。参数说明如下：
- `root`是数据集的存储路径。
- `train=True`表示使用训练数据集。
- `transform=torchvision.transforms.ToTensor()`将数据转换为PyTorch张量，并将像素值范围从[0, 255]缩放到[0, 1]之间。
- `download=False`表示不从互联网上下载数据。


```python
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的数据集
    batch_size=128,  # 批处理样本大小
    shuffle=True,  # 每次迭代前打乱数据
    num_workers=2  # 使用两个进程
)
```

定义一个数据加载器`train_loader`，用于按照指定配置加载训练数据集。
- `dataset=train_data`表示要加载的数据集。
- `batch_size=128`每个批次中的样本数量。
- `shuffle=True`表示每次迭代前对数据进行随机打乱。
- `num_workers=2`表示使用两个进程并行加载数据。

```python
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
```

迭代遍历训练数据集的一个批次。在这个例子中，我们仅迭代一次，并提取出第一个批次的数据。`b_x`是输入图像张量，`b_y`是对应的标签。

```python
print(b_x.shape)
print(b_y.shape)
```

输出训练图像张量的形状和标签张量的形状。

```python
test_data = torchvision.datasets.MNIST(
    root="./data/MNIST",  # 数据的路径
    train=False,  # 不使用训练数据集
    download=False  # 因为数据已经下载过，所以这里不再下载
)
```

这行代码创建了一个名为`test_data`的`MNIST`数据集实例，用于准备测试数据集。参数说明与训练数据集相同，只是将`train`设置为`False`表示不使用训练数据。

```python
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0  # 为数据添加一个通道维度，并将取值范围缩放到0~1之间
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets  # 测试集的标签

```

这部分代码准备了测试数据集。首先，通过`test_data.data`获取测试图像数据，并将其转换为`FloatTensor`类型。然后，将像素值范围从[0, 255]缩放到[0, 1]之间。最后，通过`torch.unsqueeze()`函数在通道维度上添加一个额外的维度（因为图片数据是单通道灰度图像）。`test_data_y`保存了对应的测试标签。

```python
print("test_data_x.shape:", test_data_x.shape)
print("test_data_y.shape:", test_data_y.shape)
```

这两行代码用于打印测试数据集的形状信息，即打印测试数据集的图像张量形状和标签张量形状。









```python
## 搭建一个卷积神经网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        ## 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,## 输入的feature map
                out_channels = 16, ## 输出的feature map
                kernel_size = 3, ##卷积核尺寸
                stride=1,   ##卷积核步长
                padding=1, # 进行填充
            ), 
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(
                kernel_size = 2,## 平均值池化层,使用 2*2
                stride=2,   ## 池化步长为2 
            ), 
        )
        ## 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1), 
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2,2) ## 最大值池化
        )
        ## 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(
                in_features = 32*7*7, ## 输入特征
                out_features = 128, ## 输出特证数
            ),
            nn.ReLU(),  # 激活函数
            nn.Linear(128,64),
            nn.ReLU()  # 激活函数
        )
        self.out = nn.Linear(64,10) ## 最后的分类层


    ## 定义网络的向前传播路径   
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        x = self.fc(x)
        output = self.out(x)
        return output
    
## 输出我们的网络结构
MyConvnet = ConvNet()
print(MyConvnet)
```

这段代码搭建了一个卷积神经网络（ConvNet）用于图像分类任务。下面我会详细解释每一部分的含义和功能：

**整体结构**
```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
```
在这段代码中，定义了一个名为`ConvNet`的类，并继承自`nn.Module`。

**第一个卷积层**
```python
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,  # 输入的特征图通道数
                out_channels = 16,  # 输出的特征图通道数
                kernel_size = 3,  # 卷积核尺寸
                stride=1,  # 卷积核步长
                padding=1,  # 进行填充
            ), 
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(
                kernel_size = 2,  # 平均值池化层大小,使用 2x2
                stride=2,  # 池化步长为2 
            ), 
        )
```
第一个卷积层(`conv1`)由一个卷积操作(`nn.Conv2d`)、ReLU激活函数(`nn.ReLU`)以及平均值池化层(`nn.AvgPool2d`)组成。输入特征图通道数是1，输出特征图通道数是16，卷积核尺寸为3x3，卷积步长为1，填充为1。平均值池化层使用2x2的窗口和步长。

**第二个卷积层**
```python
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1), 
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2,2)  # 最大值池化层
        )
```
第二个卷积层(`conv2`)由一个卷积操作 (`nn.Conv2d`)、ReLU激活函数 (`nn.ReLU`) 和最大值池化层 (`nn.MaxPool2d`) 组成。该层输入特征图通道数是16，输出特征图通道数是32，卷积核尺寸是3x3，卷积步长为1，填充为1。最大值池化层使用2x2的窗口和步长。

**全连接层**
```python
        self.fc = nn.Sequential(
            nn.Linear(
                in_features = 32*7*7,  # 输入特征数量
                out_features = 128,  # 输出特征数量
            ),
            nn.ReLU(),  # 激活函数
            nn.Linear(128,64),
            nn.ReLU()  # 激活函数
        )
        self.out = nn.Linear(64,10)  # 最后的分类层
```
定义了全连接层 (`fc`)，它由两个线性变换 (`nn.Linear`) 和 ReLU激活函数 (`nn.ReLU`) 组成。输入特征数量是32x7x7 (通过计算可以得知)，输出特征数量是128。之后是一个线性变换和ReLU激活函数，输入为128，输出为64。

最后，定义了一个分类层 (`out`)，接收来自全连接层的输入（64），并映射到10个类别。

**前向传播路径**
```python
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        x = self.fc(x)
        output = self.out(x)
        return output
```
`forward()` 方法定义了网络的前向传播路径。输入 (`x`) 首先通过 `conv1` 进行卷积、激活和池化操作，然后再经过 `conv2` 进行相同的操作。之后，使用 `view()` 函数将多维的卷积图层展平成一维张量。接下来，传递给全连接层 (`fc`)，然后通过分类层 (`out`) 得到最终输出。

**实例化和打印网络结构**
```python
MyConvnet = ConvNet()
print(MyConvnet)
```
**输出网络结构**
```
ConvNet(
  (conv1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (conv2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=1568, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
  )
  (out): Linear(in_features=64, out_features=10, bias=True)
)

```

在这里，我们实例化了 `ConvNet` 类，并打印出网络结构信息。

总结来说，代码搭建了一个简单的卷积神经网络，包括两个卷积层、两个全连接层和一个分类层。网络接收图像数据作为输入，并输出对应的十个类别的预测结果。每个卷积层通过卷积操作提取特征，并使用激活函数进行非线性映射。池化层则用于进一步减少特征图的尺寸。全连接层将展平的特征连接到神经元，并通过线性变换和激活函数处理。最后的分类层将从全连接层获得的特征向量映射到预测的类别上。













###  使用hiddenlayer包可视化网络

```python
import hiddenlayer as hl
## 可视化卷积神经网络
hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 28, 28]))
hl_graph.theme = hl.graph.THEMES["blue"].copy()  
hl_graph
```
这段代码使用了HiddenLayer库来实现卷积神经网络的可视化。下面是对每一部分代码的详细解释：

```python
import hiddenlayer as hl
```
首先，我们导入HiddenLayer库，它是一个用于可视化深度学习模型结构的工具。

```python
hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 28, 28]))
```
接下来，在这一行代码中，我们使用`build_graph`函数创建了一个图形对象`hl_graph`。此函数的第一个参数是我们要可视化的卷积神经网络对象`MyConvnet`，第二个参数是一个输入示例`torch.zeros([1, 1, 28, 28])`，表示输入图像的大小为1x28x28。该函数会根据给定的网络和输入生成一张图，用于表示网络的层次结构和连接方式。

```python
hl_graph.theme = hl.graph.THEMES["blue"].copy()
```
在这一行代码中，我们将图形对象的主题设置为蓝色（blue）。HiddenLayer库提供了一些预定义的主题，可以根据个人喜好进行选择。我们通过`THEMES["blue"]`来选择蓝色主题，并使用`copy()`函数创建了一个主题副本，以便进一步的自定义。
两个问题：
- 如何使用其他预定义的主题？
- 如何进行进一步的自定义？

```python
hl_graph
```
最后，打印或显示`hl_graph`对象，以便在Jupyter Notebook中查看可视化的卷积神经网络结构。这将显示一个图形，该图形代表了我们搭建的卷积神经网络的层次结构，以及不同层之间的连接方式。

通过这段代码，我们可以更好地理解卷积神经网络的结构，并通过可视化来展示网络中的信息流动。这对于理解和调试复杂的深度学习模型非常有用，也为教学和演示提供了一个可视化工具。









```python
## 将可视化的网路保存为图片,默认格式为pdf
hl_graph.save("data/chap4/MyConvnet_hl.png", format="png")
```
这段代码用于将可视化的网络结构保存为一张图片。下面是对每一部分代码的详细解释：

```python
hl_graph.save("data/chap4/MyConvnet_hl.png", format="png")
```

首先，我们调用`save`函数来保存可视化的网络结构。这个函数接受两个参数: 

- 第一个参数 `"data/chap4/MyConvnet_hl.png"` 是保存的路径和文件名，表示要将图像保存在 `data/chap4` 目录下，并将图像命名为 `MyConvnet_hl.png`。

- 第二个参数 `format="png"` 表示保存图像的格式为PNG。我们指定了PNG格式，但你也可以选择其他格式，例如JPEG、PDF等。

此代码通过调用`save`函数将可视化的网络结构保存为一张图片。

通过这段代码，我们可以将可视化的卷积神经网络结构保存为图片，以供进一步使用或分享给他人。保存为图片的网络结构可以更方便地被读取、显示和共享。




###  使用torchviz包可视化网络

```python
from torchviz import make_dot
## 使用make_dot可视化网络
x = torch.randn(1, 1, 28, 28).requires_grad_(True)
y = MyConvnet(x)
MyConvnetvis = make_dot(y, params=dict(list(MyConvnet.named_parameters()) + [('x', x)]))
MyConvnetvis
```
这段代码使用了`torchviz`库来可视化卷积神经网络，并对每一部分进行详细解释。下面是对代码的解释：

```python
from torchviz import make_dot
```
首先，我们导入了`torchviz`库，它是PyTorch中的一个工具，用于可视化计算图。

```python
x = torch.randn(1, 1, 28, 28).requires_grad_(True)
```
接下来，我们创建了一个张量`x`，它是一个大小为1x1x28x28的随机张量，并设置`requires_grad`为`True`，以便追踪梯度信息。

```python
y = MyConvnet(x)
```
然后，我们将输入张量`x`传递给`MyConvnet`模型，并将输出赋值给变量`y`。这样做是为了获取网络的输出结果，以便在计算图中使用。

```python
MyConvnetvis = make_dot(y, params=dict(list(MyConvnet.named_parameters()) + [('x', x)]))
```
在这一行代码中，我们使用`make_dot`函数创建了一个图形对象`MyConvnetvis`，该对象代表了使用`y`作为输出和`MyConvnet`中的参数的计算图。我们通过传递两个参数给`make_dot`函数来生成计算图：

- 第一个参数`y`是网络的输出结果，我们希望可视化这个节点。
- 第二个参数`params`是一个字典，包含了模型参数和输入张量的名称。我们使用`list(MyConvnet.named_parameters())`获取了模型中的所有参数，并将其与输入张量`x`的名称`'x'`合并成一个字典，以便在计算图中显示参数和输入节点。

```python
MyConvnetvis
```
最后，我们输出`MyConvnetvis`对象，以在Jupyter Notebook中查看可视化的计算图。

通过这段代码，我们可以使用`torchviz`库生成并显示卷积神经网络的计算图。计算图可以帮助我们理解网络的数据流和梯度传播，以及识别潜在的错误或优化点。这对于深入分析和调试复杂的神经网络非常有用，并且可以提供教学和演示目的。






```python
## 将mlpvis保存为图片
MyConvnetvis.format = "png" ## 形式转化为png,默认pdf
## 指定文件保存位置
MyConvnetvis.directory = "data/chap4/MyConvnet_vis"
MyConvnetvis.view() ## 会自动在当前文件夹生成文件
```
## 训练过程的可视化
### 利用tensorboardX

```python
## 从tensorboardX库中导入需要的API
from tensorboardX import SummaryWriter
SumWriter = SummaryWriter(log_dir="data/chap4/log")
```


```python
# 定义优化器
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)  
loss_func = nn.CrossEntropyLoss()   # 损失函数
train_loss = 0
print_step = 100 ## 每经过100次迭代后,输出损失
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(5):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):  
        ## 计算每个batch的
        output = MyConvnet(b_x)            # CNN在训练batch上的输出
        loss = loss_func(output, b_y)   # 交叉熵损失函数
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        loss.backward()                 # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        train_loss = train_loss+loss    # 计算损失的累加损失
        ## 计算迭代次数
        niter = epoch * len(train_loader) + step+1
        ## 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            ## 为日志添加训练集损失函数
            SumWriter.add_scalar("train loss",
                                 train_loss.item() / niter,
                                 global_step=niter)
            ## 计算在测试集上的精度
            output = MyConvnet(test_data_x)
            _,pre_lab = torch.max(output,1)
            acc = accuracy_score(test_data_y,pre_lab)
            ## 为日志添加在测试集上的预测精度
            SumWriter.add_scalar("test acc",acc.item(),niter)
            ## 为日志中添加训练数据的可视化图像，使用当前batch的图像
            ## 将一个batch的数据进行预处理
            b_x_im = vutils.make_grid(b_x,nrow=12)
            SumWriter.add_image('train image sample', b_x_im,niter)
            ## 使用直方图可视化网络中参数的分布情况
            for name, param in MyConvnet.named_parameters():
                SumWriter.add_histogram(name, param.data.numpy(),niter)
       
```


```python
# ## 为日志中添加训练数据的可视化图像，使用最后一个batch的图像
# ## 将一个batch的数据进行预处理
# b_x_im = vutils.make_grid(b_x,nrow=12)
# SumWriter.add_image('train image sample', b_x_im)
```


```python
# ## 使用直方图可视化网络中参数的分布情况
# for name, param in MyConvnet.named_parameters():
#     SumWriter.add_histogram(name, param.data.numpy())
```
### 利用hiddenlayer


```python
import hiddenlayer as hl
import time
```


```python
## 初始化MyConvnet
MyConvnet = ConvNet()
print(MyConvnet)
```


```python
# 定义优化器
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)  
loss_func = nn.CrossEntropyLoss()   # 损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 100 ## 每经过100次迭代后,输出损失
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(5):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):  
        ## 计算每个batch的
        output = MyConvnet(b_x)            # CNN在训练batch上的输出
        loss = loss_func(output, b_y)   # 交叉熵损失函数
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        loss.backward()                 # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        ## 计算迭代次数
        ## 计算每经过print_step次迭代后的输出
        if step % print_step == 0:
            ## 计算在测试集上的精度
            output = MyConvnet(test_data_x)
            _,pre_lab = torch.max(output,1)
            acc = accuracy_score(test_data_y,pre_lab)
            ## 计算每个epoch和step的模型的输出特征
            history1.log((epoch, step),
                         train_loss=loss,# 训练集损失
                         test_acc = acc, # 测试集精度
                         ## 第二个全连接层权重
                         hidden_weight=MyConvnet.fc[2].weight)
            # 可视网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_acc"])
                canvas1.draw_image(history1["hidden_weight"])
       
```
## 使用Visdom进行可视化

```python
from visdom import Visdom
```
### 可视化numpy类型的数据

```python
from sklearn.datasets import load_iris
iris_x,iris_y = load_iris(return_X_y=True)
print(iris_x.shape)
print(iris_y.shape)
```


```python
## 2D散点图
vis = Visdom()
vis.scatter(iris_x[:,0:2],Y = iris_y+1,win="windows1",env="main")
```


```python
## 3D散点图
vis.scatter(iris_x[:,0:3],Y = iris_y+1,win="3D 散点图",env="main",
            opts = dict(markersize = 4,# 点的大小
                        xlabel = "特征1",ylabel = "特征2") 
           )
```


```python
## 添加折线图
x = torch.linspace(-6,6,100).view((-1,1))
sigmoid = torch.nn.Sigmoid()
sigmoidy = sigmoid(x)
tanh = torch.nn.Tanh()
tanhy = tanh(x)
relu = torch.nn.ReLU()
reluy = relu(x)
## 连接3个张量
ploty = torch.cat((sigmoidy,tanhy,reluy),dim=1)
plotx = torch.cat((x,x,x),dim=1)
vis.line(Y=ploty,X=plotx,win="line plot",env="main",
         ##  设置线条的其它属性
         opts = dict(dash = np.array(["solid","dash","dashdot"]),
                     legend = ["Sigmoid","Tanh","ReLU"]))
```


```python
## 添加
x = torch.linspace(-6,6,100).view((-1,1))
y1 = torch.sin(x)
y2 = torch.cos(x)
## 连接2个张量
plotx = torch.cat((y1,y2),dim=1)
ploty = torch.cat((x,x),dim=1)
vis.stem(X=plotx,Y=ploty,win="stem plot",env="main",
         ##  设置图例
         opts = dict(legend = ["sin","cos"],
                     title = "茎叶图"))

```


```python
## 添加热力图
# 计算鸢尾花数据的相关系数
iris_corr = torch.from_numpy(np.corrcoef(iris_x,rowvar=False))
vis.heatmap(iris_corr,win="heatmap",env="main",
            ## 设置每个特征的名称
            opts=dict(rownames = ["x1","x2","x3","x4"],
                     columnnames =["x1","x2","x3","x4"],
                     title = "热力图"))
```


```python
## 创建新的可视化图像环境，可视化图像
##  获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):  
    if step > 0:
        break

## 输出训练图像的尺寸和标签的尺寸
print(b_x.shape)
print(b_y.shape)

```


```python
## 可视化其中的一张图片
vis.image(b_x[0,:,:,:],win="one image", env="MyimagePlot",
          opts = dict(title = "一张图像"))
```


```python
## 它形成一个大小（B / nrow，nrow）的图像网格
vis.images(b_x,win="my batch image", env="MyimagePlot",
           nrow = 16,opts = dict(title = "一个批次的图像"))
```


```python
## 可视化一段文本
texts = """A flexible tool for creating, organizing, 
and sharing visualizations of live,rich data.
Supports Torch and Numpy."""
vis.text(texts,win="text plot", env="MyimagePlot",
         opts = dict(title = "可视化文本"))
```










































