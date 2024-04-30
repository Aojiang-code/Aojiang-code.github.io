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

这段代码用于将使用`torchviz`库生成的卷积神经网络计算图保存为一张图片。下面是对每一部分代码的详细解释：

```python
MyConvnetvis.format = "png"
```
首先，我们使用`.format`属性将可视化的计算图的保存格式设置为PNG。默认情况下，格式是PDF，但在这里我们将其更改为PNG格式。

```python
MyConvnetvis.directory = "data/chap4/MyConvnet_vis"
```
接下来，我们使用`.directory`属性指定了保存的文件夹路径。这里我们将图像保存在`data/chap4/MyConvnet_vis`目录下，以便后续的访问和处理。

```python
MyConvnetvis.view()
```
最后，我们调用`.view()`方法来保存可视化的计算图。这个方法会自动将计算图保存到之前指定的文件夹中，并以给定的格式命名图像文件。在这种情况下，它会生成一张名为`MyConvnet_vis.png`的PNG格式的图像。

通过这段代码，我们可以将可视化的卷积神经网络计算图保存为一张图片，方便进行后续的查看、存储、共享和使用。保存为图片的计算图可以直观地呈现网络结构和数据流动，供进一步分析和调试使用。





## 训练过程的可视化
### 利用tensorboardX

```python
## 从tensorboardX库中导入需要的API
from tensorboardX import SummaryWriter
SumWriter = SummaryWriter(log_dir="data/chap4/log")
```
这段代码用于从`tensorboardX`库中导入所需的API，并创建一个`SummaryWriter`对象。下面是对每一部分代码的详细解释：

```python
from tensorboardX import SummaryWriter
```

首先，我们导入了`tensorboardX`库，并从中引入了`SummaryWriter`类。这个类是`tensorboardX`库中用于创建事件文件的主要接口。

```python
SumWriter = SummaryWriter(log_dir="data/chap4/log")
```

接下来，我们创建了一个`SummaryWriter`对象，并将其赋值给变量`SumWriter`。通过调用`SummaryWriter`类的构造函数，我们可以指定一个保存日志文件的目录路径。

在这里，我们使用`log_dir`参数指定了日志文件保存的路径为`data/chap4/log`。这个路径会在指定目录下创建一个`log`文件夹，并把所有与日志相关的信息保存到这个文件夹中。

通过创建`SummaryWriter`对象，我们可以将训练过程中的各种指标和可视化数据写入到事件文件中。这些事件文件可以被`TensorBoard`程序读取和展示，以便进行训练过程的监控、可视化和分析。

总结起来，这段代码导入了`tensorboardX`库，并创建了一个`SummaryWriter`对象，用于向事件文件中写入日志和可视化数据。这是为了方便后续使用`TensorBoard`进行训练过程的可视化和分析。





###### 以下内容未看


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

#### 定义优化器
```python
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)
```
在这部分代码中，我们使用Adam优化算法来定义一个优化器`optimizer`。我们将模型`MyConvnet`的所有可训练参数作为优化器的参数，并设置学习率为0.0003。

优化器是用于更新模型参数的工具。它通过计算模型的损失函数梯度，并使用梯度信息来调整模型参数，以最小化损失函数。Adam优化算法是一种常用的优化算法，它结合了动量方法和自适应学习率的特性，可以有效地优化深度神经网络模型。

#### 定义损失函数
```python
loss_func = nn.CrossEntropyLoss()
```
接下来，我们定义了一个损失函数`loss_func`，使用交叉熵损失函数来度量模型输出与真实标签之间的差异。交叉熵损失函数通常用于多类别分类任务，例如图像识别中的物体分类问题。

交叉熵损失函数对于分类任务非常有用，它通过比较模型输出的概率分布与真实标签的概率分布，来计算模型的预测与真实标签之间的差异。该损失函数越小，表示模型的预测越接近真实标签。

#### 初始化变量
```python
train_loss = 0
print_step = 100
```
在这部分代码中，我们初始化了两个变量。`train_loss`用于累积训练过程中的损失值，初始值为0。`print_step`表示每经过多少次迭代后输出一次损失值。

`train_loss`和`print_step`是训练过程中的辅助变量，用于监测和记录每个迭代步骤的损失情况，并在特定的迭代次数触发打印操作。


#### 外层循环：遍历训练数据集
```python
for epoch in range(5):
```
这部分代码使用`range()`函数创建一个迭代器，迭代范围为0到4，表示总共进行5轮训练。

该循环在训练阶段被称为"epoch"，一次epoch表示对整个训练数据集进行一次完整的训练。每次epoch可以由多个迭代步骤组成，通过遍历训练数据集中的每个批次来实现。

#### 内层循环：遍历训练数据集的批次
```python
for step, (b_x, b_y) in enumerate(train_loader):
```
这部分代码使用`enumerate()`函数迭代遍历训练数据集的每个批次，获取批次的索引`step`和对应的输入数据`b_x`和标签数据`b_y`。

`train_loader`是训练数据集的迭代器，用于提供每个批次的数据。每个批次通常包含一组输入样本和对应的标签，用于训练模型。

#### 前向传播及损失计算
```python
output = MyConvnet(b_x)
loss = loss_func(output, b_y)
```
这部分代码进行了两个核心操作。

首先，通过将输入数据`b_x`传递给模型`MyConvnet`，实现了模型的前向传播过程。模型对输入进行处理并生成输出结果`output`，这里假设模型是一个卷积神经网络（CNN）。

其次，使用定义好的损失函数`loss_func`计算模型预测输出`output`和对应的标签`b_y`之间的损失值。损失函数通常用于度量模型输出与真实标签之间的差异，本例中使用交叉熵损失函数。

#### 梯度清零及反向传播
```python
optimizer.zero_grad()
loss.backward()
```
在优化器开始更新参数之前，需要执行两个操作。

首先，调用`optimizer.zero_grad()`方法将模型的梯度缓存清零，以防止在下一次迭代中出现重复计算。这是因为PyTorch默认会累加梯度，而不是覆盖它们。

接下来，调用`loss.backward()`方法进行反向传播。反向传播会计算损失函数相对于模型参数的梯度，并将其存储在每个参数的`.grad`属性中。这些梯度将用于更新模型参数。

#### 参数更新及损失累加
```python
optimizer.step()
train_loss = train_loss + loss
```
这部分代码执行两个操作。

首先，调用`optimizer.step()`方法来更新模型的参数。优化器根据计算得到的梯度信息来调整模型参数，以最小化损失函数。在本例中，使用的是Adam优化算法。

其次，将当前批次的损失值`loss`累加到`train_loss`变量中。通过迭代训练过程中的每个批次，我们可以得到整个训练数据集上的累计损失值。

#### 记录训练过程中的指标和可视化
```python
niter = epoch * len(train_loader) + step + 1
if niter % print_step == 0:
    # 记录训练集损失函数
    SumWriter.add_scalar("train loss", train_loss.item() / niter, global_step=niter)
    output = MyConvnet(test_data_x)
    _, pre_lab = torch.max(output, 1)
    acc = accuracy_score(test_data_y, pre_lab)
    # 记录测试集准确率
    SumWriter.add_scalar("test acc", acc.item(), niter)
    # 记录训练数据的可视化图像
    b_x_im = vutils.make_grid(b_x, nrow=12)
    SumWriter.add_image('train image sample', b_x_im, niter)
    # 使用直方图可视化网络中参数的分布情况
    for name, param in MyConvnet.named_parameters():
        SumWriter.add_histogram(name, param.data.numpy(), niter)
```
这部分代码用于记录训练过程中的指标和进行可视化操作。

首先，通过计算`niter`（迭代次数）来判断是否需要输出信息或执行可视化操作。当`niter`能够被`print_step`整除时，达到了设定的迭代步骤。

然后，在TensorBoard日志中记录训练集损失函数的值。使用`SumWriter.add_scalar()`方法将平均损失添加到TensorBoard日志文件中，以便后续可视化和分析。

接着，对测试集数据进行模型预测，并计算预测精度。使用预测精度的值将其添加到TensorBoard日志中，以评估模型在测试集上的性能。

此外，还记录训练数据的可视化图像。使用`vutils.make_grid()`方法处理训练数据`b_x`，将其排列为网格状并添加到TensorBoard日志中。可视化训练数据有助于了解模型对图像的处理效果。

最后，使用直方图可视化网络中参数的分布情况。通过遍历模型参数并使用`SumWriter.add_histogram()`方法将参数值的分布添加到TensorBoard日志中。直方图可以提供有关参数值范围、分布及变化情况等信息。

通过这段代码，我们完成了一个完整的模型训练过程，并使用TensorBoard记录了训练过程中的指标和可视化信息。这些信息对于分析和优化模型性能非常有帮助。















```python
# ## 为日志中添加训练数据的可视化图像，使用最后一个batch的图像
# ## 将一个batch的数据进行预处理
# b_x_im = vutils.make_grid(b_x,nrow=12)
# SumWriter.add_image('train image sample', b_x_im)
```
这段代码是用于将训练数据中的图像样本可视化并添加到TensorBoard日志中。下面将逐个解释每个代码块的作用和具体实现。

```python
b_x_im = vutils.make_grid(b_x, nrow=12)
```
首先，通过调用`vutils.make_grid()`方法将一个批次的图像`b_x`进行处理。`make_grid()`函数会将多张图像按照指定的行数和列数排列成一个网格状的图像，方便进行展示。`nrow=12`参数表示设置每行显示的图像数量为12。

```python
SumWriter.add_image('train image sample', b_x_im)
```
接着，通过`SumWriter.add_image()`方法向TensorBoard日志中添加图像。第一个参数是图像的名称，这里命名为'train image sample'。第二个参数是待添加的图像数据`b_x_im`，即经过处理后的网格状图像。

这段代码的目的是在训练过程中，每经过一定次数的迭代后，将最后一个batch的图像样本转换成一个网格状的图像，并添加到TensorBoard日志中，以便可视化查看训练数据的样本情况。

其中，`vutils.make_grid()`方法负责将多张图像进行排列，使得它们按指定的行数和列数组成一个网格状图像。这样做有利于更好地观察和比较多张图像之间的关系。

`SumWriter.add_image()`方法则负责将处理后的图像数据添加到TensorBoard日志中。这样，可以通过TensorBoard来查看训练中生成的图像网格，并对图像内容进行直观分析。

该代码块为项目提供了一个非常有用的功能，即可视化训练数据的图像样本。通过观察图像，我们可以更好地理解模型在训练中遇到的数据，以及模型对不同类别的数据的表示能力和区分度。这有助于我们对模型的学习过程和性能进行调试和改进。







```python
# ## 使用直方图可视化网络中参数的分布情况
# for name, param in MyConvnet.named_parameters():
#     SumWriter.add_histogram(name, param.data.numpy())
```
这段代码用于使用直方图对神经网络中的参数进行可视化，并将其添加到TensorBoard日志中。下面将逐个解释每个代码块的作用和具体实现。

```python
for name, param in MyConvnet.named_parameters():
    SumWriter.add_histogram(name, param.data.numpy())
```
首先，通过`MyConvnet.named_parameters()`方法遍历了模型`MyConvnet`中的所有参数。`named_parameters()`函数返回一个迭代器，每次迭代返回参数的名称（name）和参数的值（param）。

然后，使用循环将每个参数的名称和数值传递给`SumWriter.add_histogram()`方法。该方法用于将直方图添加到TensorBoard日志中。第一个参数`name`表示用于标识直方图的名称，通常使用参数的名称来命名。第二个参数`param.data.numpy()`是将参数的数值转换为NumPy数组的形式，以便传递给`add_histogram()`方法。

这段代码的目的是在训练过程中，周期性地计算并可视化网络模型中所有参数的分布情况。直方图以一种直观的方式展示了参数取值的频率分布，表达了模型各参数值的集中程度、范围和偏移情况。通过观察直方图，可以帮助我们更好地理解模型参数的学习过程和收敛状态，辅助分析和调整模型的训练结果。

`MyConvnet.named_parameters()`方法用于获取模型中所有参数的名称和数值。通过遍历每个参数，可以将它们的数值传递给`add_histogram()`方法，并指定相应的直方图名称。然后，TensorBoard可通过这些直方图展示参数的分布情况，提供了对模型内部运行情况的可视化分析工具。

该代码块为项目提供了一个重要的功能，即在训练过程中监测并分析网络模型中参数的分布情况。通过观察直方图，我们可以获得对模型参数值分布的直观认识，有助于发现潜在的问题、调整训练策略和优化算法，以改进模型的性能和收敛速度。










### 利用hiddenlayer


```python
import hiddenlayer as hl
import time
```
这段代码引入了名为`hiddenlayer`的库，并导入了名为`hl`的模块。同时，引入了`time`模块。下面将逐个解释每个代码块的作用和具体实现。

```python
import hiddenlayer as hl
```
该行将名为`hiddenlayer`的库导入到当前脚本中，以便后续使用其中的功能。`hiddenlayer`库提供了一个易于使用且功能强大的工具，用于可视化神经网络的结构图、流程图和训练过程等信息。

```python
import time
```
该行导入了名为`time`的模块，用于处理与时间相关的操作。在此代码片段中，可能会使用`time`模块来计算某些操作所花费的时间，例如模型训练的持续时间、迭代时间等。

通过上述两个导入语句，我们可以利用`hiddenlayer`库提供的工具进行神经网络的可视化展示，通过图形化界面直观地观察和分析网络结构和训练过程。而`time`模块的导入则可用于计时等时间相关的操作，方便统计和追踪代码执行的时间开销。

然而，在给定的代码片段中，这些导入语句没有被其他代码使用，因此无法提供进一步的信息。若需要更详细的解释，请提供与这些导入语句相关的其他代码或问题，我将很乐意为您提供帮助。







```python
## 初始化MyConvnet
MyConvnet = ConvNet()
print(MyConvnet)
```
这段代码主要涉及神经网络模型的初始化，下面将逐个解释每个代码块的作用和具体实现。

```python
MyConvnet = ConvNet()
```
首先，使用`ConvNet()`创建了一个名为`MyConvnet`的神经网络对象。这里假设在之前已经定义好了一个名为`ConvNet`的类，该类实现了自定义的卷积神经网络结构。通过调用`ConvNet()`来实例化这个类，我们可以得到一个可用的神经网络对象。

```python
print(MyConvnet)
```
接着，通过`print(MyConvnet)`语句将`MyConvnet`打印出来。这样可以显示该神经网络对象的字符串表示形式，即在`ConvNet`类中所定义的`__str__()`方法的返回值。这通常用于查看网络对象的基本信息，例如网络结构、层的数量和参数等。

该段代码的目的是初始化一个名为`MyConvnet`的卷积神经网络对象，并打印出其基本信息。通过实例化网络对象，我们可以方便地访问和操作该网络，包括设置网络参数、进行前向传播和反向传播等操作。

需要注意的是，这段代码中的`ConvNet()`用于示例目的，实际上需要在代码中定义一个名为`ConvNet`的类来描述具体的神经网络结构和功能。因此，对于详细的解释和说明，请提供`ConvNet`类的定义或相关代码，以便更具体地了解该神经网络的结构和功能。








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

这段代码主要用于定义优化器、损失函数以及记录训练过程中的指标和可视化。下面将逐个解释每个代码块的作用和具体实现。

```python
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)
```
首先，通过`torch.optim.Adam`创建了一个Adam优化器对象。将`MyConvnet.parameters()`作为参数传递给优化器，使得该优化器可以更新神经网络模型`MyConvnet`中所有可学习的参数（权重和偏置）。`lr=0.0003`表示设置学习率为0.0003，即优化器在每次更新参数时所采用的步长大小。

然后，通过`nn.CrossEntropyLoss()`创建了一个交叉熵损失函数对象`loss_func`。交叉熵损失函数通常用于多分类任务中，用于衡量模型输出与目标类别之间的差异。

```python
history1 = hl.History()
canvas1 = hl.Canvas()
```
接着，创建`hiddenlayer`库中的`History`和`Canvas`对象。`History`用于记录训练过程中的指标，例如训练集损失和测试集精度等。`Canvas`则用于进行数据可视化，包括绘制损失曲线和精度曲线等。

```python
print_step = 100
```
定义了一个变量`print_step`，用于指定每经过多少次迭代后进行输出。在此例中，每经过100次迭代（即每经过100个batch）后，在控制台打印一些额外的信息。

```python
for epoch in range(5):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = MyConvnet(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % print_step == 0:
            output = MyConvnet(test_data_x)
            _,pre_lab = torch.max(output,1)
            acc = accuracy_score(test_data_y,pre_lab)
            history1.log((epoch, step),
                         train_loss=loss,
                         test_acc=acc,
                         hidden_weight=MyConvnet.fc[2].weight)
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_acc"])
                canvas1.draw_image(history1["hidden_weight"])
```
最后，使用嵌套的循环对模型进行迭代训练。外层循环`for epoch in range(5)`用于遍历指定轮数的训练过程（这里设定为5轮）。内层循环`for step, (b_x, b_y) in enumerate(train_loader)`用于遍历训练数据集中的每个批次。

在每个批次中，首先通过前向传播计算神经网络模型`MyConvnet`在当前输入`b_x`上的输出结果 `output`。然后，使用损失函数`loss_func`计算输出结果和目标`b_y`之间的损失`loss`。接着，通过`optimizer.zero_grad()`将优化器中的梯度置零，以便进行梯度更新。然后，调用`loss.backward()`方法进行反向传播计算梯度，并使用`optimizer.step()`方法根据梯度值来更新模型参数。

在每经过`print_step`次迭代后，执行一些额外的操作。首先，计算模型在测试集上的精度`acc`。然后，通过`history1.log()`方法将当前的训练损失、测试精度以及指定层的权重信息记录到`history1`中。最后，通过`canvas1`对象进行可视化，使用`draw_plot()`绘制训练损失曲线和测试精度曲线，使用`draw_image()`绘制指定层的权重图像。

该段代码的目的是对已定义好的神经网络模型`MyConvnet`进行迭代训练，并在每个batch或指定的迭代步骤后记录训练过程中的指标和进行可视化展示。通过使用Adam优化器和交叉熵损失函数进行模型训练，可以实现神经网络的参数优化和模型性能提升。记录和可视化训练过程的指标有助于了解模型的学习进展和性能表现，辅助分析和调整模型的训练策略和超参数设置。














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

