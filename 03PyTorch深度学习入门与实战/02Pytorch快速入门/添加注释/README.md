# Pytorch快速入门(有注释)

```python
import torch
import torch.nn as nn
from torchvision.transforms import Compose
```
上述代码导入了一些与PyTorch深度学习框架相关的模块和类，并定义了一个名为 `Compose` 的函数。

具体解释如下：

- `import torch`: 导入了PyTorch库，这是一个用于构建深度神经网络的开源机器学习库。

- `import torch.nn as nn`: 导入了PyTorch中的`nn`模块，该模块包含了构建神经网络所需的各种层（例如全连接层、卷积层等）和损失函数（例如交叉熵损失函数）。

- `from torchvision.transforms import Compose`: 从`torchvision.transforms`模块中导入`Compose`函数。`Compose`是一个用于组合多个图像预处理操作的函数。通过使用`Compose`函数，可以将多个图像预处理操作串联起来便于一次性应用到输入数据上。

这段代码的目的是导入PyTorch和相关模块以及定义一个函数，为后续构建和训练深度神经网络做准备。

```python
nn.Linear(in_features=10,out_features=10)
```
上述代码使用 `nn.Linear` 类创建了一个线性层（全连接层），该线性层具有 10 个输入特征和 10 个输出特征。

具体解释如下：

- `nn.Linear`: 是PyTorch中的一个类，用于创建线性层。线性层也被称为全连接层，它将输入数据与权重矩阵相乘，再加上偏置向量，得到输出结果。

- `in_features=10`: 这是 `nn.Linear` 类的参数，指定线性层的输入特征数量。在这个例子中，指定的输入特征数量为 10。

- `out_features=10`: 这是 `nn.Linear` 类的参数，指定线性层的输出特征数量。在这个例子中，指定的输出特征数量为 10。

因此，以上代码的作用是创建了一个具有 10 个输入特征和 10 个输出特征的线性层。在模型训练过程中，该线性层将根据输入数据学习权重和偏置，以便拟合输入和输出之间的关系，并在后续的前向传播中生成输出结果。
```python
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```
上述代码用于设置和配置Jupyter Notebook中的内联图形显示方式。

具体解释如下：

- `%config InlineBackend.figure_format = 'retina'`: 这行代码设置图像显示的清晰度为 retina 级别。Retina是一种高分辨率显示技术，通过使用更多的像素点来提高图像的细节和清晰度。

- `%matplotlib inline`: 这行代码指示Jupyter Notebook在每个代码块执行后自动显示生成的图形，并将其嵌入到Notebook中（内联模式）。这使得在Notebook中可以直接输出和查看图像结果，而不需要额外的命令或操作。

综上所述，这些代码片段通过设置图像显示的清晰度和指定内联图形显示模式，提供了更高质量的图像展示和方便的交互式数据可视化功能。
## 张量的数据类型
```python
## 导入需要的库
import torch
```
上述代码是用于导入PyTorch库，以便在Python脚本或Jupyter Notebook中使用PyTorch的功能。

具体解释如下：

- `import torch`: 这行代码导入了PyTorch库，它是一个用于构建深度神经网络的开源机器学习库。PyTorch提供了丰富的工具和函数，使得构建、训练和部署神经网络变得更加方便和高效。

通过导入PyTorch库，我们可以使用其提供的模型定义、优化算法、损失函数、数据处理工具等功能，帮助我们更轻松地进行深度学习任务的开发和实验。
```python
## 获取张量的数据类型
torch.tensor([1.2, 3.4]).dtype
```
上述代码用于获取一个张量（Tensor）的数据类型。

具体解释如下：

- `torch.tensor([1.2, 3.4])`: 这行代码创建了一个包含两个元素的张量。张量是PyTorch中的主要数据结构，可以表示多维数组。

- `.dtype`: 这是张量对象的属性，用于获取张量的数据类型。数据类型指示张量中存储元素的方式，例如浮点数、整数等。

在这个例子中，我们创建了一个包含浮点数的张量。通过使用 `.dtype` 属性，可以获知该张量的数据类型是什么，以便进一步判断和使用。

因此，以上代码的作用就是获取一个张量的数据类型，并返回结果。对于本例中的张量而言，返回的结果应该是 `torch.float32` 或类似的数据类型信息。
```python
## 张量的默认数据类型设置为其它类型
torch.set_default_tensor_type(torch.DoubleTensor)
torch.tensor([1.2, 3.4]).dtype
## 注意：set_default_tensor_type()函数只支持设置浮点类型数据
```
上述代码用于将PyTorch张量的默认数据类型设置为其他类型，特别是浮点类型。

具体解释如下：

- `torch.set_default_tensor_type(torch.DoubleTensor)`: 这行代码调用了`torch.set_default_tensor_type()`函数，并传入`torch.DoubleTensor`作为参数。该函数的作用是设置PyTorch张量的默认数据类型为双精度浮点型（`torch.double`）。

- `torch.tensor([1.2, 3.4]).dtype`: 这行代码创建了一个包含两个元素的张量，并通过`.dtype`属性获取该张量的数据类型。

在这个例子中，我们先调用了`torch.set_default_tensor_type()`函数来设置张量的默认数据类型为双精度浮点型。然后，我们创建了一个包含浮点数的张量，由于我们已经将默认数据类型设置为双精度浮点型，因此该张量的数据类型应该是`torch.float64`或类似的双精度浮点数据类型。

需要注意的是，`torch.set_default_tensor_type()`函数只支持设置浮点类型数据，即可以使用诸如`torch.FloatTensor`、`torch.DoubleTensor`等类型，但不支持整数类型。如果要设置整数类型的默认数据类型，请使用`torch.set_default_dtype()`函数。

综上所述，以上代码的作用是将PyTorch张量的默认数据类型设置为双精度浮点型，并根据该设置创建一个张量并获取其数据类型。
```python
## 将张量数据类型转化为整型
a = torch.tensor([1.2, 3.4])
print("a.dtype:",a.dtype)
print("a.long()方法:",a.long().dtype)
print("a.int()方法:",a.int().dtype)
print("a.float()方法:",a.float().dtype)
```
上述代码的结果和解释如下：

```python
a = torch.tensor([1.2, 3.4])
print("a.dtype:", a.dtype)
print("a.long()方法:", a.long().dtype)
print("a.int()方法:", a.int().dtype)
print("a.float()方法:", a.float().dtype)
```

输出结果：
```
a.dtype: torch.float32
a.long()方法: torch.int64
a.int()方法: torch.int32
a.float()方法: torch.float32
```

解释：
- `a.dtype`: 输出原始张量 `a` 的数据类型，默认情况下为 `torch.float32`，该类型是指定张量中元素的数据类型。

- `a.long().dtype`: 使用 `.long()` 方法将张量 `a` 的数据类型转换为长整型（64位带符号整数），所以返回 `torch.int64`。

- `a.int().dtype`: 使用 `.int()` 方法将张量 `a` 的数据类型转换为整型（32位带符号整数），所以返回 `torch.int32`。

- `a.float().dtype`: 使用 `.float()` 方法将张量 `a` 的数据类型转换回浮点型（32位浮点数），与原始的 `a` 张量数据类型保持一致，所以返回 `torch.float32`。

通过使用不同的类型转换方法，可以将张量的数据类型修改为所需的类型。这对于在深度学习模型中处理输入数据和进行计算时非常有用，因为可能需要匹配特定的数据类型要求或减少内存使用。在给定代码中，我们展示了如何将浮点型张量转换为长整型、整型和再转回浮点型，同时演示了不同数据类型的具体结果。
```python
## 恢复torch默认的数据类型
torch.set_default_tensor_type(torch.FloatTensor)
torch.tensor([1.2, 3.]).dtype
```
上述代码的结果和解释如下：

```python
torch.set_default_tensor_type(torch.FloatTensor)
torch.tensor([1.2, 3.]).dtype
```

输出结果：
```
torch.float32
```

解释：
- `torch.set_default_tensor_type(torch.FloatTensor)`: 这行代码调用了 `torch.set_default_tensor_type()` 函数，并将参数 `torch.FloatTensor` 传递给它。该函数的作用是将PyTorch张量的默认数据类型重新设置为单精度浮点数型（`torch.float32`）。

- `torch.tensor([1.2, 3.])`: 这行代码创建了一个张量，包含两个元素 `[1.2, 3.]`。由于我们在之前的代码中将默认数据类型设置为单精度浮点数型，因此这个张量的数据类型将会是 `torch.float32`。

因此，根据以上代码，我们将默认数据类型恢复为单精度浮点数型后，创建的新张量的数据类型就会是 `torch.float32`。
```python
## 获取默认的数据类型
torch.get_default_dtype()
```
上述代码的结果和解释如下：

```python
torch.get_default_dtype()
```

输出结果：
```
torch.float32
```

解释：
- `torch.get_default_dtype()`: 这行代码调用了 `torch.get_default_dtype()` 函数，该函数的作用是返回当前设置的默认数据类型。

根据以上代码，我们可以得知默认数据类型当前被设置为 `torch.float32`，这是单精度浮点数的数据类型。这是在没有进行特别设置时，PyTorch默认使用的数据类型。
## 生成张量
### 基本方法
```python
A = torch.tensor([[1.0,1.0],[2,2]])
A
```

```python
## 获取张量的形状
A.shape
```

```python
## 获取张量的形状
A.size()
```

```python
## 计算张量中所含元素的个数
A.numel()
```

```python
## 指定张量的数据类型和是否要计算梯度
B = torch.tensor((1,2,3),dtype=torch.float32,requires_grad=True)
B
```

```python
## 因为张量B是可计算梯度的，所以可以计算sum(B^2)的梯度
y  = B.pow(2).sum()
y.backward()
B.grad
```

```python
## 注意只有浮点类型的张量允许计算梯度
# B = torch.tensor((1,2,3),dtype=torch.int32,requires_grad=True)
```

```python
## 利用torch.Tensor()获得张量
## 使用预先存在的数据创建张量
C = torch.Tensor([1,2,3,4])
C
```

```python
## 创建具有特定大小的张量
D = torch.Tensor(2,3)
D
```

```python
## 创建与另一个张量相同大小和类型相同的张量
torch.ones_like(D)
```

```python
torch.zeros_like(D)
```

```python
torch.rand_like(D)
```

```python
## 创建一个类型相似但尺寸不同的张量
E = [[1,2],[3,4]]
E = D.new_tensor(E)
print("D.dtype : ",D.dtype)
print("E.dtype : ",E.dtype)
E
```

```python
D.new_full((3,3), fill_value = 1)
D.new_zeros((3,3))
D.new_empty((3,3))
D.new_empty((3,3))
```

```python
## 利用numpy数组生成张量
import numpy as np
F = np.ones((3,3))
## 使用torch.as_tensor()函数
Ftensor = torch.as_tensor(F)
Ftensor
```

```python
## 使用torch.from_numpy()函数
Ftensor = torch.from_numpy(F)
Ftensor
```

```python
## 使用张量的.numpy()将张量转化为numpy数组
Ftensor.numpy()
```
### 随机数生成张量






