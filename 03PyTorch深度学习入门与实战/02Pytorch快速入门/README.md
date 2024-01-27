# Pytorch快速入门(无注释)

```python
import torch
import torch.nn as nn
from torchvision.transforms import Compose
```


```python
nn.Linear(in_features=10,out_features=10)
```

```python
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```
## 张量的数据类型
```python
## 导入需要的库
import torch
```

```python
## 获取张量的数据类型
torch.tensor([1.2, 3.4]).dtype
```

```python
## 张量的默认数据类型设置为其它类型
torch.set_default_tensor_type(torch.DoubleTensor)
torch.tensor([1.2, 3.4]).dtype
## 注意：set_default_tensor_type()函数只支持设置浮点类型数据
```

```python
## 将张量数据类型转化为整型
a = torch.tensor([1.2, 3.4])
print("a.dtype:",a.dtype)
print("a.long()方法:",a.long().dtype)
print("a.int()方法:",a.int().dtype)
print("a.float()方法:",a.float().dtype)
```

```python
## 恢复torch默认的数据类型
torch.set_default_tensor_type(torch.FloatTensor)
torch.tensor([1.2, 3.]).dtype
```

```python
## 获取默认的数据类型
torch.get_default_dtype()
```
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
```python

```

```python

```

```python

```




