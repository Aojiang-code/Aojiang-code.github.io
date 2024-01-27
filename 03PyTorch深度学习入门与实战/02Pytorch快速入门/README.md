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
## 设置随机数种子
torch.manual_seed(123)
```

```python
## 通过指定均值和标准差生成随机数
torch.manual_seed(123)
A = torch.normal(mean = 0.0,std = torch.tensor(1.0))
A
```

```python
## 通过指定均值和标准差生成随机数
torch.manual_seed(123)
A = torch.normal(mean = 0.0,std=torch.arange(1,5.0))
A
```



```python
torch.manual_seed(123)
A = torch.normal(mean = torch.arange(1,5.0),std=torch.arange(1,5.0))
A
```


```python
## 在区间[0,1)上生成服从均匀分布的张量
torch.manual_seed(123)
B = torch.rand(3,4)
B
```



```python
## 生成和其它张量尺寸相同的随机数张量
torch.manual_seed(123)
C = torch.ones(2,3)
D = torch.rand_like(C)
D
```



```python
## 生成服从标准正态分布的随机数
print(torch.randn(3,3))
print(torch.randn_like(C))
```



```python
## 将0～10（不包括10）之间的整数随机排序
torch.manual_seed(123)
torch.randperm(10)
```
###  其它生成张量的函数


```python
## 使用torch.arange()生成张量
torch.arange(start=0, end = 10, step=2)
```



```python
## 在范围内生成固定数量的等间隔张量
torch.linspace(start = 1, end = 10, steps=5)
```



```python
## 生成以对数间隔的点
torch.logspace(start=0.1, end=1.0, steps=5)
```



```python
10**(torch.linspace(start = 0.1, end = 1, steps=5))
```



```python
torch.zeros(3,3)
torch.ones(3,3)
torch.eye(3)
torch.empty(3,3)
torch.full((3,3),fill_value = 0.25)
```

## 张量的操作
### 改变张量的尺寸
```python
## 使用tensor.reshape()函数设置张量的尺寸
A = torch.arange(12.0).reshape(3,4)
A
```



```python
## 使用torch.reshape()
torch.reshape(input = A,shape = (2,-1))
```



```python
## 使用resize_方法
A.resize_(2,6)
A
```



```python
## 使用
B = torch.arange(10.0,19.0).reshape(3,3)
A.resize_as_(B)
```



```python
B
```



```python
## torch.unsqueeze()返回在指定维度插入尺寸为1的新张量
A = torch.arange(12.0).reshape(2,6)
B = torch.unsqueeze(A,dim = 0)
B.shape
```



```python
## torch.squeeze()函数移除所有维度为1的维度
C = B.unsqueeze(dim = 3)
print("C.shape : ",C.shape)
D = torch.squeeze(C)
print("D.shape : ",D.shape)
## 移除指定维度为1的维度
E = torch.squeeze(C,dim = 0)
print("E.shape : ",E.shape)
```



```python
## 使用.expand()方法拓展张量
A = torch.arange(3)
B = A.expand(3,-1)
B
```



```python
## 使用.expand_as()方法拓展张量
C = torch.arange(6).reshape(2,3)
B = A.expand_as(C)
B
```



```python
## 使用.repeat()方法拓展张量
D = B.repeat(1,2,2)
print(D)
print(D.shape)
```

### 获取张量中的元素

```python
## 利用切片和索引获取张量中的元素
A = torch.arange(12).reshape(1,3,4)
A
```



```python
A[0]
```



```python
## 获取第0维度下的矩阵前两行元素
A[0,0:2,:]
```



```python
## 获取第0维度下的矩阵，最后一行－4～－1列
A[0,-1,-4:-1]
```



```python
## 根据条件筛选
B = - A
torch.where(A>5,A,B)
```



```python
## 获取A中大于5的元素
A[ A > 5]
```



```python
## 获取其中的某个元素
A[0,2,3]
```



```python
## 获取矩阵张量的下三角部分
torch.tril(A,diagonal=0,)
```



```python
## diagonal参数控制要考虑的对角线
torch.tril(A,diagonal=1)
```



```python
## 获取矩阵张量的上三角部分
torch.triu(A,diagonal=0)
```



```python
## 获取矩阵张量的上三角部分,input,需要是一个二维的张量
C = A.reshape(3,4)
print(C)
print(torch.diag(C,diagonal=0))
print(torch.diag(C,diagonal=1))
```



```python
## 提供对角线元素生成矩阵张量
torch.diag(torch.tensor([1,2,3]))
```

### 拼接和拆分

```python
## 在给定维度中连接给定的张量序列
A = torch.arange(6.0).reshape(2,3)
B = torch.linspace(0,10,6).reshape(2,3)
## 在0纬度连接张量
C = torch.cat((A,B),dim=0)
C
```



```python
## 在1纬度连接张量
D = torch.cat((A,B),dim=1)
D
```



```python
## 在1纬度连接3个张量
E = torch.cat((A[:,1:2],A,B),dim=1)
E
```



```python
## 沿新维度连接张量
F = torch.stack((A,B),dim=0)
print(F)
print(F.shape)
## 2个2＊3的矩阵组合在一起
```



```python
G = torch.stack((A,B),dim=2)
print(G)
print(G.shape)
## 2＊3*2的矩阵组合在一起
```



```python
## 将张量分割为特定数量的块
## 在行上将张量E分为两块
torch.chunk(E,2,dim=0)
```



```python
D1,D2 = torch.chunk(D,2,dim=1)
print(D1)
print(D2)
```



```python
## 如果沿给定维度dim的张量大小不能被块整除，则最后一个块将最小
E1,E2,E3 = torch.chunk(E,3,dim=1)
print(E1)
print(E2)
print(E3)
```



```python
## 将张量切分为块,指定每个块的大小
D1,D2,D3 = torch.split(D,[1,2,3],dim=1)
print(D1)
print(D2)
print(D3)
```

## 张量计算
### 比较大小

$\lvert \text{self} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert$$

```python
## 比较两个数是否接近
A = torch.tensor([10.0])
B = torch.tensor([10.1])
print(torch.allclose(A, B, rtol=1e-05, atol=1e-08, equal_nan=False))
print(torch.allclose(A, B, rtol=0.1, atol=0.01, equal_nan=False))
```



```python
## 如果equal_nan=True，那么缺失值可以判断接近
A  = torch.tensor(float("nan"))
print(torch.allclose(A, A,equal_nan=False))
print(torch.allclose(A, A,equal_nan=True))
```



```python
## 计算元素是否相等
A = torch.tensor([1,2,3,4,5,6])
B = torch.arange(1,7)
C = torch.unsqueeze(B,dim = 0)
print(torch.eq(A,B))
print(torch.eq(A,C))
```



```python
## 判断两个张量是否具有相同的尺寸和元素
print(torch.equal(A,B))
print(torch.equal(A,C))
```



```python
## 逐元素比较大于等于
print(torch.ge(A,B))
print(torch.ge(A,C))
```



```python
## 大于
print(torch.gt(A,B))
print(torch.gt(A,C))
```



```python
## 小于等于
print(torch.le(A,B))
print(torch.lt(A,C))
```



```python
## 不等于
print(torch.ne(A,B))
print(torch.ne(A,C))
```



```python
## 判断是否为缺失值
torch.isnan(torch.tensor([0,1,float("nan"),2]))
```

### 基本运算

```python
## 矩阵逐元素相乘
A = torch.arange(6.0).reshape(2,3)
B = torch.linspace(10,20,steps=6).reshape(2,3)
print("A:",A)
print("B:",B)
print(A * B)
## 逐元素相除
print(A / B)
```



```python
## 逐元素相加
print(A + B)
## 逐元素相减
print(A - B)
## 逐元素整除
print(B//A)
```



```python
## 张量的幂
print(torch.pow(A,3))
print(A ** 3)
```



```python
## 张量的指数
torch.exp(A)
```



```python
## 张量的对数
torch.log(A)
```



```python
## 张量的平方根
print(torch.sqrt(A))
print(A**0.5)
```



```python
## 张量的平方根倒数
print(torch.rsqrt(A))
print( 1 / (A**0.5))
```



```python
## 张量数据裁剪
torch.clamp_max(A,4)
```



```python
## 张量数据裁剪
torch.clamp_min(A,3)
```



```python
## 张量数据裁剪
torch.clamp(A,2.5,4)
```



```python
## 矩阵的转置
C = torch.t(A)
C
```



```python
## 矩阵运算，矩阵相乘,A的行数要等于C的列数
A.matmul(C)
```



```python
A = torch.arange(12.0).reshape(2,2,3)
B = torch.arange(12.0).reshape(2,3,2)
AB = torch.matmul(A,B)
AB
```



```python
## 矩阵相乘只计算最后面的两个纬度的乘法
print(AB[0].eq(torch.matmul(A[0],B[0])))
print(AB[1].eq(torch.matmul(A[1],B[1])))
```



```python
## 计算矩阵的逆
C = torch.rand(3,3)
D = torch.inverse(C)
torch.mm(C,D)
```



```python
## 计算张量矩阵的迹，对角线元素的和
torch.trace(torch.arange(9.0).reshape(3,3))
```

### 统计相关的计算

```python
## 1维张量的最大值和最小值
A = torch.tensor([12.,34,25,11,67,32,29,30,99,55,23,44])
## 最大值及位置
print("最大值:",A.max())
print("最大值位置:",A.argmax())
## 最小值及位置
print("最小值:",A.min())
print("最小值位置:",A.argmin())
```



```python
## 2维张量的最大值和最小值
B = A.reshape(3,4)
print("2-D张量B:\n",B)
## 最大值及位置(每行)
print("最大值:\n",B.max(dim=1))
print("最大值位置:",B.argmax(dim=1))
## 最小值及位置(每列)
print("最小值:\n",B.min(dim=0))
print("最小值位置:",B.argmin(dim=0))
```



```python
## 张量排序,分别输出从小到大的排序结果和相应的元素在元素位置的索引
torch.sort(A)
```



```python
## 按照降序排列
torch.sort(A,descending=True)
```



```python
## 对2-D张量进行排序
Bsort, Bsort_id= torch.sort(B)
print("B sort:\n",Bsort)
print("B sort index:\n",Bsort_id)
print("B argsort:\n",torch.argsort(B))
```



```python
## 获取张量前几个大的数值
torch.topk(A,4)
```



```python
## 获取2D张量每列前几个大的数值
Btop2,Btop2_id = torch.topk(B,2,dim=0)
print("B 每列 top2:\n",Btop2)
print("B 每列 top2 位置:\n",Btop2_id)
```



```python
## 获取张量第K小的数值和位置
torch.kthvalue(A,3)
```



```python
## 获取2D张量第K小的数值和位置
torch.kthvalue(B,3,dim = 1)
```



```python
## 获取2D张量第K小的数值和位置
Bkth,Bkth_id = torch.kthvalue(B,3,dim = 1,keepdim=True)
Bkth
```



```python
## 平均值,计算每行的均值
print(torch.mean(B,dim = 1,keepdim = True))
## 平均值,计算每列的均值
print(torch.mean(B,dim = 0,keepdim = True))
```



```python
## 计算每行的和
print(torch.sum(B,dim = 1,keepdim = True))
## 计算每列的和
print(torch.sum(B,dim = 0,keepdim = True))
```



```python
## 按照行计算累加和
print(torch.cumsum(B,dim = 1))
## 按照列计算累加和
print(torch.cumsum(B,dim = 0))
```



```python
## 计算每行的中位数
print(torch.median(B,dim = 1,keepdim = True))
## 计算每列的中位数
print(torch.median(B,dim = 0,keepdim = True))
```



```python
## 按照行计算乘积
print(torch.prod(B,dim = 1,keepdim = True))
## 按照列计算乘积
print(torch.prod(B,dim = 0,keepdim = True))
```



```python
## 按照行计算累乘积
print(torch.cumprod(B,dim = 1))
## 按照列计算累乘积
print(torch.cumprod(B,dim = 0))
```


```python
## 标准差
torch.std(A)
```
## Pytorch中的自动微分

```python
## torch.autograd.backward()函数计算网络图中给定张量的和
## 如计算 y = sum(x^2+2*x+1)
x = torch.tensor([[1.0,2.0],[3.0,4.0]],requires_grad=True) ## 默认requires_grad＝False
y = torch.sum(x**2+2*x+1)
print("x.requires_grad:",x.requires_grad)
print("y.requires_grad:",y.requires_grad)
print("x:",x)
print("y:",y)

```


```python
## 计算y在x上的梯度
y.backward()
x.grad
```
## torch.nn模块
### torch.nn.Conv2d()

使用一张图像来展示经过卷积后的图像效果


使用灰度图像



```python
## 使用一张图像来展示经过卷机后的图像效果
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

```


```python
import numpy as np
```


```python
## 读取图像－转化为灰度图片－转化为numpy数组
myim = Image.open("data/chap2/Lenna.png")
myimgray = np.array(myim.convert("L"),dtype=np.float32)
## 可视化图片
plt.figure(figsize=(6,6))
plt.imshow(myimgray,cmap=plt.cm.gray)
plt.axis("off")
plt.show()
```


```python
## 读取图像－转化为灰度图片－转化为numpy数组
myim = Image.open(r"C:\Users\Administrator\Desktop\teacher_tang\《PyTorch深度学习入门与实践》\请用电脑客户端勾选本文件夹整体下载！PyTorch深度学习入门与实战【配套资源】\程序\programs.7z\programs\data\chap2\Lenna.png")
myimgray = np.array(myim.convert("L"),dtype=np.float32)
## 可视化图片
plt.figure(figsize=(6,6))
plt.imshow(myimgray,cmap=plt.cm.gray)
plt.axis("off")
plt.show()
```


```python
h,w = myimgray.shape
h,w
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```





