# Pytorch深度神经网络及训练

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
























































