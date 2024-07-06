import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

"Contextual Transformer Networks for Visual Recognition"

class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape

        k1=self.key_embed(x)  # 编码静态上下文信息key,表示为k1: (B,C,H,W) --> (B,C,H,W)
        v=self.value_embed(x).view(bs,c,-1)  # 编码value矩阵: (B,C,H,W) --> (B,C,H,W) --> (B,C,HW)

        y=torch.cat([k1,x],dim=1)  # 将上下文信息key和query在通道上进行拼接: (B,2C,H,W)

        att=self.attention_embed(y) # 通过两个连续的1×1卷积操作: (B,2C,H,W)-->(B,D,H,W)-->(B,C×k×k,H,W)   这里的C:把它看作是注意力头的个数
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)  # (B,C×k×k,H,W) --> (B,C,k×k,H,W)

        # (B,C,k×k,H,W) --> (B,C,H,W) --> (B,C,HW)   每个坐标点在每个注意力头的的注意力矩阵为：k×k, 然后对窗口内的值取平均, 因此: (k×k,HW)-> (1,HW), 每个坐标点只有一个值 （忽略BC维度）
        att=att.mean(2,keepdim=False).view(bs,c,-1)
        # 对N=HW个坐标点(虽然每个坐标点现在只有一个值,但是是通过k×k窗口内的值共同获得的,利用了上下文信息),使用softmax求权重, 然后使用权重与Value相乘，生成动态上下文表示
        k2=F.softmax(att,dim=-1)*v  # 得到动态上下文表示k2: (B,C,HW) * (B,C,HW) =  (B,C,HW)    权重*Value
        k2=k2.view(bs,c,h,w)

        return k1+k2


# 简单来讲, 43-49行代码的含义就是: 融合静态上下文信息k1和query信息,来生成每个像素点的权重。 这个权重是基于上下文信息获得的,所以是有效的。
if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(1,512,7,7)
    Model = CoTAttention(dim=512,kernel_size=3)
    output=Model(input)
    print(output.shape)

    