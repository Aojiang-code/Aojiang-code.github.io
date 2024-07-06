import numpy as np
import torch
from torch import nn
from torch.nn import init

"CBAM: Convolutional Block Attention Module "

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):

        max_result=self.maxpool(x) # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        avg_result=self.avgpool(x) # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        max_out=self.se(max_result) # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        avg_out=self.se(avg_result) # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        output=self.sigmoid(max_out+avg_out) # 相加,然后通过sigmoid获得权重:(B,C,1,1)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        # x:(B,C,H,W)
        max_result,_=torch.max(x,dim=1,keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result=torch.mean(x,dim=1,keepdim=True)   # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result=torch.cat([max_result,avg_result],1)   # 在通道上拼接两个矩阵:(B,2,H,W)
        output=self.conv(result)                      # 然后重新降维为1维:(B,1,H,W)
        output=self.sigmoid(output)                   # 通过sigmoid获得权重:(B,1,H,W)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ChannelAttention=ChannelAttention(channel=channel,reduction=reduction)
        self.SpatialAttention=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B,C,H,W = x.size()
        residual=x
        out=x * self.ChannelAttention(x)     # 将输入与通道注意力权重相乘: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)
        out=out * self.SpatialAttention(out) # 将更新后的输入与空间注意力权重相乘:(B,C,H,W) * (B,1,H,W) = (B,C,H,W)
        return out+residual


if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(1,512,7,7)
    Model = CBAMBlock(channel=512,reduction=16,kernel_size=7)
    output=Model(input)
    print(output.shape)

    