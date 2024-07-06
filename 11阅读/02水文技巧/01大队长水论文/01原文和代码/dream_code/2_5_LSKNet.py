import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

"Large Selective Kernel Network for Remote Sensing Object Detection"


class LSKmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3) # 应用padding使输入输出shape保持一致
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # x: (B,C,H,W)

        # (k:卷积核尺寸,d:膨胀率)  将(23,1)分解为: (5,1) and (7,3)
        attn1 = self.conv0(x)     # 应用第一个卷积层: (B,C,H,W)--> (B,C,H,W)
        attn2 = self.convl(attn1) # 应用第二个卷积层: (B,C,H,W)--> (B,C,H,W)

        attn1 = self.conv0_s(attn1)  # 应用1×1Conv建模通道间相关性,并将通道降维到C/2: (B,C,H,W)--> (B,C/2,H,W)   注意:如果分解为了三个卷积层,那就将通道C降维到C/3, 以便于在Concat的时候能恢复到原通道数量
        attn2 = self.conv1_s(attn2)  # 应用1×1Conv建模通道间相关性,并将通道降维到C/2: (B,C,H,W)--> (B,C/2,H,W)   # 原文中并没有提出需要降维,应该是为了提高计算效率选择降维了

        attn = torch.cat([attn1, attn2], dim=1)  # 将多个不同尺度的特征图在通道上进行拼接,恢复原通道数量: (B,C,H,W)
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # 应用全局平均池化: (B,C,H,W)-->(B,1,H,W)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # 应用全局最大池化: (B,C,H,W)-->(B,1,H,W)
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 将平均池化和最大池化特征进行拼接: (B,2,H,W)
        sig = self.conv_squeeze(agg).sigmoid()  # 将2个通道映射为N个通道, N是尺度的个数, 并通过sigmoid函数得到每个尺度对应的权重表示: (B,N,H,W), 在这里N==2
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1) # 对多个尺度的信息进行加权求和: (B,C/2,H,W)
        attn = self.conv_m(attn) # 将通道恢复为原通道数量: (B,C/2,H,W)-->(B,C,H,W)
        return x * attn  # 最后与输入特征执行逐元素乘法


if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(1,512,7,7)
    Model = LSKmodule(dim=512)
    output=Model(input)
    print(output.shape)