import math
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class MSTF(nn.Module):
    def __init__(self, in_channels):
        super(MSTF, self).__init__()
        out_channels = in_channels

        self.project1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), )

        self.project2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)


    def forward(self, x0,x1,x2,x3):
        # (B,T,N,C)
        B,T,N,C = x0.shape


        x_ = torch.cat([x0,x1,x2,x3],0) # 将多个尺度的输入进行拼接: (B,T,N,C)--concat--> (M,B,T,N,C)

        x__ = x_.reshape(-1,T,N*B,C) # 对其进行reshape,以便后续计算: (M,B,T,N,C)--reshape-->(M,T,N*B,C)
        x__ = self.project1(x__.permute(0,3,2,1)).permute(0,3,2,1) # 通过一个线性层学习通道之间的相关性: (M,T,N*B,C)--permute-->(M,C,N*B,T)--project1-->(M,C,N*B,T)--permute-->(M,T,N*B,C)

        weight = F.softmax(x__, dim=0) # 在M维度上执行softmax,得到每个尺度的权重:(M,T,N*B,C)

        # 加权和
        x_ = x_.reshape(-1,T,N*B,C)  # 将输入重塑为与weight相同的shape: (M,B,T,N,C)-->(M,T,N*B,C)
        out = (weight * x_).sum(0)  # 每个尺度的权重与对应的输入相乘, 然后将多个尺度的输出相加: (M,T,N*B,C) * (M,T,N*B,C)=(M,T,N*B,C); (M,T,N*B,C)--sum-->(T,N*B,C)

        out = out.reshape(B,T,N,C) # (T,N*B,C)-->(B,T,N,C)

        return self.project2(out.permute(0,3,2,1)).permute(0,3,2,1)


if __name__ == '__main__':
    # (B,T,N,C)
    x0 = torch.rand(5, 36, 1, 64)
    x1 = torch.rand(5, 36, 1, 64)
    x2 = torch.rand(5, 36, 1, 64)
    x3 = torch.rand(5, 36, 1, 64)
    Model = MSTF(in_channels=64)
    out = Model(x0,x1,x2,x3)
    print(out.shape)