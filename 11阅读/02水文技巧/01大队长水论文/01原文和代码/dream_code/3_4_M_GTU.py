import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting"


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        # 通过1D卷积层提取局部时间依赖,并将通道映射到2C: (B,C,N,T) --> (B,2C,N,T-K+1)
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]        # 前C个通道,后接tanh函数: (B,C,N,T-K+1)
        x_q = x_causal_conv[:, -self.in_channels:, :, :]        # 后C个通道,后接sogmoid门控函数: (B,C,N,T-K+1)
        x_gtu = self.tanh(x_p) * self.sigmoid(x_q)    # 两部分执行逐点积相乘: (B,C,N,T-K+1)
        return x_gtu


class Multi_GTU(nn.Module):
    def __init__(self, num_of_timesteps, in_channels, time_strides, kernel_size, pool=False):
        super(Multi_GTU, self).__init__()
        kernel_size_sum = sum(kernel_size)
        self.gtu0 = GTU(in_channels, time_strides, kernel_size[0])
        self.gtu1 = GTU(in_channels, time_strides, kernel_size[1])
        self.gtu2 = GTU(in_channels, time_strides, kernel_size[2])
        self.pool = pool
        if self.pool:
            self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0, return_indices=False, ceil_mode=False)
            self.fcmy = nn.Sequential(
                nn.Linear(int((3*num_of_timesteps-kernel_size_sum+3)/2), num_of_timesteps),
                nn.Dropout(0.05),
            )
        else:
            self.fcmy = nn.Sequential(
                nn.Linear(3 * num_of_timesteps - kernel_size_sum + 3, num_of_timesteps),
                nn.Dropout(0.05),
            )
    def forward(self, X):
        # x:(B,C,N,T)

        x_gtu = []                   # 计算公式 == 以卷积核为[3,5,7]的计算
        x_gtu.append(self.gtu0(X))  # 执行第一个尺度的1D因果卷积: (B,C,N,T-K1+1) == (B,C,N,T-2)
        x_gtu.append(self.gtu1(X))  # 执行第二个尺度的1D因果卷积: (B,C,N,T-K2+1) == (B,C,N,T-4)
        x_gtu.append(self.gtu2(X))  # 执行第三个尺度的1D因果卷积: (B,C,N,T-K3+1) == (B,C,N,T-6)
        time_conv = torch.cat(x_gtu, dim=-1)  # 拼接三个尺度的因果卷积层的输出: (B,C,N,3T+3-K1-K2-K3) = (B,C,N,3T-12)

        # 如果有池化层的话,且池化窗口==2, 那么时间步长要缩短一半,这意味着输入到池化层的时间长度应该能被(池化窗口==2)整除,这一步需要精心设置,以保证计算的正确性。 再多说一句,官方论文将池化放在了每一个GTU的后面,这里则是对Concat后的总长度执行池化,个人认为基本是一致的【主要是作者没写池化的代码，我当然想怎么写就怎么写了】
        if self.pool:
            time_conv = self.pooling(time_conv) # (B,C,N,(3T+3-K1-K2-K3)/2) = (B,F,N,(3T-12)/2)  T==12时,池化后: (B,C,N,12)
            time_conv = self.fcmy(time_conv)
        else:
            # 没有池化操作的情况下: 恢复为与输入相同的shape: (B,C,N,3T+3-K1-K2-K3) -->(B,C,N,T)
            time_conv = self.fcmy(time_conv)

        # 添加残差连接和 relu激活函数,得到最终输出
        time_conv_output = F.relu(X + time_conv)
        return time_conv_output


if __name__ == '__main__':
    # (B,C,N,T)  N:序列的个数  T:序列的长度  更改输入的时候记得把64行对应的时间步和输入通道数改整
    X = torch.randn(1, 64, 1, 24)
    Model = Multi_GTU(num_of_timesteps=24, in_channels=64, time_strides=1, kernel_size=[3,5,7], pool=True)
    out = Model(X)
    print(out.shape)
