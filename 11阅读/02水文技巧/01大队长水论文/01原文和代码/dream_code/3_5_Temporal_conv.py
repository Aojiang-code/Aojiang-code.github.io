import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor,seq_len):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.padding=0
        self.seq_len = seq_len
        self.kernel_set = [2,3,6,7]
        # 将通道平均分为N组. N是卷积层的个数
        cout = int(cout/len(self.kernel_set))
        # k个1D因果膨胀卷积
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

        # 如果dilation_factor=1, out=input-k+1
        # 如果dilation_factor>=1, out=[input-d*(k-1)+2*p-1]/stride+1  d:dilation_factor; p:padding  计算公式见pytorch官网：https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
        # 这里是两层全连接,先映射到高维,再映射回与输入相同的时间维数,增加表达能力
        self.out = nn.Sequential(
            nn.Linear(self.seq_len-dilation_factor*(self.kernel_set[-1]-1)+self.padding*2-1+1, cin),
            nn.ReLU(),
            nn.Linear(cin, self.seq_len)
        )
    def forward(self,input):
        # input: (B, C, N, T)

        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input)) # 执行k==4个1D因果膨胀卷积操作: 1th:(B,C,N,T)-->(B,C/4,N,T1); 2th:(B,C,N,T)-->(B,C/4,N,T2); 3th:(B,C,N,T)-->(B,C/4,N,T3); 4th:(B,C,N,T)-->(B,C/4,N,T4);
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]  #以时间维度最少的特征为标准, 截取其他特征的时间维数以保持一致, 由于卷积核逐渐增大,因此T4最小: x=[(B,C/4,N,T4),(B,C/4,N,T4),(B,C/4,N,T4),(B,C/4,N,T4)]

        x = torch.cat(x,dim=1) # 将k==4个1D因果膨胀卷积层的输出在通道上进行拼接: (B,C,N,T4)
        x = self.out(x) # 两层线性层映射,在时间维度上先升维,后降维: (B,C,N,T4)-->(B,C,N,C)-->(B,C,N,T)
        return x


class temporal_conv(nn.Module):
    def __init__(self, cin, cout, dilation_factor,seq_len):
        super(temporal_conv, self).__init__()

        self.filter_convs = dilated_inception(cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len)
        self.gated_convs = dilated_inception(cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len)

    def forward(self, X):
        # X:(B,C,N,T)
        filter = self.filter_convs(X)  # 执行左边的DIL层: (B,C,N,T)-->(B,C,N,T)
        filter = torch.tanh(filter)  # 左边的DIL层后接一个tanh激活函数,生成输出:(B,C,N,T)-->(B,C,N,T)
        gate = self.gated_convs(X) # 执行右边的DIL层: (B,C,N,T)-->(B,C,N,T)
        gate = torch.sigmoid(gate) # 右边的DIL层后接一个sigmoid门控函数,生成权重表示:(B,C,N,T)-->(B,C,N,T)
        out = filter * gate  # 执行逐元素乘法: (B,C,N,T) * (B,C,N,T) = (B,C,N,T)
        return out



if __name__ == '__main__':
    # (B,C,N,T)  N:序列的个数  T:序列的长度
    X = torch.randn(1, 32, 1, 24)
    Model = temporal_conv(cin=32, cout=32, dilation_factor=1,seq_len=24)
    out = Model(X)
    print(out.shape)