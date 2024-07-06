import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting"


class Trend_aware_attention(nn.Module):
    '''
    Trend_aware_attention  mechanism
    X:      [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, kernel_size):
        super(Trend_aware_attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_v = nn.Linear(D,D)
        self.FC = nn.Linear(D,D)
        self.kernel_size = kernel_size
        self.padding = self.kernel_size-1
        self.cnn_q = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.cnn_k = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.norm_q = nn.BatchNorm2d(D)
        self.norm_k = nn.BatchNorm2d(D)
    def forward(self, X):
        # (B,T,N,D)
        batch_size = X.shape[0]

        X_ = X.permute(0, 3, 2, 1) # 对x进行变换, 便于后续执行CNN: (B,T,N,D) --> (B,D,N,T)

        # key: (B,T,N,D)  value: (B,T,N,D)
        query = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1) # 通过1×k的因果卷积来生成query表示: (B,D,N,T)--permute-->(B,T,N,D)
        key = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1) # 通过1×k的因果卷积来生成key表示: (B,D,N,T)--permute-->(B,T,N,D)
        value = self.FC_v(X) # 通过简单的线性层生成value: (B,T,N,D)-->(B,T,N,D)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0) # 将query划分为多头: (B,T,N,D)-->(B*k,T,N,d);  D=k*d; h:注意力头的个数; d:每个注意力头的通道上
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  # 将key划分为多头: (B,T,N,D)-->(B*k,T,N,d);
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0) # 将value划分为多头: (B,T,N,D)-->(B*k,T,N,d);

        query = query.permute(0, 2, 1, 3)  # query: (B*k,T,N,d) --> (B*k,N,T,d)
        key = key.permute(0, 2, 3, 1)      # key: (B*k,T,N,d) --> (B*k,N,d,T)
        value = value.permute(0, 2, 1, 3)  # key: (B*k,T,N,d) --> (B*k,N,T,d)

        attention = (query @ key) * (self.d ** -0.5) # 以上下文的方式计算注意力矩阵: (B*k,N,T,d) @ (B*k,N,d,T) = (B*k,N,T,T)
        attention = F.softmax(attention, dim=-1) # 通过softmax进行归一化

        X = (attention @ value) # 通过上下文化的注意力矩阵对value进行加权: (B*k,N,T,T) @ (B*k,N,T,d) = (B*k,N,T,d)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1) # (B*k,N,T,d)-->(B,N,T,d*k)==(B,N,T,D)
        X = self.FC(X) # 融合多个子空间的特征进行输出
        return X.permute(0, 2, 1, 3) # (B,N,T,D)-->(B,T,N,D)




if __name__ == '__main__':

    # k: attention head, d: dimension of each head
    Model = Trend_aware_attention(K=8, d=8, kernel_size=3)

    # (B,T,N,D)  N:序列的个数
    X = torch.randn(1,12,1,64)
    out = Model(X) # (B,T,N,D)-->(B,T,N,D)
    print(out.shape)