import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import einops

"SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications"


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=512):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim)
        self.to_key = nn.Linear(in_dims, token_dim)

        self.w_a = nn.Parameter(torch.randn(token_dim, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim, token_dim)
        self.final = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        B,N,D = x.shape

        # 生成初步的query、key矩阵
        query = self.to_query(x)  # 通过变换生成query矩阵: (B,N,D) --> (B,N,C)
        key = self.to_key(x)   # 通过变换生成key矩阵: (B,N,D) --> (B,N,C)

        #进行标准化
        query = torch.nn.functional.normalize(query, dim=-1) # 对query进行标准化: (B,N,C)
        key = torch.nn.functional.normalize(key, dim=-1) # 对key进行标准化: (B,N,C)

        # 学习query的注意力权重
        query_weight = query @ self.w_a   # 将query矩阵乘上一个可学习的向量,得到注意力权重:(B,N,C) @ (C,1) = (B,N,1)
        A = query_weight * self.scale_factor # 对注意力权重缩放: (B,N,1)
        A = torch.nn.functional.normalize(A, dim=1) # 对注意力权重标准化:(B,N,1)

        # 通过注意力权重加权
        q = torch.sum(A * query, dim=1) # 通过注意力权重对query进行加权,生成全局query向量:(B,N,1)*(B,N,C)=(B,N,C)--sum-->(B,C)
        #q = einops.repeat(q, "b d -> b repeat d", repeat=key.shape[1]) # BxNxD
        q = q.reshape(B, 1, -1) # 将全局query向量恢复三维: (B,C)--> (B,1,C)

        out = self.Proj(q * key) + query # 通过使用逐元素乘积对全局query向量和key矩阵之间的交互进行编码,生成全局上下文表示: (B,1,C) * (B,N,C) = (B,N,C)
        out = self.final(out) #对全局上下文通过线性层,学习token的隐藏表示,并生成输出: (B,N,C)-->(B,N,C)

        return out

if __name__ == '__main__':
    # (B,N,D)  N:token的数量
    X = torch.randn(1, 50, 512)
    Model = EfficientAdditiveAttnetion(in_dims=512, token_dim=512)
    out = Model(X)
    print(out.shape)