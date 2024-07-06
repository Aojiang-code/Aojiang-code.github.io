import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"AirFormer: Predicting Nationwide Air Quality in China with Transformers"


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # 创建窗口大小的mask矩阵,屏蔽无因果性的连接, 应当是一个值为1的下三角矩阵（这意味着每个时间只与自己和自己之前的时间步有连接）
        self.mask = torch.tril(torch.ones(window_size, window_size)).to(device)  # mask for causality

    def forward(self, x):
        #(b*n, t, c)  B_prev=b*n; t=T_prev, c=C_prev
        B_prev, T_prev, C_prev = x.shape

        # 如果窗口尺寸大于0, 就在时间维度上划分多组窗口
        if self.window_size > 0:
            # 创建局部窗口: (b*n,t,c) --reshape-> (b*n*t/ws,ws,c)
            x = x.reshape(-1, self.window_size, C_prev)

        # B=b*n*t/ws, T=ws, C=c
        B, T, C = x.shape

        # 通过x生成qkv: (B,T,C) --qkv-> (B,T,3C) --reshape-> (B,T,3,h,d) --permute-> (3,B,h,T,d)   C=h*d; h:注意力头个数; d:每个头的通道数
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)

        # 将qkv划分为q/k/v: q:(B,h,T,d), k:(B,h,T,d), v:(B,h,T,d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算窗口内的注意力矩阵: (B,h,T,d) @ (B,h,d,T) = (B,h,T,T)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加因果关系: t时间步的状态受之前时间步的影响,而不受t之后时间步的影响,所以应对建立mask
        if self.causal:
            # 把attention矩阵的上三角部分设置为负无穷,这样的话在进行softmax归一化后值趋近于0, 使softmax仅仅在有值的地方进行
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        # 通过注意力矩阵对窗口内的时间步进行加权: (B,h,T,T) @ (B,h,T,d) = (B,h,T,d) --transpose->(B,T,h,d) -->reshape-->(B,T,C)
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)  #添加全连接层: (B,T,C)-->(B,T,C)
        x = self.proj_drop(x)

        # 如果窗口尺寸大于0,那么计算完之后应当恢复与输入相同的shape
        if self.window_size > 0:
            # 计算完窗口注意力之后, 应当恢复与输入相同的shape: (B,T,C)--reshape->(b*n, t, c)
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


# Pre Normalization in Transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# FFN in Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 dim,  # hidden dim
                 depth,  # the number of MSA in CT-MSA
                 heads,  # the number of heads
                 window_size,  # the size of local window
                 mlp_dim,  # mlp dimension
                 num_time,  # the number of time slot
                 dropout=0.,  # dropout rate
                 device=None):  # device, e.g., cuda
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        # 设置1层temporal attention即可
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim,
                                  heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: (b, c, n, t)
        b, c, n, t = x.shape
        # 对输入x进行reshape,转换成便于计算的shape: (b, c, n, t) --> (b,n,t,c) --> (b*n,t,c)
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)
        # 为输入x添加位置编码: (b*n,t,c) + (1,t,c) = (b*n,t,c)
        x = x + self.pos_embedding

        # 执行注意力机制 和 前馈神经网络层
        for attn, ff in self.layers:
            x = attn(x) + x   # 执行注意力机制并添加残差连接: (b*n,t,c) + (b*n,t,c) = (b*n,t,c)
            x = ff(x) + x   # 执行前馈神经网络并添加残差连接: (b*n,t,c) + (b*n,t,c) = (b*n,t,c)
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2) #  (b*n,t,c)--reshape-->(b,n,t,c)--permute-->(b,c,n,t)
        return x


if __name__ == '__main__':
    blocks = 3
    seq_len = 8
    t_modules = nn.ModuleList()
    for b in range(blocks):
        # 计算不同的block中的窗口尺寸, 以seq_len=24, blocks=4为例: --> 1th: b=0, w_s=24/8=3;   2th:  b=1, w_s=24/4=6;   3th:  b=2, w_s=24/2=12;   4th:  b=3, w_s=24/1=24
        # 计算不同的block中的窗口尺寸, 以seq_len=8, blocks=3为例: --> 1th: b=0, w_s=8/4=2;   2th:  b=1, w_s=8/2=4;   3th:  b=2, w_s=8/1=8;
        window_size = seq_len // 2 ** (blocks - b - 1)
        t_modules.append(CT_MSA(dim=32,
                                depth=1,
                                heads=8,
                                window_size=window_size,
                                mlp_dim=32 * 2,
                                num_time=seq_len, device=device))

    # (b, c, n, t)  n:序列的个数, t:序列长度,等于seq_len
    x = torch.randn(1, 32, 1, 8)
    for i in range(blocks):
        x = t_modules[i](x)  # (b,c,n,t)-->(b,c,n,t)

    print(x.shape)