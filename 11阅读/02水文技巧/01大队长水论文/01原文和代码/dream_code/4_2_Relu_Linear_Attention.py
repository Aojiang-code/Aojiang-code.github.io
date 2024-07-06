import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


"EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction"


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=1E-6)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))




class LiteMLA(torch.nn.Module):
    """
    Lightweight Multiscale Linear Attention
    """

    def __init__(self,in_channels,out_channels,heads=None,dim=8,scales=(5,),eps=1.0e-15):
        super().__init__()
        self.eps = eps
        # 注意力头的个数; 如果没有设置每个注意力头的个数,按照每个头的维度为8,来求注意力头的个数
        heads = heads or int(in_channels // dim)

        # total_dim == in_channels
        total_dim = heads * dim

        self.dim = dim
        self.qkv = torch.nn.Conv2d(in_channels, 3 * total_dim, kernel_size=1, bias=False)
        self.aggreg = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(3 * total_dim, 3 * total_dim,
                                                                               kernel_size=scale,
                                                                               padding=scale // 2,
                                                                               groups=3 * total_dim, bias=False),
                                                               torch.nn.Conv2d(3 * total_dim, 3 * total_dim,
                                                                               kernel_size=1,
                                                                               groups=3 * heads, bias=False))
                                           for scale in scales])
        self.kernel_func = torch.nn.ReLU(inplace=False)

        self.proj = Conv(total_dim * (1 + len(scales)), out_channels, torch.nn.Identity())

    @torch.cuda.amp.autocast(enabled=False)
    def relu_linear_att(self, qkv):
        # (B,N*3C,H,W);  如果是把多尺度信息拼接起来,然后进入线性注意力的话,shape:(B,N*3C,H,W); 如果是每个尺度单独进入线性注意力的话,shape:(B,3C,H,W); 下面的注释是以第一种情况为例写的
        B, C, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        # 将qkv变换维度,以便进行计算: (B,N*3C,H,W) --> (B,N*h,3*d,HW);  C=h*d
        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W,), )
        # 将qkv变换维度,以便进行计算: (B,N*h,3*d,HW) --> (B,N*h,HW,3*d)
        qkv = torch.transpose(qkv, -1, -2)
        # 将qkv分别拆分为: q:(B,N*h,HW,d)  k:(B,N*h,HW,d)  v:(B,N*h,HW,d)
        q, k, v = (qkv[..., 0: self.dim], qkv[..., self.dim: 2 * self.dim], qkv[..., 2 * self.dim:],)

        # lightweight linear attention
        q = self.kernel_func(q) # query通过relu激活函数
        k = self.kernel_func(k) # key通过relu激活函数

        v = F.pad(v, (0, 1), mode="constant", value=1)  # 填充:(B,N*h,HW,d) -->(B,N*h,HW,d+1);  填充操作:(0,1) = (左边填充数, 右边填充数)
        kv = torch.matmul(k.transpose(-1, -2), v)  # 线性注意力第一步,先计算kv: (B,N*h,d,HW) @ (B,N*h,HW,d+1) = (B,N*h,d,d+1)
        out = torch.matmul(q, kv)  # 线性注意力第二步,再计算q(kv): (B,N*h,HW,d) @ (B,N*h,d,d+1) = (B,N*h,HW,d+1)
        out = out[..., :-1] / (out[..., -1:] + self.eps)  # eps是为了防止分母为0,除以最后一个通道应该是起到一个标准化的作用: (B,N*h,HW,d)


        out = torch.transpose(out, -1, -2)  # 重塑维度: (B,N*h,HW,d) --> (B,N*h,d,HW)
        out = torch.reshape(out, (B, -1, H, W))   # 重塑维度: (B,N*h,d,HW) --> (B,N*C,H,W)
        return out

    def forward(self, x):
        # x:(B, C, H, W)

        # generate multi-scale q, k, v
        qkv = self.qkv(x)  # 将输入x通过线性层生成qkv: (B,C,H,W) --> (B,3C,H,W)
        multi_scale_qkv = [qkv] # 将qkv放入列表

        # 执行多个尺度的卷积操作,从而生成多尺度的qkv表示, 然后将每一个尺度的输出放入到列表内
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))

        # 这里是将多尺度信息拼接之后输入到线性注意力机制, 也可以将多个尺度信息分别输入进去
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)  # 将多个尺度的信息进行拼接 (B,3C,H,W)-->(B,N*3C,H,W);  N是尺度的个数(包括自身)
        out = self.relu_linear_att(multi_scale_qkv) # 拼接后的信息输入到线性注意力, 得到输出: (B,N*3C,H,W)-->(B,N*C,H,W)


        #对每个尺度信息逐个执行线性注意力  (打开102-104行的时候,记得注释96-97行)
        # out_list = []
        # for i in range(len(multi_scale_qkv)):
        #     out_list.append(self.relu_linear_att(multi_scale_qkv[i]))  # (B,3C,H,W)-->(B,C,H,W)  每个尺度输出:(B,C,H,W)
        # out = torch.cat(out_list, dim=1)  # (B,C,H,W) --> (B,N*C,H,W)

        # 将通道映射到与输入相同的通道数,并添加残差: (B,N*C,H,W) --> (B,C,H,W)
        return x + self.proj(out)


if __name__ == '__main__':
    # (B, C, H, W)
    X = torch.randn(1, 64, 7,7)
    # scales: 尺度的设置 单个尺度:(5,); 多个尺度:(3,5)
    Model = LiteMLA(in_channels=64, out_channels=64, scales=(5,))
    out = Model(X)
    print(out.shape)
