from einops import rearrange
import numbers
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

'''
   论文题目：使用高低频分解进行图像修复和增强     CVPR 2024顶会
   
本文中的高低频分解块 (HLFD) 是一种为了解决图像不同退化特征的模块，
主要基于高频信息和低频信息在处理图像退化方面的不同作用。
本文提出的 HLFD 模块通过分离图像特征中的高频和低频信息，
并根据它们的特性分别处理这些信息，从而更好地恢复图像的细节和结构。

高低频分解块原理：
1.高频信息： 通常反映图像中的局部细节（例如边缘、纹理等），在图像去模糊、降噪等任务中非常重要。
因此，HLFD 模块通过多个卷积块提取这些高频特征，并通过密集连接机制增强高频信息的细节。

2.低频信息： 通常代表图像的整体结构，主要用于恢复图像的大尺度信息，比如图像的轮廓和背景。
在这方面低频信息起着关键作用。

3.HLFDB 模块通过多级通道自注意力机制：作用--学习长距离的依赖关系，捕捉图像的全局上下文信息。

高低频分解块作用：
HLFDB 模块的主要作用在于它能够根据图像退化的不同特性，分别处理高频和低频信息，
从而针对性地进行图像恢复和增强。例如，图像去模糊和降噪任务需要增强高频细节，超分辨率图像任务需要更多地关注低频信息。
通过分解高频和低频信息，HLFDB 模块有效地提升了图像的细节恢复能力，并且减少了结构信息在下采样过程中丢失的可能性

适用于：图像恢复，图像增强，图像去噪，暗光增强，超分辨率图像，目标检测，图像分割等所有CV2维任务通用涨点模块
'''

'''
这段代码实现了一个高低频分解块（HLFD），用于图像的修复和增强。通过离散小波变换（DWT）和逆离散小波变换（IDWT），将图像分解为高频和低频信息，并分别进行处理。高频信息通过密集卷积网络（Dense）增强，低频信息通过U-Net网络处理，最后通过融合模块（Fusion）将处理后的高频和低频信息合并，生成增强后的图像。
'''



'''
什么是对输入张量进行低频低频卷积？

在信号处理和图像处理领域，小波变换是一种常用的技术，用于将信号分解为不同的频率分量。低频低频卷积（LL卷积）是小波变换中的一个步骤，通常用于提取信号或图像中的低频信息。

具体来说：

1. **低频信息**：在图像中，低频信息通常代表图像的整体结构和大范围的颜色变化，而不是细节和边缘。低频信息通常是图像的平滑部分。

2. **低频低频卷积（LL卷积）**：在小波变换中，LL卷积是指对图像进行两次低通滤波（一次在水平方向，一次在垂直方向），以提取图像的低频信息。这个过程可以看作是对图像进行平滑处理，去除高频噪声和细节。

在代码中，`torch.nn.functional.conv2d` 函数用于实现卷积操作。`w_ll` 是用于低频低频卷积的滤波器权重。通过对输入张量 `x` 进行卷积操作，提取出低频信息，结果存储在 `x_ll` 中。

这种操作在图像处理、压缩和特征提取中非常有用，因为它可以有效地减少数据量，同时保留图像的主要特征。
'''

# 定义一个自定义的离散小波变换函数
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 确保输入张量在内存中是连续的
        x = x.contiguous()
        # 保存小波滤波器的权重以备反向传播使用
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        # 保存输入张量的形状信息
        ctx.shape = x.shape

        # 获取输入张量的通道数
        dim = x.shape[1]
        # 对输入张量进行低频低频卷积
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 对输入张量进行低频高频卷积
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 对输入张量进行高频低频卷积
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 对输入张量进行高频高频卷积
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将四个卷积结果在通道维度上拼接
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回拼接后的结果
        return x

    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要对输入进行梯度计算
        if ctx.needs_input_grad[0]:
            # 从上下文中恢复保存的张量（小波滤波器的权重）
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            # 获取输入张量的形状信息
            B, C, H, W = ctx.shape
            # 将梯度张量重新调整形状为 (B, 4, -1, H//2, W//2)
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            # 交换维度并重新调整形状为 (B, -1, H//2, W//2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            # 将四个滤波器在通道维度上拼接，并重复 C 次以匹配输入通道数
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 使用转置卷积计算输入的梯度
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        # 返回输入的梯度和其他参数的梯度（这里为 None，因为它们不需要梯度）
        return dx, None, None, None, None

# 定义一个自定义的逆离散小波变换函数
class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        # 保存滤波器以备反向传播使用
        ctx.save_for_backward(filters)
        # 保存输入张量的形状信息
        ctx.shape = x.shape

        # 获取输入张量的批次大小和空间维度
        B, _, H, W = x.shape
        # 将输入张量调整为 (B, 4, -1, H, W) 并交换维度
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # 获取调整后张量的通道数
        C = x.shape[1]
        # 将张量重新调整为 (B, -1, H, W)
        x = x.reshape(B, -1, H, W)
        # 将滤波器在通道维度上重复 C 次以匹配输入通道数
        filters = filters.repeat(C, 1, 1, 1)
        # 使用转置卷积进行逆离散小波变换
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        # 返回重建后的张量
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

# 定义一个逆离散小波变换的2D模块
class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

# 定义一个离散小波变换的2D模块
class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# 定义一个离散小波变换的2D模块，使用float32精度
class DWT_2Dfp32(nn.Module):
    def __init__(self, wave):
        super(DWT_2Dfp32, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# 定义一个通道注意力层
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # 全局平均池化：特征 --> 点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 特征通道降维和升维 --> 通道权重
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# 定义一个密集卷积网络
class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        # self.norm = nn.LayerNorm([in_channels, 128, 128])  # 假设输入大小为 [224, 224]
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.gelu(x1 + x)

        x2 = self.conv2(x1)
        x2 = self.gelu(x2 + x1 + x)

        x3 = self.conv3(x2)
        x3 = self.gelu(x3 + x2 + x1 + x)

        x4 = self.conv4(x3)
        x4 = self.gelu(x4 + x3 + x2 + x1 + x)

        x5 = self.conv5(x4)
        x5 = self.gelu(x5 + x4 + x3 + x2 + x1 + x)

        x6 = self.conv6(x5)
        x6 = self.gelu(x6 + x5 + x4 + x3 + x2 + x1 + x)
        return x6

# 定义一个残差网络
class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 += x  # 残差连接
        return out2

# 定义一个融合模块
class Fusion(nn.Module):
    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        self.dwt = DWT_2D(wave)
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.high = ResNet(in_channels)
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.low = ResNet(in_channels)
        self.idwt = IDWT_2D(wave)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x_dwt = self.dwt(x1)
        ll, lh, hl, hh = x_dwt.split(c, 1)
        high = torch.cat([lh, hl, hh], 1)
        high1 = self.convh1(high)
        high2 = self.high(high1)
        highf = self.convh2(high2)
        b1, c1, h1, w1 = ll.shape
        b2, c2, h2, w2 = x2.shape

        #
        if (h1 != h2):
            x2 = F.pad(x2, (0, 0, 1, 0), "constant", 0)
        low = torch.cat([ll, x2], 1)
        low = self.convl(low)
        lowf = self.low(low)

        out = torch.cat((lowf, highf), 1)
        out_idwt = self.idwt(out)

        return out_idwt

# 定义一个U-Net网络
class UNet(nn.Module):
    def __init__(self, in_channels, wave):
        super(UNet, self).__init__()
        # 定义层
        self.trans1 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        self.trans2 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        self.trans3 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.upsample1 = Fusion(in_channels, wave)
        self.upsample2 = Fusion(in_channels, wave)

    def forward(self, x):
        x1 = x
        # print(x1.shape)
        x1_r = self.trans1(x)
        x2 = self.avgpool1(x1)
        # print(x2.shape)
        x2_r = self.trans2(x2)
        x3 = self.avgpool2(x2)
        # print(x3.shape)
        x3_r = self.trans3(x3)
        x4 = self.upsample1(x2_r, x3_r)
        out = self.upsample2(x1_r, x4)
        b1, c1, h1, w1 = out.shape
        b2, c2, h2, w2 = x.shape

        if (h1 != h2):
            out = F.pad(out, (0, 0, 1, 0), "constant", 0)

        return out + x

# 定义一个将张量转换为3D的函数
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

# 定义一个将张量转换为4D的函数
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# 定义一个无偏置的层归一化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# 定义一个层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# 定义一个带偏置的层归一化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# 定义一个前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# 定义一个注意力模块
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# 定义一个Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

# 定义一个高低频分解块
class HLFD(nn.Module):
    def __init__(self, dim, wave='haar'):
        super(HLFD, self).__init__()
        n_feats = dim
        self.down = nn.AvgPool2d(kernel_size=2)
        self.dense = Dense(n_feats)
        self.unet = UNet(n_feats, wave)

        self.alise1 = nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # one_module(n_feats)

        self.att = CALayer(n_feats)

    def forward(self, x):
        low = self.down(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)

        lowf = self.unet(low)
        highfeat = self.dense(high)
        lowfeat = F.interpolate(lowf, size=x.size()[-2:], mode='bilinear', align_corners=True)

        out = self.alise2(self.att(self.alise1(torch.cat([highfeat, lowfeat], dim=1)))) + x

        return out

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 实例化模型对象
    model = HLFD(dim=32)
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 执行前向传播
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())