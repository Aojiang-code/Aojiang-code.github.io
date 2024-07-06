import torch
import torch.nn as nn
import torch.nn.functional as F

"Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution"

# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        # 表示有多少个尺度
        self.n_levels = n_levels
        # 每个尺度的通道是多少
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        # (B,C,h,w)
        h, w = x.size()[-2:]

        # 将通道平均分为n_levels份,n_levels是尺度的个数: (B,C,h,w) --chunk--> (B,C/n_levels,h,w)
        xc = x.chunk(self.n_levels, dim=1)

        out = []
        # 遍历多个尺度,四个尺度的下采样比例是[1,2,4,8],第一个尺度保持原有分辨率,因此从第二个尺度开始遍历
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h// 2**i, w//2**i)  # 1th: p_size=(h/2,w/2); 2th: p_size=(h/4,w/4); 3th: p_size=(h/8,w/8)
                s = F.adaptive_max_pool2d(xc[i], p_size)  # 以1th为例, 执行最大池化: (B,C/n_levels,h,w) --> (B,C/n_levels,h/2,w/2)
                s = self.mfr[i](s) # 执行3×3的深度卷积: (B,C/n_levels,h/2,w/2) --> (B,C/n_levels,h/2,w/2)
                s = F.interpolate(s, size=(h, w), mode='nearest') #通上采样恢复与输入相同的shape:(B,C/n_levels,h/2,w/2) --> (B,C/n_levels,h,w)
            else:
                s = self.mfr[i](xc[i]) # 0th: 第一个尺度保持原有分辨率(h,w), 然后执行3×3的深度卷积:  (B,C/n_levels,h,w)--> (B,C/n_levels,h,w)
            out.append(s)

        # 将四个尺度的输出在通道上拼接,恢复原shape: (B,C,h,w), 然后通过1×1Conv来聚合多个子空间的不同尺度的通道特征:
        out = self.aggr(torch.cat(out, dim=1))
        # 通过gelu激活函数进行规范化,来得到注意力图,然后与原始输入执行逐元素乘法（空间上的多尺度池化会造成空间上的信息丢失，通过与原始输入相乘能够保留一些空间上的细节）, 得到最终输出
        out = self.act(out) * x
        return out

if __name__ == '__main__':
    # (B,C,H,W)
    x = torch.randn(1, 36, 224, 224)
    Model = SAFM(dim=36)
    out = Model(x)
    print(out.shape)