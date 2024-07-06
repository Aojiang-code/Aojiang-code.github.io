import math
import torch.nn as nn
import torch
import itertools
import torch.nn.functional as F
# from mmcv.cnn.bricks import ConvModule

"TransXNet: Learning Both Global and Local Dynamics with a Dual Dynamic Token Mixer for Visual Recognition"


class OSRAttention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=8,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=2,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim,kernel_size=sr_ratio+3,stride=sr_ratio,padding=(sr_ratio+3)//2,groups=dim,bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                # ConvModule(dim, dim,
                #            kernel_size=sr_ratio+3,
                #            stride=sr_ratio,
                #            padding=(sr_ratio+3)//2,
                #            groups=dim,
                #            bias=False,
                #            norm_cfg=dict(type='BN2d'),
                #            act_cfg=dict(type='GELU')),
                # ConvModule(dim, dim,
                #            kernel_size=1,
                #            groups=dim,
                #            bias=False,
                #            norm_cfg=dict(type='BN2d'),
                #            act_cfg=None,),)
                nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),)
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)  # 对输入x进行变换得到q,然后通过reshape重塑shape: (B,C,H,W)--q()->(B,C,H,W)--reshape-->(B,h,d,HW) --transpose--> (B,h,HW,d);  C=h*d

        # 通过OSR操作得到k/v表示
        kv = self.sr(x) # 执行空间缩减(spatial reduction)操作,也就是通过卷积来实现下采样,得到kv: (B,C,H,W)-->(B,C,H‘,W’)
        kv = self.local_conv(kv) + kv  #通过3×3卷积对局部空间建模,并添加残差连接: (B,C,H‘,W’)-->(B,C,H‘,W’)
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1) #分割为k、v: (B,C,H‘,W’)--kv()-->(B,2C,H‘,W’)--chunk--> k:(B,C,H‘,W’); v:(B,C,H‘,W’)

        k = k.reshape(B, self.num_heads, C//self.num_heads, -1) # (B,C,H‘,W’) --reshape--> (B,h,d,H'W');  c=h*d
        v = v.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)  #(B,C,H‘,W’)--reshape-->(B,h,d,H'W')--transpose-->(B,h,H'W',d)

        attn = (q @ k) * self.scale # 对qk计算注意力矩阵: (B,h,HW,d) @ (B,h,d,H'W') = (B,h,HW,H'W')

        # 为注意力矩阵添加位置编码
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc

        attn = torch.softmax(attn, dim=-1) # 对注意力矩阵进行归一化
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2) # 通过注意力矩阵对value进行加权: (B,h,HW,H'W') @ (B,h,H'W',d) = (B,h,HW,d);  (B,h,HW,d)--transpose-->(B,h,d,HW)
        return x.reshape(B, C, H, W) # 对x进行reshape,重塑为与输入相同的shape: (B,h,HW,d) --> (B, C, H, W)


if __name__ == '__main__':
    # (B,C,H,W)
    x = torch.randn(1, 64, 7, 7)
    Model = OSRAttention(dim=64)
    out = Model(x)
    print(out.shape)