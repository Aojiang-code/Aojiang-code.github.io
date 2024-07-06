import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"P2T: Pyramid Pooling Transformer for Scene Understanding"


class P2TAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1 ,2 ,3 ,6]):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([ t *t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios  # 池化窗口大小
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)
        self.d_convs1 = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in self.pool_ratios])

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape

        # 通过输入x生成q矩阵: (B,N,C) --q-> (B,N,C) --reshape-> (B,N,h,d) --permute-> (B,h,N,d);   C=h*d
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pools = []

        # 为了便于在x上执行多尺度池化操作,我们将其reshape重塑为2D类型: (B,N,C) --permute-> (B,C,N) --reshape-> (B,C,H,W)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # 遍历多个池化层, 假设池化窗口为: [1 ,2 ,3 ,6]
        for (pool_ratio, l) in zip(self.pool_ratios, self.d_convs1):
            # 分别计算当前池化窗口下的输出: input:(B,C,H,W);  1th_pool: (B,C,H/1,W/1); 2th_pool: (B,C,H/2,W/2); 3th_pool: (B,C,H/3,W/3); 4th_pool: (B,C,H/6,W/6)
            pool = F.adaptive_avg_pool2d(x_, (round( H /pool_ratio), round( W /pool_ratio)))
            # 将每一个尺度对应的池化层的输出, 再通过3*3的深度卷积进行相对位置编码, 然后与池化的输出相加
            pool = pool + l(pool)
            # 将每个尺度的输出重塑为与原始输入相同的shape: 1th_pool: (B,C,H/1,W/1) -->(B,C,(HW/1^2));  2th_pool: (B,C,H/2,W/2) --> (B,C,(HW/2^2));  3th_pool: (B,C,H/3,W/3) --> (B,C,(HW/3^2));   3th_pool: (B,C,H/6,W/6) --> (B,C,(HW/6^2));
            pools.append(pool.view(B, C, -1))

        # 将多个尺度池化层的输出在token维度进行拼接,其具有多尺度的上下文信息: (B,C,(HW/1^2)+(HW/2^2)+(HW/3^2)+(HW/6^2))==(B,C,token_num) , 令token_num = (HW/1^2)+(HW/2^2)+(HW/3^2)+(HW/6^2)
        pools = torch.cat(pools, dim=2)
        # 将其进行维度转换, 以便于后续计算: (B,C,token_num)--permute->(B,token_num,C)
        pools = self.norm(pools.permute(0 ,2 ,1))

        # 多尺度的上下文信息生成kv: (B,token_num,C) --kv-> (B,token_num,2C) --reshape-> (B,token_num,2,h,d) --permute-> (2,B,h,token_num,d);   C=h*d
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k:(B,h,token_num,d); v:(B,h,token_num,d)
        k, v = kv[0], kv[1]

        # 计算Token-to-Region化的注意力矩阵(region是指池化是在窗口上进行的,窗口可以看作region): (B,h,N,d) @ (B,h,d,token_num) = (B,h,N,token_num)  N:输入的token总数, token_num:池化后的Token总数量
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对value加权求和: (B,h,N,token_num) @ (B,h,token_num,d) = (B,h,N,d)
        x = (attn @ v)

        # 通过对输入进行重塑shape得到与原始输入相同的shape: (B,h,N,d) --transpose-> (B,N,h,d) --reshape-> (B,N,C)
        x = x.transpose(1 ,2).contiguous().reshape(B, N, C)
        # 最后通过一个线性层进行映射, 得到最终输出: (B,N,C)-->(B,N,C)
        x = self.proj(x)

        return x


class PatchEmbed(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=7, in_chans=3, embed_dim=64, overlap=True):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                                  padding=kernel_size // 2)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (B, C’, H', W') --> (B, C, H, W)    H', W'是图像的高度和宽度, 划分完patch之后,图像每列有H个patch，每行有W个patch
        x = self.proj(x)
        B, C, H, W = x.shape
        # (B, C, H, W) --> (B,C,HW) --> (B,HW,C)   HW:patch的总数;  flatten(): 指定维度后的所有维度合并到指定维度上;
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



if __name__ == '__main__':

    # (B, C‘, H', W')
    x = torch.randn(1, 3, 224, 224)

    ### 对图像中的patch进行划分并编码 ###
    PatchEmbed = PatchEmbed(img_size=224, patch_size=16, kernel_size=16, in_chans=3, embed_dim=512, overlap=False)
    x,H,W = PatchEmbed(x)  # x:(B,HW,C), HW:patch的总数, H: 每列有h个patch  W: 每行有w个patch
    print(x.shape,H,W)

    dim=x.shape[-1]  # patch embedding之后的通道数

    ### P2TAttention ###
    Model = P2TAttention(
        dim=dim, num_heads=8, qkv_bias=True, qk_scale=None,
        attn_drop=0., proj_drop=0., pool_ratios=[1 ,2 ,3 ,6])

    # 输入: x:(B, N, C)   N==HW: patch(Token)的数量    H: 每列有h个patch  W: 每行有w个patch
    # 输出: out: (B, N, C)
    out = Model(x, H, W)
    print(out.shape)


