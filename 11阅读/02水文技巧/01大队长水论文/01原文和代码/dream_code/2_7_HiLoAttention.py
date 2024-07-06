import math
import torch
import torch.nn as nn

"Fast Vision Transformers with HiLo Attention"


class HiLo(nn.Module):
    """
    HiLo Attention

    Paper: Fast Vision Transformers with HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads) # 每个注意力头的通道数
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)  # 根据alpha来确定分配给低频注意力的注意力头的数量
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim   # 确定低频注意力的通道数

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads # 总的注意力头个数-低频注意力头的个数==高频注意力头的个数
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim  # 确定高频注意力的通道数, 总通道数-低频注意力通道数==高频注意力通道数

        # local window size. The `s` in our paper.
        self.ws = window_size  # 窗口的尺寸, 如果ws==2, 那么这个窗口就包含4个patch(或token)

        # 如果窗口的尺寸等于1,这就相当于标准的自注意力机制了, 不存在窗口注意力了; 因此,也就没有高频的操作了,只剩下低频注意力机制了
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        # 如果低频注意力头的个数大于0, 那就说明存在低频注意力机制。 然后,如果窗口尺寸不为1, 那么应当为每一个窗口应用平均池化操作获得低频信息,这样有助于降低低频注意力机制的计算复杂度 （如果窗口尺寸为1,那么池化层就没有意义了）
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        # 如果高频注意力头的个数大于0, 那就说明存在高频注意力机制
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    # 高频注意力机制
    def hifi(self, x):
        B, H, W, C = x.shape

        # 每行有w_group个窗口, 每列有h_group个窗口;
        h_group, w_group = H // self.ws, W // self.ws

        # 总共有total_groups个窗口; 例如：HW=14*14=196个patch; 窗口尺寸为ws=2表示:每行每列有2个patch; 总共有:(14/2)*(14/2)=49个窗口,每个窗口有2*2=4个patch
        total_groups = h_group * w_group

        #通过reshape操作重塑X: (B,H,W,C) --> (B,h_group,ws,w_group,ws,C) --> (B,h_group,w_group,ws,ws,C)   H=h_group*ws, W=w_group*ws
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        # 通过线性层生成qkv: (B,h_group,w_group,ws,ws,C) --> (B,h_group,w_group,ws,ws,3*h_dim) --> (B,total_groups,ws*ws,3,h_heads,head_dim) -->(3,B,total_groups,h_heads,ws*ws,head_dim)    h_dim=h_heads*head_dim
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        # q:(B,total_groups,h_heads,ws*ws,head_dim); k:(B,total_groups,h_heads,ws*ws,head_dim); v:(B,total_groups,h_heads,ws*ws,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 在每个窗口内计算: 所有patch pairs之间的注意力得分: (B,total_groups,h_heads,ws*ws,head_dim) @ (B,total_groups,h_heads,head_dim,ws*ws) = (B,total_groups,h_heads,ws*ws,ws*ws);  ws*ws:表示一个窗口内的patch的数量
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对Value矩阵进行加权: (B,total_groups,h_heads,ws*ws,ws*ws) @ (B,total_groups,h_heads,ws*ws,head_dim) = (B,total_groups,h_heads,ws*ws,head_dim) --transpose->(B,total_groups,ws*ws,h_heads,head_dim)--reshape-> (B,h_group,w_group,ws,ws,h_dim) ;    h_dim=h_heads*head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)

        # 通过reshape操作重塑, 恢复与输入相同的shape: (B,h_group,w_group,ws,ws,h_dim) --transpose-> (B,h_group,ws,w_group,ws,h_dim) --reshape-> (B,h_group*ws,w_group*ws,h_dim) ==(B,H,W,h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        # 通过映射层进行输出: (B,H,W,h_dim)--> (B,H,W,h_dim)
        x = self.h_proj(x)
        return x

    # 低频注意力机制
    def lofi(self, x):
        B, H, W, C = x.shape
        # 低频注意力机制中的query来自原始输入x: (B,H,W,C) --> (B,H,W,l_dim) --> (B,HW,l_heads,head_dim) -->(B,l_heads,HW,head_dim);   l_dim=l_heads*head_dim;
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        # 如果窗口尺寸大于1, 在每个窗口执行池化 (如果窗口尺寸等于1,没有池化的必要)
        if self.ws > 1:
            # 重塑维度以便进行池化操作:(B,H,W,C) --> (B,C,H,W)
            x_ = x.permute(0, 3, 1, 2)
            # 在每个窗口执行池化操作: (B,C,H,W) --sr-> (B,C,H/ws,W/ws) --reshape-> (B,C,HW/(ws^2)) --permute-> (B, HW/(ws^2), C);   HW=patch的总数, 每个池化窗口内有: (ws^2)个patch, 池化完还剩下：HW/(ws^2)个patch; 例如：HW=196个patch,每个池化窗口有(2^2=4)个patch,池化完还剩下49个patch【每个patch汇总了之前4个patch的信息】
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # 将池化后的输出通过线性层生成kv:(B,HW/(ws^2),C) --l_kv-> (B,HW/(ws^2),l_dim*2) --reshape-> (B,HW/(ws^2),2,l_heads,head_dim) --permute-> (2,B,l_heads,HW/(ws^2),head_dim)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            # 如果窗口尺寸等于1, 那么kv和q一样, 来源于原始输入x: (B,H,W,C) --l_kv-> (B,H,W,l_dim*2) --reshape-> (B,HW,2,l_heads,head_dim) --permute-> (2,B,l_heads,HW,head_dim);  【注意: 如果窗口尺寸为1,那就不会执行池化操作,所以patch的数量也不会减少,依然是HW个patch】
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)

        # 以ws>1为例: k:(B,l_heads,HW/(ws^2),head_dim);  v:(B,l_heads,HW/(ws^2),head_dim)
        k, v = kv[0], kv[1]

        # 计算q和k之间的注意力矩阵: (B,l_heads,HW,head_dim) @ (B,l_heads,head_dim,HW/(ws^2)) == (B,l_heads,HW,HW/(ws^2))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对Value矩阵进行加权: (B,l_heads,HW,HW/(ws^2)) @ (B,l_heads,HW/(ws^2),head_dim) == (B,l_heads,HW,head_dim) --transpose->(B,HW,l_heads,head_dim)--reshape-> (B,H,W,l_dim);   l_dim=l_heads*head_dim
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        # 通过映射层输出: (B,H,W,l_dim)-->(B,H,W,l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        B, N, C = x.shape
        # H = W = 每一列/行有多少个patch
        H = W = int(N ** 0.5)
        # 将X重塑为四维: (B,N,C) --> (B,H,W,C)   【注意: 这里的H/W并不是图像的高和宽】
        x = x.reshape(B, H, W, C)

        # 如果分配给高频注意力的注意力头的个数为0,那么仅仅执行低频注意力
        if self.h_heads == 0:
            # (B,H,W,C) --> (B,H,W,l_dim)  此时,C=l_dim,因为所有的注意力头都分配给了低频注意力
            x = self.lofi(x)
            return x.reshape(B, N, C)

        # 如果分配给低频注意力的注意力头的个数为0,那么仅仅执行高频注意力
        if self.l_heads == 0:
            # 执行高频注意力: (B,H,W,C) --> (B,H,W,h_dim); 此时,C=h_dim,因为所有的注意力头都分配给了高频注意力
            x = self.hifi(x)
            return x.reshape(B, N, C)

        # 执行高频注意力: (B,H,W,C) --> (B,H,W,h_dim)
        hifi_out = self.hifi(x)
        # 执行低频注意力: (B,H,W,C) --> (B,H,W,l_dim)
        lofi_out = self.lofi(x)

        # 在通道方向上拼接高频注意力和低频注意力的输出: (B,H,W,h_dim+l_dim)== (B,H,W,C)
        x = torch.cat((hifi_out, lofi_out), dim=-1)
        # 将输出重塑为与输入相同的shape: (B,H,W,C)-->(B,N,C)
        x = x.reshape(B, N, C)

        return x




if __name__ == '__main__':
    # (B, num_token, C)   p:patch的尺寸, (H*W)/(p*p)=(H/p)*(W/p)=num_token 一般情况下,H=W,所以平方根: num_token**(-0.5)==(H/p)==(W/p)==每一行/列有多少个patch
    x = torch.randn(1, 196, 64)
    Model = HiLo(dim=64, num_heads=8, window_size=2, alpha=0.5)
    out = Model(x)
    print(out.shape)
