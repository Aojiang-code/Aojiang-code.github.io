import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class ParC_operator(nn.Module):
    def __init__(self, dim, type, global_kernel_size, use_pe=True):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.use_pe = use_pe
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim)
        if use_pe:
            if self.type == 'H':
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            elif self.type == 'W':
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        if self.use_pe:
            x = x + self.pe.expand(1, self.dim, self.global_kernel_size, self.global_kernel_size)

        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type == 'H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = self.gcc_conv(x_cat)

        return x


class ParC_example(nn.Module):
    def __init__(self, dim, global_kernel_size=14, use_pe=True):
        super().__init__()
        self.gcc_H = ParC_operator(dim // 2, 'H', global_kernel_size, use_pe)
        self.gcc_W = ParC_operator(dim // 2, 'W', global_kernel_size, use_pe)

    def forward(self, x):
        x_H, x_W = torch.chunk(x, 2, dim=1)
        x_H, x_W = self.gcc_H(x_H), self.gcc_W(x_W)
        x = torch.cat((x_H, x_W), dim=1)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = ParC_example(dim=64, global_kernel_size=56)  # global_kernel_size 设置为和图像大小一样
    input = torch.rand(3, 64, 56, 56)
    output = block(input)
    print(input.size(), output.size())
