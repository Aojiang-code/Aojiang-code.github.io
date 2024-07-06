import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class gatedFusion(nn.Module):

    def __init__(self, dim):
        super(gatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        x11 = self.fc1(x1)
        x22 = self.fc2(x2)
        # 通过门控单元生成权重表示
        z = torch.sigmoid(x11+x22)
        # 对两部分输入执行加权和
        out = z*x1 + (1-z)*x2
        return out



if __name__ == '__main__':
    # 时间序列: (B, N, T, C)
    # x1 = torch.randn(1, 10, 24, 64)
    # x2 = torch.randn(1, 10, 24, 64)

    # 图像：(B,H,W,C)
    x1 = torch.randn(1,224,224,64)
    x2 = torch.randn(1,224,224,64)

    Model = gatedFusion(dim=64)
    out = Model(x1,x2)
    print(out.shape)