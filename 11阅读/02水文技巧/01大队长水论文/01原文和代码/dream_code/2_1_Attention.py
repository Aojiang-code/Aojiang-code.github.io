import numpy as np
import torch
from torch import nn
from torch.nn import init

"Attention Is All You Need"

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        # (B, N, C), N=nq
        B, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(B, nq, self.h, self.d_k).permute(0, 2, 1, 3)  #  (B,N,C)-->(B,nq,h*d_k)-->(B,nq,h,d_k)-->(B,h,nq,d_k)  h:注意力头的个数, d_k:QK每一个注意力头的通道数
        k = self.fc_k(keys).view(B, nk, self.h, self.d_k).permute(0, 2, 3, 1)  #  (B,N,C)-->(B,nq,h*d_k)-->(B,nk,h,d_k)-->(B,h,d_k,nk)
        v = self.fc_v(values).view(B, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (B,N,C)-->(B,nk,h*d_v)-->(B,nk,h,d_v)-->(B,h,nk,d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (B, h, nq, nk)
        # 如果需要为注意力矩阵额外添加一个参数矩阵,那么执行逐点相乘即可
        if attention_weights is not None:
            att = att * attention_weights
        # 如果需要为注意力矩阵添加mask,那么在对应需要mask的地方填充为负无穷数值,这样在计算softmax的时候,负无穷的归一化得分将趋近于0
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, nq, self.h * self.d_v)  #  (B,h,nq,nk)@(B,h,nk,d_v)=(B,h,nq,d_v)-->(B,nq,h,d_v)-->(B,nq,h*d_v)
        out = self.fc_o(out)  # (B,nq,C)
        return out


if __name__ == '__main__':
    # (B, N, C)
    input=torch.randn(2,50,64)
    Model = ScaledDotProductAttention(d_model=64, d_k=64, d_v=64, h=8)
    output=Model(input,input,input)
    print(output.shape)

    