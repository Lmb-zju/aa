import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


def test_multihead_attention():
    # 设置参数
    batch_size = 2
    seq_len = 3  # 输入序列长度
    dim_in = 64  # 输入特征维度
    dim_k = 64  # 键/查询总维度
    dim_v = 64  # 值总维度
    num_heads = 4  # 注意力头数

    # 初始化模块
    mhsa = MultiHeadSelfAttention(dim_in, dim_k, dim_v, num_heads)

    # 构造测试输入（手工设计正交特征方便观察）
    x = torch.stack([
        # 样本1：三个正交向量
        torch.eye(3, dim_in).unsqueeze(0),
        # 样本2：逐步偏移的向量
        torch.cat([torch.eye(3), torch.zeros(3, dim_in - 3)], dim=1).unsqueeze(0)
    ], dim=1).squeeze(0)  # shape: (2,3,64)

    print("输入形状:", x.shape)

    # 前向传播
    att_output = mhsa(x)

    """ 验证关键步骤 """
    # 1. 线性变换后的维度拆分
    q = mhsa.linear_q(x)
    k = mhsa.linear_k(x)
    v = mhsa.linear_v(x)
    print("\n投影后形状:")
    print("Q shape:", q.shape)  # (2,3,64)
    print("K shape:", k.shape)
    print("V shape:", v.shape)

    # 2. 多头拆分验证
    q_multi = q.reshape(batch_size, seq_len, num_heads, dim_k // num_heads).transpose(1, 2)
    print("\n多头Q的形状:", q_multi.shape)  # (2,4,3,16)

    # 3. 注意力得分计算
    att_scores = torch.matmul(q_multi, q_multi.transpose(-2, -1)) * mhsa._norm_fact
    print("\n注意力得分形状:", att_scores.shape)  # (2,4,3,3)
    print("缩放因子:", mhsa._norm_fact)  # 1/sqrt(16)=0.25

    # 4. Softmax归一化
    softmax_scores = torch.softmax(att_scores, dim=-1)
    print("\nSoftmax后每行和:", torch.sum(softmax_scores[0, 0, 0, :]).item())  # 应接近1.0

    # 5. 值加权聚合
    v_multi = v.reshape(batch_size, seq_len, num_heads, dim_v // num_heads).transpose(1, 2)
    weighted_v = torch.matmul(softmax_scores, v_multi)
    print("\n加权值形状:", weighted_v.shape)  # (2,4,3,16)

    # 6. 最终输出合并
    final_output = weighted_v.transpose(1, 2).reshape(batch_size, seq_len, dim_v)
    print("\n最终输出形状:", final_output.shape)  # (2,3,64)

    # 7. 梯度反向传播验证
    final_output.sum().backward()
    print("\n梯度检查:")
    print("Q权重梯度存在:", mhsa.linear_q.weight.grad is not None)
    print("K权重梯度存在:", mhsa.linear_k.weight.grad is not None)

if __name__ == '__main__':
    test_multihead_attention()