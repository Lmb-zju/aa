import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att


def test_self_attention():
    # 设置参数
    batch_size = 2
    seq_len = 3  # 输入序列长度（相当于n）
    dim_in = 4  # 输入特征维度
    dim_k = 2  # 键/查询的投影维度
    dim_v = 5  # 值的投影维度

    # 初始化自注意力模块
    sa = SelfAttention(dim_in, dim_k, dim_v)

    # 创建测试输入（手工设计可解释的值）
    x = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],  # 第一个样本的序列
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],

        [[1.0, 1.0, 0.0, 0.0],  # 第二个样本的序列
         [0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 1.0, 1.0]]
    ], requires_grad=True)  # 添加梯度追踪

    print("输入形状:", x.shape)  # 应该输出 [2, 3, 4]

    # 前向传播
    att = sa(x)

    """ 验证关键步骤 """
    # 1. 检查线性变换后的形状
    q = sa.linear_q(x)
    k = sa.linear_k(x)
    v = sa.linear_v(x)
    print("\n查询(q)形状:", q.shape)  # 应为 [2,3,2]
    print("键(k)形状:", k.shape)  # 应为 [2,3,2]
    print("值(v)形状:", v.shape)  # 应为 [2,3,5]

    # 2. 验证注意力得分计算
    dist = torch.bmm(q, k.transpose(1, 2)) * sa._norm_fact
    print("\n原始注意力得分矩阵形状:", dist.shape)  # 应为 [2,3,3]
    print("缩放因子:", sa._norm_fact)  # 应为 1/sqrt(2) ≈ 0.7071

    # 3. 验证softmax应用
    softmax_dist = torch.softmax(dist, dim=-1)
    print("\nsoftmax后每行求和:", torch.sum(softmax_dist[0, 0, :]).item())  # 应接近1.0

    # 4. 验证最终输出
    print("\n最终注意力输出形状:", att.shape)  # 应为 [2,3,5]
    print("第一个样本的注意力输出:", att[0].detach().numpy().round(4))

    # 5. 验证梯度反向传播
    att.sum().backward()
    print("\n输入梯度是否存在:", x.grad is not None)  # 应为True


if __name__ == "__main__":
    test_self_attention()
