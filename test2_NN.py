# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import random
# import numpy as np

# print(np.__version__)
# print(torch.__version__)
# x = torch.ones(2, 2, dtype=torch.float)
# print(x)
# # y = x.view(-1)
# # print(y)
# # z = torch.empty(5, 3)
# # print(z)
# y = torch.add(x, 1)
#
# t = [[2, 2], [2, 2]]
# l = torch.tensor(t, dtype=torch.float)
# print(t)
# print(l)
#
# z = torch.empty(2, 2, requires_grad = True)
# torch.add(x, l, out=z)
# print(z)
#
# z.requires_grad = True
# print(z)
# print(z.grad_fn)

'''
创建神经网络模型
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(10, 4)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return x


net = Net()
print(net)


'''
内部参数（权重）可视化
'''
para = list(net.parameters())
# print(para)
# print(len(para))
# print(para[0].size())
print(f'"before backward"：{para[0].data}')
# print(para[1].size())

'''
准备输入输出（数据集）及其可视化
'''
# 读取 .mat 文件
mat_data = sio.loadmat('F:/DataSet/Demodulation DataSets/normalized data2/QPSK/10/0cm.mat')
mat_label = sio.loadmat('F:/DataSet/Demodulation DataSets/normalized data2/QPSK/10/label.mat')

data = mat_data['data_10p_0cm']
data = data[:, 0:1000]
label = mat_label['org_label']
label = label[:, 0:1000]
label_onehot = np.zeros((4, 1000))
for col in label:
    for i, val in enumerate(col):
        label_onehot[val, i] = 1

# 设定随机数种子，不需要可重复就注释掉
torch.manual_seed(0)
# x = [random.random()] * 10
# y = [random.random()] * 10
input = torch.rand(1, 10)
output = torch.rand(1, 4)
target = torch.rand(1, 4)
# z = torch.rand(1, 10)
# print(input)
# print(output)
# print(target)
# print(z)

# 创建 DataLoader 实例
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 迭代 DataLoader
for data in test_loader:
    imgs, targets = data
    print(imgs.shape) # 打印图片的形状
    print(targets) # 打印目标标签


'''
正向传播
'''
output = net(input)
print(output)

'''
定义损失函数
'''
criterion = nn.MSELoss()
loss = criterion(output, target)
print(f'loss:{loss}')
print(loss.grad_fn)

'''
定义参数优化方式：优化器
'''
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 设置迭代次数
max_epochs = 1000
for epoch in range(max_epochs):
    output = net(input)
    loss = criterion(output, target)
    if epoch % 200 == 0:
        print(f'loss:{loss}')
    # print(loss.grad_fn)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f'"after backward":{para[0].data}')

# # 每次更新需要执行梯度清零，否则梯度会保留叠加上一批（batch）数据
# optimizer.zero_grad()
# '''
# 反向传播
# '''
# loss.backward()
# # 执行step才会更新
# optimizer.step()
# print(f'"after backward":{para[0].data}')

