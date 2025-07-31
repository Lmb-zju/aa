# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

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

# 创建 DataLoader 实例
test_loader = DataLoader(dataset=data, batch_size=128, shuffle=True, num_workers=0, drop_last=False)

# 迭代 DataLoader
for data_loader in test_loader:
    imgs, targets = data
    print(imgs.shape)  # 打印图片的形状
    print(targets)  # 打印目标标签


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
        print(f'epoch{epoch} loss:{loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f'epoch{epoch} weight:{para[0].data}') # 打印权重


