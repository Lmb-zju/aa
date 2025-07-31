import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# 读取 .mat 文件
mat_data = sio.loadmat('F:/DataSet/Demodulation DataSets/normalized data2/QPSK/10/0cm.mat')
mat_label = sio.loadmat('F:/DataSet/Demodulation DataSets/normalized data2/QPSK/10/label.mat')


# 打印所有变量名
print("变量名:", mat_data.keys())
print("变量名:", mat_label.keys())

# 提取名为 'data_13_dB' 的变量
data = mat_data['data_10p_0cm']
data = data[:, 0:1000]
label = mat_label['org_label']
label = label[:, 0:1000]
label_onehot = np.zeros((4, 1000))
for col in label:
    for i, val in enumerate(col):
        label_onehot[val, i] = 1

print("数据:\n", data)
print("标签:\n", label)
print("onehot标签:\n", label_onehot)

# 查看变量属性
print(f'shape:{data.shape}') # 10*72000
print(f'shape:{label.shape}') # 1*72000


# plt.plot(data[0, 0:100])
# plt.show()

# plt.plot(data_13_dB[0, 0:100])
# plt.show()

# 创建 DataLoader 实例
test_loader = DataLoader(dataset=data, batch_size=4 , shuffle=True, num_workers=0, drop_last=False, pin_memory=True)

# 迭代 DataLoader
for data_loader in test_loader:
    print(data_loader)
