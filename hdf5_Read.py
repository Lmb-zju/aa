import h5py
import numpy as np
import matplotlib.pyplot as plt
# Dataset = h5py.File('F:\DataSet\调制格式识别数据集\RML数据集2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
# for name in Dataset:
#     print(name)
# with Dataset as f:
#     x = f["Y"][0:2000]
# print(x.dtype,x.shape ,x.size)
# # print(x)
# plt.plot(x)
# plt.show()

h5file = h5py.File('F:\DataSet\调制格式识别数据集\RML数据集2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
savefile=h5py.File('F:\DataSet\调制格式识别数据集\RML数据集2018.01.OSC.0001_1024x2M.h5/2018.01/mini_dataset.hdf5', 'w')

X = h5file['X'][15 * 4096:16 * 4096]  # 15是因为10db在  26个信噪比的第15个
Y = h5file['Y'][15 * 4096:16 * 4096]
Z = h5file['Z'][15 * 4096:16 * 4096]
for i in range(1, 24, 1):
    X = np.vstack((X, h5file['X'][4096 * 26 * i + 15 * 4096:4096 * 26 * i + 16 * 4096]))
    Y = np.vstack((Y, h5file['Y'][4096 * 26 * i + 15 * 4096:4096 * 26 * i + 16 * 4096]))
    Z = np.vstack((Z, h5file['Z'][4096 * 26 * i + 15 * 4096:4096 * 26 * i + 16 * 4096]))

savefile['X']=X
savefile['Y']=Y
savefile['Z']=Z

print(savefile['X'].shape)