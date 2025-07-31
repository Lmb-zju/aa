import numpy as np
import torch
import torchonn
import torch.nn as nn
import torch.nn.functional as F
import torchonn as onn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import torchvision
import matplotlib.pyplot as plt
from torchonn.models import ONNBaseModel

datapath = 'F:/DataSet/MNIST/mnist/MNIST/raw'  # 本地mnist数据集

# 加载的.pt是dict(str:tensor)的形式
model = torch.load("F:\Code_Test\github\PNN\pytorch-onn-main\examples\checkpoint\MZI_CLASS_CNN_acc-99.15_epoch-182.pt")


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name,
                                              label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


trainDataset = DealDataset(datapath, "train-images-idx3-ubyte.gz",
                           "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

# 训练数据和测试数据的装载
train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=1,  # 一个批次可以认为是一个包，每个包中含有10张图片
    shuffle=False,
)

# # 实现单张图片可视化
images, labels = next(iter(train_loader))
temp = images.shape
print(temp)
# img = torchvision.utils.make_grid(images)
#
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
# print(labels)
# plt.imshow(img)
# plt.show()

class ONNModel(ONNBaseModel):
    def __init__(self, device=torch.device("cuda:0")):
        # super().__init__(device=device)
        super().__init__()
        self.conv1 = onn.layers.MZIBlockConv2d(
            in_channels=1,
            # out_channels=8,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            miniblock=8,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
        self.conv2 = onn.layers.MZIBlockConv2d(
            in_channels=64,
            # out_channels=8,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            miniblock=8,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
        self.pool = nn.AdaptiveAvgPool2d(5)
        self.linear = onn.layers.MZIBlockLinear(
            # in_features=8*5*5,
            in_features=64*5*5,
            out_features=10,
            bias=True,
            miniblock=8,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )

        # self.conv.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # x = self.pool(x)
        # 加一层卷积
        ''''''''''''''''''''''''''''''''''''''
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        ''''''''''''''''''''''''''''''''''''''
        x = x.flatten(1)
        x = self.linear(x)
        return x


model1 = ONNModel()
res = model1.forward(x=images.to(torch.device("cuda:0")))
print(res)
print(res.shape)

# 加载最好的模型.pt
model2 = ONNModel()
model2.load_parameters(model)
res2 = model2.forward(x=images.to(torch.device("cuda:0")))
print(res2)
print(res2.shape)