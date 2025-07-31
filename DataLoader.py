import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 自定义一个类并继承自Dataset类,此类用来封装数据
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # xy是一个N行9列的矩阵，所以xy.shape拿到的就是（N，9）这样一个元组，取索引0位的数字就可以得到数据的条数。
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # getitem函数，可以使用索引拿到数据
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 返回数据的条数/长度
    def __len__(self):
        return self.len


# 实例化自定义类，并传入数据地址
dataset = DiabetesDataset('diabetes.csv.gz')

# num_workers是否要进行多线程服务，num_worker=2 就是2个进程并行运行
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 防止报错
if __name__ == '__main__':
    # 采用Mini-Batch的方法训练要采用多层嵌套循环
    # 所有数据都跑100遍
    for epoch in range(100):
        # enumerate可以获得当前是第几次迭代，内部迭代每一次跑一个Mini-Batch
        for i, data in enumerate(train_loader, 0):
            # data从train_loader中取出数据（取出的是一个元组数据）
            # inputs获取到data中的x的值，labels获取到data中的y值
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


