import torch

import torchonn
import torch.nn as nn
import torch.nn.functional as F
import torchonn as onn
from torchonn.models import ONNBaseModel

class ONNModel(ONNBaseModel):
    def __init__(self, device=torch.device("cuda:0")):
        super().__init__(device=device)
        self.conv = onn.layers.MZIBlockConv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
    self.pool = nn.AdaptiveAvgPool2d(5)
    self.linear = onn.layers.MZIBlockLinear(
        in_features=8*5*5,
        out_features=10,
        bias=True,
        miniblock=4,
        mode="usv",
        decompose_alg="clements",
        photodetect=True,
        device=device,
    )

    self.conv.reset_parameters()
    self.linear.reset_parameters()

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    ob = ONNModel()
    x = "F:/DataSet/MNIST/mnist_jpg/test_0_7.jpg"