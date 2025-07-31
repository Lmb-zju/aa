import torch
from torch.utils.data import Dataset, DataLoader
model = torch.load("F:\Code_Test\github\PNN\pytorch-onn-main\examples\checkpoint\MZI_CLASS_CNN_acc-99.15_epoch-182.pt")
model.forward()
test = DataLoader(dataset="F:\DataSet\MNIST\mnist\MNIST")
from torch.utils.data import DataLoader, TensorDataset
import pytest


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

# # 测试类
# class TestONNModel:
#     @pytest.fixture
#     def model(self):
#         """创建模型实例"""
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         return ONNModel(device=device)
#
#     @pytest.fixture
#     def dummy_input(self):
#         """创建虚拟输入数据"""
#         return torch.randn(4, 1, 28, 28)  # 批量大小4, 通道1, 高28, 宽28
#
#     def test_model_initialization(self, model):
#         """测试模型初始化"""
#         # 验证设备设置
#         assert next(model.parameters()).device.type in ['cuda', 'cpu']
#
#         # 验证层类型
#         assert isinstance(model.conv, layers.MZIBlockConv2d)
#         assert isinstance(model.pool, nn.AdaptiveAvgPool2d)
#         assert isinstance(model.linear, layers.MZIBlockLinear)
#
#         # 验证参数初始化
#         for name, param in model.named_parameters():
#             assert not torch.isnan(param).any(), f"NaN found in {name}"
#             assert not torch.isinf(param).any(), f"Inf found in {name}"
#
#     def test_forward_pass(self, model, dummy_input):
#         """测试前向传播"""
#         # 运行前向传播
#         output = model(dummy_input)
#
#         # 验证输出形状
#         assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
#
#         # 验证输出值
#         assert not torch.isnan(output).any(), "NaN in output"
#         assert not torch.isinf(output).any(), "Inf in output"
#         assert torch.all(torch.isfinite(output)), "Non-finite values in output"
#
#     def test_training_step(self, model, dummy_input):
#         """测试训练步骤"""
#         # 设置优化器和损失函数
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         criterion = nn.CrossEntropyLoss()
#
#         # 创建标签
#         labels = torch.tensor([3, 7, 2, 5])  # 随机标签
#
#         # 训练步骤
#         model.train()
#         optimizer.zero_grad()
#         output = model(dummy_input)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#
#         # 验证损失值
#         assert not torch.isnan(loss), "Loss is NaN"
#         assert loss.item() > 0, "Loss should be positive"
#
#         # 验证梯度
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
#                 assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
#
#     def test_conv_layer_output(self, model, dummy_input):
#         """测试卷积层输出"""
#         # 获取卷积层输出
#         conv_output = model.conv(dummy_input)
#
#         # 验证输出形状
#         assert conv_output.shape == (4, 8, 28, 28), (
#             f"Expected shape (4, 8, 28, 28), got {conv_output.shape}"
#         )
#
#         # 验证ReLU激活
#         assert torch.all(conv_output >= 0), "Conv output should be non-negative after ReLU"
#
#     def test_pooling_layer_output(self, model, dummy_input):
#         """测试池化层输出"""
#         # 通过卷积和池化层
#         conv_output = torch.relu(model.conv(dummy_input))
#         pool_output = model.pool(conv_output)
#
#         # 验证输出形状
#         assert pool_output.shape == (4, 8, 5, 5), (
#             f"Expected shape (4, 8, 5, 5), got {pool_output.shape}"
#         )
#
#     def test_linear_layer_input(self, model, dummy_input):
#         """测试线性层输入形状"""
#         # 通过所有层直到线性层
#         x = torch.relu(model.conv(dummy_input))
#         x = model.pool(x)
#         flattened = x.flatten(1)
#
#         # 验证展平后的形状
#         assert flattened.shape == (4, 8 * 5 * 5), (
#             f"Expected shape (4, 200), got {flattened.shape}"
#         )
#
#     def test_device_consistency(self):
#         """测试设备一致性"""
#         # 测试CPU设备
#         cpu_model = ONNModel(device=torch.device("cpu"))
#         cpu_input = torch.randn(4, 1, 28, 28)
#         cpu_output = cpu_model(cpu_input)
#         assert cpu_output.device.type == "cpu"
#
#         # 测试GPU设备（如果可用）
#         if torch.cuda.is_available():
#             gpu_model = ONNModel(device=torch.device("cuda"))
#             gpu_input = torch.randn(4, 1, 28, 28).cuda()
#             gpu_output = gpu_model(gpu_input)
#             assert gpu_output.device.type == "cuda"
#
#     def test_photodetect_property(self, model):
#         """测试photodetect属性设置"""
#         assert model.conv.photodetect == True, "Conv layer photodetect should be True"
#         assert model.linear.photodetect == True, "Linear layer photodetect should be True"
#
#     def test_decompose_alg_property(self, model):
#         """测试分解算法设置"""
#         assert model.conv.decompose_alg == "clements", "Conv layer should use Clements decomposition"
#         assert model.linear.decompose_alg == "clements", "Linear layer should use Clements decomposition"
#
#     def test_miniblock_property(self, model):
#         """测试miniblock属性设置"""
#         assert model.conv.miniblock == 4, "Conv layer miniblock should be 4"
#         assert model.linear.miniblock == 4, "Linear layer miniblock should be 4"
#
#     def test_model_with_dataloader(self, model):
#         """测试模型与数据加载器的配合"""
#         # 创建虚拟数据集
#         dataset = TensorDataset(
#             torch.randn(32, 1, 28, 28),  # 32个样本
#             torch.randint(0, 10, (32,))  # 随机标签
#         )
#         dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#         # 测试模型处理整个数据集
#         for inputs, labels in dataloader:
#             inputs = inputs.to(model.device)
#             outputs = model(inputs)
#             assert outputs.shape == (4, 10), "Batch output shape mismatch"
#
#     def test_model_save_load(self, model, dummy_input, tmp_path):
#         """测试模型保存和加载"""
#         # 保存模型
#         model_path = tmp_path / "onn_model.pth"
#         torch.save(model.state_dict(), model_path)
#
#         # 创建新模型并加载
#         new_model = ONNModel(device=model.device)
#         new_model.load_state_dict(torch.load(model_path))
#
#         # 验证输出一致性
#         original_output = model(dummy_input)
#         new_output = new_model(dummy_input)
#
#         assert torch.allclose(original_output, new_output, atol=1e-6), "Outputs differ after save/load"
