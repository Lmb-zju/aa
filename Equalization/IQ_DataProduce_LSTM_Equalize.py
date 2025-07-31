# 1. 数据生成：模拟非线性信道损伤
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import explained_variance_score

# 生成非线性信道损伤的模拟数据
class NonlinearChannelDataset(Dataset):
    def __init__(self, num_samples=10000, memory_length=5, snr_db=0):
        """
        参数:
            num_samples: 总样本数
            memory_length: 信道记忆长度（模拟有记忆非线性）
            snr_db: 信噪比（dB）
        """
        self.num_samples = num_samples
        self.memory_length = memory_length
        self.snr = 10 ** (snr_db / 10)  # 线性信噪比

        # 生成原始QPSK信号（I/Q两路）
        self.x = np.random.choice([-1, 1], size=(num_samples, 2))  # [num_samples, 2]

        # 模拟非线性信道损伤（含记忆效应）
        self.y = self._apply_nonlinear_channel(self.x)

    def _apply_nonlinear_channel(self, x):
        """施加非线性损伤模型（含记忆效应和噪声）"""
        y = np.zeros_like(x, dtype=np.float32)
        for n in range(self.memory_length, len(x)):
            # 1. 无记忆非线性：AM/AM转换（立方非线性模型）
            x_current = x[n]
            nonlinear_part = 0.1 * (x_current ** 3)  # 非线性分量

            # 2. 有记忆效应：加权历史信号
            memory_part = 0
            for k in range(1, self.memory_length + 1):
                memory_part += 0.2 * (x[n - k] * (0.8 ** k))  # 指数衰减记忆

            # 3. 合并非线性与记忆效应
            y[n] = x_current + nonlinear_part + memory_part

        # 添加高斯噪声
        noise_power = np.var(y) / self.snr
        y += np.random.normal(0, np.sqrt(noise_power), size=y.shape)
        return y

    def __len__(self):
        return self.num_samples - self.memory_length  # 有效样本数

    def __getitem__(self, idx):
        # 输入为记忆长度+当前时刻的信号窗口，输出为当前时刻的理想信号
        window = self.y[idx: idx + self.memory_length + 1]  # [memory_length+1, 2]
        x_target = self.x[idx + self.memory_length]  # [2]
        return torch.FloatTensor(window), torch.FloatTensor(x_target)


# 生成数据集
dataset = NonlinearChannelDataset(num_samples=10000, memory_length=5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 2. 可视化原始信号与受损信号
def plot_signals(dataset, num_points=100):
    """绘制原始信号与受损信号的对比"""
    fig = plt.figure(figsize=(12, 6))

    # 取前num_points个样本
    x = dataset.x[:num_points, :]
    y = dataset.y[:num_points, :]

    # I路信号对比
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x[:, 0], 'b-', label="Original I")
    ax1.plot(y[:, 0], 'r--', label="Distorted I")
    plt.title("I Component Comparison")
    ax1.legend()
    # plt.show()

    # Q路信号对比
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x[:, 1], 'b-', label="Original Q")
    ax2.plot(y[:, 1], 'r--', label="Distorted Q")
    plt.title("Q Component Comparison")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'IQ_origin_distort_SNR_{dataset.snr}_MemoryLength_{dataset.memory_length}_NumPoints_{num_points}.png')
    plt.show(block=False)  # 非阻塞显示
    plt.pause(5)  # 展示5s关闭，否则无法进行下一步的训练部分
    plt.close()


# 绘制信号对比图
plot_signals(dataset)


# 3. 神经网络补偿模型（完整训练代码）
class NeuralCompensator(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=3):
        """
        参数:
            input_size: 输入维度（I/Q两路信号，默认为2）
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
        """
        super().__init__()
        # LSTM层：处理时序依赖性（模拟信道记忆效应）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入格式为 [batch, seq_len, features]
        )
        # 全连接层：映射到补偿后的I/Q信号
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        # 仅取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        compensated_signal = self.fc(last_output)  # [batch_size, input_size]
        return compensated_signal


def train_model(dataloader, num_epochs=50, lr=0.001):
    # 初始化模型、损失函数和优化器
    model = NeuralCompensator()
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录训练过程中的损失
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()  # 训练模式

        for batch_x, batch_y in dataloader:
            # batch_x: [batch_size, seq_len, 2], batch_y: [batch_size, 2]
            optimizer.zero_grad()
            outputs = model(batch_x)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            epoch_loss += loss.item()

        # 记录平均epoch损失
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model, train_losses


# 训练模型
model, train_losses = train_model(dataloader, num_epochs=50)


def plot_training_loss(train_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, 'b-o', linewidth=2)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(f'TrainingLoss_SNR_{dataset.snr}_MemoryLength_{dataset.memory_length}_NumSamples_{dataset.num_samples}.png')
    plt.show(block=False)  # 非阻塞显示
    plt.pause(5)
    plt.close()


plot_training_loss(train_losses)


# 选取多个测试样本
def evaluate_model(model, dataset, num_test_samples=5, show_plots=True, save_plots=True):
    """
    评估模型性能并绘制多样本星座图对比

    参数:
        model: 训练好的补偿模型
        dataset: 测试数据集
        num_test_samples: 要展示的测试样本数量
        show_plots: 是否显示图表（默认为True）
        save_plots: 是否保存图表（默认为True）
    """
    model.eval()
    device = next(model.parameters()).device

    # 随机选择样本索引（确保不重复）
    sample_indices = np.random.choice(len(dataset), size=num_test_samples, replace=False)

    # 存储所有样本的真实值和预测值
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for idx in sample_indices:
            x_window, y_true = dataset[idx]
            x_window = x_window.unsqueeze(0).to(device)  # [1, seq_len, 2]

            # 预测
            y_pred = model(x_window).squeeze().cpu().numpy()
            y_true = y_true.numpy()

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

    # 转换为NumPy数组便于处理
    y_true_array = np.array(all_y_true)  # [num_samples, 2]
    y_pred_array = np.array(all_y_pred)  # [num_samples, 2]

    # --- 绘制星座图 ---
    plt.figure(figsize=(10, 10))

    # 原始信号（蓝色圆圈）
    plt.scatter(
        y_true_array[:, 0], y_true_array[:, 1],
        c='blue', marker='o', s=80, alpha=0.7,
        label=f'Original (n={num_test_samples})'
    )

    # 补偿后信号（红色叉号）
    plt.scatter(
        y_pred_array[:, 0], y_pred_array[:, 1],
        c='red', marker='x', s=100, alpha=0.7,
        label=f'Compensated (n={num_test_samples})'
    )

    # 标注理想QPSK位置（绿色虚线）
    qpsk_ideal = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    plt.scatter(
        qpsk_ideal[:, 0], qpsk_ideal[:, 1],
        c='green', marker='+', s=200, alpha=0.3,
        label='Ideal QPSK'
    )

    # 图表装饰
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(f"Constellation Comparison\nSNR={dataset.snr}dB | Memory={dataset.memory_length}")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 保存图像
    if save_plots:
        filename = (
            f"constellation_SNR{dataset.snr}_"
            f"Mem{dataset.memory_length}_"
            f"Samples{num_test_samples}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    # 显示图像
    if show_plots:
        plt.show(block=False)
        plt.pause(3)  # 显示3秒后自动关闭
        plt.close()
    else:
        plt.close()

    # --- 计算统计指标 ---
    mse = mean_squared_error(y_true_array, y_pred_array)
    evar = explained_variance_score(y_true_array, y_pred_array)

    print(f"\nEvaluation Results ({num_test_samples} samples):")
    print(f"- MSE: {mse:.6f}")
    print(f"- Explained Variance: {evar:.4f}")
    print(f"- Max Deviation: {np.max(np.abs(y_true_array - y_pred_array)):.4f}")


# 评估模型并显示5个样本的星座图
evaluate_model(model, dataset, num_test_samples=5)

# 评估10个样本但不显示图表（仅保存）
evaluate_model(model, dataset, num_test_samples=10, show_plots=False)


# 选取单个测试样本
# def evaluate_model(model, dataset):
#     model.eval()  # 评估模式
#     with torch.no_grad():
#         # 随机选取一个测试样本
#         # idx = np.random.randint(0, len(dataset))
#         # x_window, y_true = dataset[idx]
#         # x_window = x_window.unsqueeze(0)  # 增加batch维度 [1, seq_len, 2]
#
#         # 随机选取多个测试样本
#         num_test_samples = 10
#         for i in range(num_test_samples):
#             idx = np.random.randint(0, len(dataset))
#             x_window, y_true = dataset[idx]
#             x_window = x_window.unsqueeze(0)  # 增加batch维度 [1, seq_len, 2]
#
#         # 模型预测
#         # y_pred = model(x_window).squeeze().numpy()
#         # y_true = y_true.numpy()
#
#             # 多样本测试
#             y_pred_multi_sample, y_true_multi_sample = [], []
#             y_pred_multi_sample.append(model(x_window).squeeze().numpy())
#             y_true_multi_sample.append(y_true.numpy())
#
#         # 绘制星座图对比
#         plt.figure(figsize=(8, 8))
#         plt.scatter([y_true_multi_sample[0]], [y_true_multi_sample[1]], c='b', marker='o', s=100, label="Original Signal")
#         plt.scatter([y_pred_multi_sample[0]], [y_pred_multi_sample[1]], c='r', marker='x', s=100, label="Compensated Signal")
#         plt.axhline(0, color='k', linestyle='--')
#         plt.axvline(0, color='k', linestyle='--')
#         plt.xlabel("I Component")
#         plt.ylabel("Q Component")
#         plt.title("Constellation Diagram (Before & After Compensation)")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(
#             f'Constellation Diagram (Before & After Compensation)_SNR_{dataset.snr}_MemoryLength_'
#             f'{dataset.memory_length}_NumSamples_{dataset.num_samples}.png')
#         plt.show(block=False)  # 非阻塞显示
#         plt.pause(5)
#         plt.close()
#
#         # 计算MSE
#         mse = mean_squared_error(y_pred_multi_sample, y_pred_multi_sample)
#         print(f"Compensation MSE: {mse:.6f}")


# 评估模型
evaluate_model(model, dataset)
