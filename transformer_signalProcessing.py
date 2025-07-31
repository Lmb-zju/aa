import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义位置编码（适配连续信号）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 调整后的Transformer模型
class SignalTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, num_heads=4, num_layers=3, d_ff=256, max_seq_length=100, dropout=0.1):
        super().__init__()

        # 输入投影层（替代原Embedding）
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, src):
        # 输入形状: (batch_size, seq_len, input_dim)
        src = self.input_proj(src)  # 投影到模型维度
        src = self.pos_encoder(src)  # 添加位置编码

        # Transformer需要(seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)

        # 恢复形状并输出
        output = output.permute(1, 0, 2)
        return self.output_layer(output)


# 多径信道模型
class MultipathChannel:
    def __init__(self, taps=[0.8, 0.3, 0.1], noise_std=0.1):
        self.taps = np.array(taps)  # 信道抽头系数
        self.noise_std = noise_std  # 噪声标准差

    def transmit(self, signals):
        # 添加多径效应
        convolved = np.convolve(signals, self.taps, mode='same')
        # 添加高斯噪声
        noisy = convolved + np.random.normal(0, self.noise_std, convolved.shape)
        return noisy


# 数据生成函数
def generate_pam4_data(num_samples, seq_length, channel):
    # 生成二进制序列
    binary_data = np.random.randint(0, 2, (num_samples, seq_length * 2))

    # 转换为PAM4信号（每2bit映射到4个电平）
    pam4_map = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 0): 1,
        (1, 1): 3
    }

    pam4_signals = np.zeros((num_samples, seq_length))
    distorted_signals = np.zeros((num_samples, seq_length))

    for i in range(num_samples):
        # 生成理想PAM4信号
        ideal = [pam4_map[tuple(binary_data[i, 2 * j:2 * (j + 1)])] for j in range(seq_length)]
        # 通过信道传输
        distorted = channel.transmit(ideal)

        pam4_signals[i] = ideal
        distorted_signals[i] = distorted

    return (
        torch.FloatTensor(distorted_signals).unsqueeze(-1).to(device),  # 输入：失真信号
        torch.FloatTensor(pam4_signals).unsqueeze(-1).to(device)  # 标签：理想信号
    )


# 超参数配置
# 根据信号复杂程度调整模型参数
config = {
    'd_model': 128,         # 增加模型容量
    'num_heads': 8,         # 增加注意力头
    'num_layers': 6,        # 加深网络
    'd_ff': 512,            # 扩大前馈网络
    'seq_length': 128,      # 支持更长序列
    'batch_size': 128,      # 更大批量
    'lr': 0.001,           # 调整学习率
    'epochs': 200
}

# 初始化模型和优化器
model = SignalTransformer(
    input_dim=1,
    d_model=config['d_model'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    d_ff=config['d_ff'],
    max_seq_length=config['seq_length']
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.MSELoss()

# 信道实例
# 可根据实际测量数据调整信道参数
channel = MultipathChannel(
    taps=[0.7, 0.25, 0.1, 0.05],  # 多径抽头系数
    noise_std=0.2                 # 噪声强度
)


# 训练循环
for epoch in range(config['epochs']):
    # 生成训练数据
    inputs, labels = generate_pam4_data(config['batch_size'], config['seq_length'], channel)

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证
    with torch.no_grad():
        val_inputs, val_labels = generate_pam4_data(64, config['seq_length'], channel)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{config['epochs']} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")


# 测试函数
def evaluate_model(model, channel, seq_length=64, num_samples=1000):
    model.eval()
    total_mse = 0
    ber = 0

    with torch.no_grad():
        for _ in range(num_samples // 100):
            inputs, labels = generate_pam4_data(100, seq_length, channel)
            outputs = model(inputs)

            # 计算MSE
            total_mse += criterion(outputs, labels).item() * 100

            # 计算误码率
            predicted = torch.clamp(torch.round(outputs.squeeze() / 2), min=-3, max=3)
            original = labels.squeeze()
            ber += torch.sum(predicted != original).item()

    avg_mse = total_mse / num_samples
    avg_ber = ber / (num_samples * seq_length)
    return avg_mse, avg_ber


# 性能评估
test_mse, test_ber = evaluate_model(model, channel)
print(f"Test MSE: {test_mse:.4f} | Test BER: {test_ber:.4%}")


import matplotlib.pyplot as plt

def plot_signals(input_signal, output_signal, label_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(input_signal.cpu().detach().numpy(), 'r--', alpha=0.5, label='Distorted')
    plt.plot(output_signal.cpu().detach().numpy(), 'g-', lw=2, label='Predicted')
    plt.plot(label_signal.cpu().detach().numpy(), 'b--', alpha=0.3, label='Ideal')
    plt.legend()
    plt.title("Signal Comparison")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

# 生成示例信号
test_input, test_label = generate_pam4_data(1, 64, channel)
test_output = model(test_input)
plot_signals(test_input[0,:,0], test_output[0,:,0], test_label[0,:,0])
