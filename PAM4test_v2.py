import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义位置编码（适配长序列）
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


# 改进的Transformer模型（处理过采样信号）
class OversamplingTransformer(nn.Module):
    def __init__(self, samples_per_symbol=4, d_model=64, num_heads=4,
                 num_layers=3, d_ff=256, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.sps = samples_per_symbol

        # 输入处理
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出处理
        self.symbol_decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4)  # 输出4个电平的概率
        )

    def forward(self, x):
        # x形状: (batch_size, seq_len, 1)
        x = self.input_proj(x)  # (batch, seq, d_model)
        x = self.pos_encoder(x)

        # Transformer处理
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        encoded = self.transformer(x)
        encoded = encoded.permute(1, 0, 2)  # (batch, seq, d_model)

        # 符号级处理
        batch_size, seq_len, _ = encoded.shape
        # 降采样到符号率
        encoded = encoded[:, ::self.sps, :]  # (batch, symbols, d_model)

        # 输出每个符号的电平概率
        return self.symbol_decoder(encoded)


# 信号生成参数
class SignalConfig:
    def __init__(self):
        self.sps = 4  # 每个符号的采样点数
        self.symbol_rate = 25e9  # 符号速率 (25 GBaud)
        self.rolloff = 0.3  # 升余弦滚降系数
        self.filter_span = 16  # 滤波器跨度（符号数）
        self.taps = [0.8, 0.3, 0.1]  # 多径信道抽头
        self.noise_std = 0.15  # 噪声标准差


def root_raised_cosine(beta, span, sps):
    """手动实现根升余弦滤波器"""
    t = np.linspace(-span / 2, span / 2, span * sps + 1)
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif abs(4 * beta * ti) == 1:
            h[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    # 能量归一化
    h /= np.sqrt(np.sum(h ** 2))
    return h


# 改进的信号生成器（包含脉冲成型和过采样）
class PAM4Generator:
    def __init__(self, config):
        self.config = config
        self.rrc_filter = root_raised_cosine(
            beta=config.rolloff,
            span=config.filter_span,
            sps=config.sps
        )

        # 定义PAM4映射（二进制到类别索引）
        self.pam4_map = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }

    def generate_symbols(self, num_symbols):
        # 生成二进制数据 (num_symbols, 2)
        binary = np.random.randint(0, 2, (num_symbols, 2))
        # 映射到类别索引 (num_symbols,)
        symbols = np.array([self.pam4_map[tuple(b)] for b in binary])
        return symbols

    def generate_batch(self, batch_size, seq_length):
        batch_signals = []
        batch_labels = []

        for _ in range(batch_size):
            # 生成符号类别索引 (shape: seq_length,)
            symbols = self.generate_symbols(seq_length)
            # 转换为电平值用于脉冲成型
            levels = np.array([[-3, -1, 1, 3][idx] for idx in symbols])

            # 脉冲成型和信道传输
            shaped = self.apply_pulse_shaping(levels)
            distorted = self.apply_channel(shaped)

            batch_signals.append(distorted)
            batch_labels.append(symbols)  # 存储类别索引

        # 转换为张量
        signals_tensor = torch.FloatTensor(np.array(batch_signals)).unsqueeze(-1).to(device)  # (batch, seq*sps, 1)
        labels_tensor = torch.LongTensor(np.array(batch_labels)).to(device)  # (batch, seq)

        return signals_tensor, labels_tensor


# 配置参数
config = {
    'sps': 4,
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 3,
    'd_ff': 256,
    'symbol_length': 64,  # 符号数量
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 100
}

# 初始化组件
sig_config = SignalConfig()
sig_config.sps = config['sps']
generator = PAM4Generator(sig_config)
model = OversamplingTransformer(
    samples_per_symbol=config['sps'],
    d_model=config['d_model'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    d_ff=config['d_ff'],
    max_seq_length=config['symbol_length'] * config['sps']
).to(device)

# 使用交叉熵损失（分类任务）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

# 训练循环
for epoch in range(config['epochs']):
    # 生成训练数据
    inputs, labels = generator.generate_batch(config['batch_size'], config['symbol_length'])

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, 4), labels.view(-1))

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证
    with torch.no_grad():
        val_inputs, val_labels = generator.generate_batch(64, config['symbol_length'])
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs.view(-1, 4), val_labels.view(-1))

    # 计算准确率
    _, predicted = torch.max(outputs, dim=-1)
    accuracy = (predicted == labels).float().mean()

    if (epoch + 1) % 10 ==0:
        print(f"Epoch {epoch + 1}/{config['epochs']} | "
              f"Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
              f"Acc: {accuracy.item():.2%}")


# 评估函数
def evaluate_model(model, generator, num_samples=1024):
    model.eval()
    total_correct = 0
    total_symbols = 0

    with torch.no_grad():
        for _ in range(num_samples // 64):
            inputs, labels = generator.generate_batch(64, config['symbol_length'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=-1)
            total_correct += (predicted == labels).sum().item()
            total_symbols += labels.numel()

    ber = 1 - total_correct / total_symbols
    return ber


# 性能评估
test_ber = evaluate_model(model, generator)
print(f"Test BER: {test_ber:.2%}")
