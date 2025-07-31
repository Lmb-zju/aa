import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
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