import numpy as np

# 已知信道特性的均衡

# 信道参数
h = np.array([0.9, 0.6, 0.3])  # 信道脉冲响应
SNR_db = 20                    # 信噪比 (dB)
sigma_x2 = 1.0                 # 发送信号功率 (PAM4符号方差)

# 计算噪声功率
SNR_linear = 10**(SNR_db / 10)
sigma_v2 = sigma_x2 / SNR_linear

# 计算信道自相关
corr_h = np.correlate(h, h, mode='full')
corr_h = corr_h[len(h)-1:]      # 取非负延迟部分

# 构建自相关矩阵 R (4x4)
R = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        k = abs(i - j)
        if k < len(corr_h):
            R[i, j] = sigma_x2 * corr_h[k]
            if i == j:
                R[i, j] += sigma_v2  # 对角线添加噪声项

# 构建互相关向量 p (延迟Δ=2)
Delta = 2
p = np.zeros(4)
for k in range(4):
    idx = Delta - k
    if 0 <= idx < len(h):
        p[k] = sigma_x2 * h[idx]

# 求解滤波器系数 w = R^{-1} p
w = np.linalg.inv(R) @ p

print("滤波器系数 w =", w)
