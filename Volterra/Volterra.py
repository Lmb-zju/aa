import numpy as np
import matplotlib.pyplot as plt

# 定义一阶和二阶Volterra核函数
def h1(tau):
    """一阶核函数，简单的指数衰减"""
    return np.exp(-tau)

def h2(tau1, tau2):
    """二阶核函数，双指数衰减"""
    return np.exp(-(tau1 + tau2))

# 生成输入信号
t = np.linspace(0, 10, 100)  # 时间范围从0到10，共100个点
x = np.sin(t)  # 输入信号为正弦函数

# 计算输出响应
def volterra_output(t, x, h1, h2):
    """
    计算系统的输出响应
    :param t: 时间数组
    :param x: 输入信号
    :param h1: 一阶核函数
    :param h2: 二阶核函数
    :return: 输出响应
    """
    y = np.zeros_like(x)  # 初始化输出信号数组
    dt = t[1] - t[0]  # 计算时间步长

    for i in range(len(t)):
        # 计算一阶项的积分
        integral_h1 = 0
        for tau in t[:i + 1]:
            # 对输入信号在时间t[i] - tau处的值进行加权积分
            integral_h1 += h1(t[i] - tau) * x[int((tau - t[0]) / dt)] * dt

        # 计算二阶项的积分
        integral_h2 = 0
        for tau1 in t[:i + 1]:
            for tau2 in t[:i + 1]:
                # 对输入信号在时间t[i] - tau1和t[i] - tau2处的值进行加权积分
                integral_h2 += h2(t[i] - tau1, t[i] - tau2) * x[int((tau1 - t[0]) / dt)] * x[
                    int((tau2 - t[0]) / dt)] * dt * dt

        # 输出响应为一阶项和二阶项的和
        y[i] = integral_h1 + integral_h2

    return y

# 计算输出
y = volterra_output(t, x, h1, h2)

# 绘制输入和输出信号
plt.figure(figsize=(12, 6))
plt.plot(t, x, label='Input Signal x(t)')
plt.plot(t, y, label='Output Signal y(t)')
plt.xlabel('Time t')
plt.ylabel('Signal')
plt.legend()
plt.title('Volterra Series Response')

fig = plt.gcf()
fig.savefig('Volterra Series Response.png')

plt.show()