# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

# 定义电磁问题的参数
frequency = 300e6  # 频率，单位：Hz
wavelength = 3e8 / frequency  # 波长，单位：m


# 假设的设计变量
design_variable = np.random.rand()

# 目标透射系数
T_target = 0.8

# 计算当前设计的透射系数（这只是一个简化的示例）
def compute_transmission(design_var):
    return design_var

# 计算目标函数
def objective_function(design_var):
    T = compute_transmission(design_var)
    return (T - T_target)**2

# 计算梯度
def compute_gradient(design_var):
    # 这里我们使用一个简化的伴随方程来计算梯度
    return 2 * (compute_transmission(design_var) - T_target)


if __name__ == '__main__':
    # 梯度下降法
    learning_rate = 0.1
    num_iterations = 100

    # for i in range(num_iterations):
    #     gradient = compute_gradient(design_variable)
    #     design_variable -= learning_rate * gradient
    #     if i % 10 == 0:
    #         print(f"Iteration {i}: Objective = {objective_function(design_variable)}")

    # 存储每次迭代的目标函数值
    objectives = []

    # 修改之前的梯度下降代码，以存储每次迭代的目标函数值
    for i in range(num_iterations):
        gradient = compute_gradient(design_variable)
        design_variable -= learning_rate * gradient
        objectives.append(objective_function(design_variable))

    # 在Jupyter中绘制目标函数值随迭代次数的变化
    plt.figure(figsize=(10, 6))
    plt.plot(objectives)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Convergence of the Design Process')
    plt.grid(True)
    plt.show()

