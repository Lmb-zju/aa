import matplotlib.pyplot as plt
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, target_state):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_state = target_state
        self.n = len(target_state)
        self.last_error = [0.0] * self.n
        self.integral_error = [0.0] * self.n

    def compute_action(self, current_state):
        error = [current_state[i] - self.target_state[i] for i in range(self.n)]            # 误差现态
        derivative_error = [error[i] - self.last_error[i] for i in range(self.n)]           # 误差微分（离散差）
        self.integral_error = [self.integral_error[i] + error[i] for i in range(self.n)]    # 误差积分（离散和）
        action = [
            self.kp * error[i] + self.ki * self.integral_error[i] + self.kd * derivative_error[i]
            for i in range(self.n)                                                          # 更新action向量
        ]
        self.last_error = error.copy()
        return action


def system_model(action, current_state):
    """示例系统模型：当前状态减去动作（模拟负反馈）"""
    return [current_state[i] - action[i] for i in range(len(action))]


def check_convergence(current, target, tolerance=0.01):
    return all(abs(c - t) <= tolerance for c, t in zip(current, target))

if __name__ == '__main__':
    # 初始化参数
    kp = 0.01  # 需根据实际系统调整参数
    ki = 0.05
    kd = 0.5
    target_state_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]   # 目标状态（多维示例）
    target_state = [i/9 for i in target_state_list]                     # 归一化
    current_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 初始状态
    current_state = np.random.rand(9)                               # 随机初始化
    max_iterations = 1000
    tolerance = 0.01

    # 初始化PID控制器
    pid = PIDController(kp, ki, kd, target_state)
    for i in range(current_state.size):
        locals()['current_state_'+str(i)+'_copy'] = [current_state[i]] # 状态向量化，添加新的状态便于绘制训练曲线

    # 控制循环
    for iteration in range(1, max_iterations + 1):
        action = pid.compute_action(current_state)
        current_state = system_model(action, current_state) # 更新状态

        print(f"Iteration {iteration}: Current state = {current_state}, Action = {action}")
        for i in range(len(current_state)): # 因为system model返回的是列表，所以循环的时候size(计算ndarray1D向量长度)要改为len
            locals()['current_state_'+str(i)+'_copy'].append(current_state[i]) # 记录状态轨迹，方便可视化训练过程

        if check_convergence(current_state, target_state, tolerance):
            print(f"\nSystem converged to target state {target_state} after {iteration} iterations.")
            convergence_iteration = iteration
            break
        else:
            print("\nSystem did not converge within the maximum iterations.")

    # 可视化(未绘制action)
    for i in range(len(current_state)):
        plt.plot(locals()['current_state_'+str(i)+'_copy'], label='state['+str(i)+']')
    plt.legend()
    plt.title("Convergence Process")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.show()

