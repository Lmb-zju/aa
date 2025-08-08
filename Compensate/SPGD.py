import random
import numpy as np


class SPGDController:
    def __init__(self, learning_rate, A, action_size):
        self.learning_rate = learning_rate  # 学习率α
        self.A = A  # 扰动强度
        self.action = np.zeros(action_size)  # 初始动作向量

    def generate_random_noise(self):
        """生成随机扰动向量，元素为±A"""
        return np.array([random.choice([-self.A, self.A]) for _ in range(len(self.action))])

    def update_action(self, delta_reward, random_noise):
        """根据奖励差更新动作"""
        self.action += self.learning_rate * delta_reward * random_noise


class System:
    def __init__(self, dynamic_model):
        self.state = None
        self.dynamic_model = dynamic_model  # 系统动态模型函数

    def apply_action(self, action):
        """应用动作并获取新状态"""
        self.state = self.dynamic_model(action)
        return self.state.copy()


# 示例配置
def system_dynamics(action):
    """示例系统动态模型：状态=action + 噪声，目标需要自行定义"""
    noise = np.random.normal(0, 0.1, len(action))
    return action + noise  # 状态=动作+噪声


def reward_function(state, target):
    """示例奖励函数：负的平方误差"""
    return -((np.sum((state - target) ** 2)) / len(state)) ** 0.5


def check_convergence(state, target, tolerance):
    """收敛判断：所有维度误差小于容差"""
    return np.all(np.abs(state - target) < tolerance)

if __name__ == '__main__':

    # 参数初始化
    TARGET_STATE = np.array([1.0, 2.0, 3.0,4.0, 5.0, 6.0, 7.0, 8.0, 9.0])  # 目标状态
    TARGET_STATE = [i/9 for i in TARGET_STATE]
    ACTION_DIM = 9  # 动作维度
    LEARNING_RATE = 0.2  # 学习率α
    DISTURBANCE = 1  # 扰动强度A
    MAX_ITER = 10000
    TOLERANCE = 0.1

    # 初始化组件
    controller = SPGDController(LEARNING_RATE, DISTURBANCE, ACTION_DIM)
    system = System(system_dynamics)
    history = []

    # 主控制循环
    for iter in range(MAX_ITER):
        # 生成随机扰动
        noise = controller.generate_random_noise()

        # 正向扰动评估
        pos_action = controller.action + noise
        pos_state = system.apply_action(pos_action)
        pos_reward = reward_function(pos_state, TARGET_STATE)

        # 负向扰动评估
        neg_action = controller.action - noise
        neg_state = system.apply_action(neg_action)
        neg_reward = reward_function(neg_state, TARGET_STATE)

        # 更新控制动作
        delta_reward = pos_reward - neg_reward
        controller.update_action(delta_reward, noise)

        # 应用新动作并记录
        current_state = system.apply_action(controller.action)
        history.append(current_state.copy())

        # 收敛判断
        if check_convergence(current_state, TARGET_STATE, TOLERANCE):
            print(f"Converged after {iter + 1} iterations")
            print(f"Final state: {np.round(current_state, 4)}")
            break
        else:
            print(f"未收敛，最后状态: {np.round(current_state, 4)}")

    # 可视化收敛过程（可选）
    import matplotlib.pyplot as plt

    plt.plot(np.array(history) - TARGET_STATE)
    plt.title("Convergence Process")
    plt.xlabel("Iterations")
    plt.ylabel("State Error")
    plt.show()

    #扩展
    #动态模型改进：
    # def advanced_dynamics(action):
    #     """含惯性的动态模型示例"""
    #     global prev_state
    #     new_state = 0.8*prev_state + 0.2*action
    #     prev_state = new_state
    #     return new_state + np.random.normal(0, 0.05, len(action))

    #自适应参数调整：
    # 在SPGDController类中添加
    # def adapt_parameters(self, reward_diff):
    #     """根据奖励变化自动调整参数"""
    #     self.learning_rate *= 1.01 if abs(reward_diff) < 0.1 else 0.99
    #     self.A = max(0.01, self.A * (0.95 if abs(reward_diff) < 0.2 else 1.05))
