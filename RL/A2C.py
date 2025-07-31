import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 配置类，集中管理所有超参数
class Config:
    def __init__(self):
        # 神经网络结构参数
        self.hidden_dim = 128  # 共享网络隐藏层维度
        self.actor_hidden_dim = 64  # Actor网络隐藏层维度
        self.critic_hidden_dim = 64  # Critic网络隐藏层维度

        # 算法训练参数
        self.lr = 0.001  # 学习率
        self.gamma = 0.99  # 折扣因子，用于计算未来奖励的现值
        self.entropy_coef = 0.01  # 熵系数，用于鼓励探索
        self.value_loss_coef = 0.5  # 价值损失系数，平衡actor和critic损失

        # 训练过程参数
        self.max_episodes = 5000  # 最大训练回合数
        self.max_steps = 500  # 每个回合的最大步数

        # 其他设置
        self.seed = 42  # 随机种子，确保实验可重复性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
        self.rewards_history = []  # 用于存储每个episode的reward


# Actor-Critic网络结构
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ActorCritic, self).__init__()
        # 共享特征提取层
        self.common = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),  # 状态到特征的映射
            nn.ReLU()  # 非线性激活函数
        )
        # Actor网络：决策策略
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, action_dim),  # 特征到动作概率的映射
            nn.Softmax(dim=-1)  # 输出动作概率分布
        )
        # Critic网络：状态价值评估
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, 1)  # 特征到状态价值的映射
        )

    def forward(self, state):
        common_out = self.common(state)  # 提取共享特征
        policy = self.actor(common_out)  # 计算动作概率
        value = self.critic(common_out)  # 估计状态价值
        return policy, value


# A2C算法实现
class A2C:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.model = ActorCritic(state_dim, action_dim, config)  # 创建神经网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)  # Adam优化器
        self.gamma = config.gamma  # 折扣因子

    def compute_loss(self, states, actions, rewards, dones, next_states):
        # 数据预处理：转换为PyTorch张量
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # 获取当前状态的策略和价值估计
        policy, values = self.model(states)
        _, next_values = self.model(next_states)

        # 计算TD目标和优势函数
        # TD目标 = 即时奖励 + 折扣因子 * 下一状态的价值 * (1-终止标志)
        td_targets = rewards + self.gamma * next_values.squeeze(1) * (1 - dones)
        # 优势函数 = TD目标 - 当前状态的价值估计
        advantages = td_targets - values.squeeze(1)

        # 计算策略(Actor)损失
        action_probs = policy.gather(1, actions.unsqueeze(1)).squeeze(1)
        # 计算策略熵，用于鼓励探索
        entropy = -(policy * torch.log(policy + 1e-10)).sum(1).mean()
        # Actor损失 = -(log策略 * 优势函数) - 熵系数 * 熵
        actor_loss = -(torch.log(action_probs) * advantages.detach()).mean() - \
                     self.config.entropy_coef * entropy

        # Critic损失 = 优势函数的平方误差
        critic_loss = self.config.value_loss_coef * advantages.pow(2).mean()

        # 总损失 = Actor损失 + Critic损失
        return actor_loss + critic_loss

    def update(self, states, actions, rewards, dones, next_states):
        loss = self.compute_loss(states, actions, rewards, dones, next_states)
        self.optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新网络参数


# 训练主循环
def train_a2c():
    config = Config()  # 创建配置对象

    # 设置随机种子确保可重复性
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 创建CartPole环境
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 状态空间维度
    action_dim = env.action_space.n  # 动作空间维度

    # 创建A2C智能体
    agent = A2C(state_dim, action_dim, config)

    # 训练循环
    for episode in range(config.max_episodes):
        state, _ = env.reset()  # 重置环境
        # 初始化回合数据存储
        states, actions, rewards, next_states, dones = [], [], [], [], []
        total_reward = 0  # 记录回合总奖励
        episode_reward = 0  # 记录每个episode的reward

        # 单回合交互循环
        for step in range(config.max_steps):
            # 将状态转换为张量并添加批次维度
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # 获取动作概率分布
            policy, _ = agent.model(state_tensor)
            # 从概率分布中采样动作
            action = torch.multinomial(policy, 1).item()

            # 执行动作
            next_state, reward, done, _, _ = env.step(action)

            # 存储交互数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

            total_reward += reward  # 累积奖励
            episode_reward += reward  # 累积每个episode的reward
            state = next_state  # 更新状态

            if done:  # 如果回合结束，跳出循环
                break

        # 回合结束后更新模型
        agent.update(states, actions, rewards, dones, next_states)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # 记录每个episode的reward
        config.rewards_history.append(episode_reward)

        # 如果达到目标奖励，提前结束训练
        if total_reward >= 500:
            print("Solved!")
            break

    # 绘制训练奖励曲线
    plot_rewards(config.rewards_history)

    env.close()  # 关闭环境


# 绘图函数
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.show()


# 程序入口
if __name__ == "__main__":
    train_a2c()

