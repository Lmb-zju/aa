import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import os
from math import sqrt
from collections import deque
import random

# from DRC_StateTest2 import OpticalState


class OpticalState:
    def __init__(self, output_differences):
        """
        光学状态的专业表示
        """
        # 确保输入是NumPy数组
        self.raw_differences = np.array(output_differences, dtype=np.float32)

        # 数值稳定性处理
        min_val = np.maximum(np.min(self.raw_differences), 1e-8)

        # 计算相对强度比例
        self.relative_intensity = self.raw_differences / min_val - 1

        # 应用物理范围约束 [0, 8]
        self.relative_intensity = np.clip(self.relative_intensity, 0, 8)

        # 计算系统均匀性指标
        self.uniformity = 1 / (1 + np.std(self.relative_intensity))

    def get_vector(self):
        """获取状态向量 (用于神经网络输入)"""
        return self.relative_intensity

    # def is_valid(self):
    #     """验证状态是否物理可实现"""
    #     # 检查1: 所有差分值应为正
    #     # if np.any(self.raw_differences <= 0):
    #     #     return False
    #
    #     # 检查2: 相对强度在合理范围
    #     if np.any(self.relative_intensity < 0) or np.any(self.relative_intensity > 10):
    #         return False
    #
    #     # 检查3: 单调性 (输出应递增)
    #     # if not np.all(np.diff(self.raw_differences) >= -1e-6):  # 允许小的数值误差
    #     #     return False
    #
    #     return True
    def is_valid(self):
        # 检查1: 相对强度在合理范围
        if np.any(self.relative_intensity < 0) or np.any(self.relative_intensity > 8):
            return False
        return True


class OpticalEnvironment:
    def __init__(self, num_comb=20, lambda_val=0.1):
        self.num_comb = num_comb
        self.lambda_val = lambda_val

        # 物理约束参数
        self.MIN_DIFFERENCE = 0.01  # 最小可测量差分
        self.MAX_DIFFERENCE = 5.0  # 最大可测量差分

        # 初始化衰减参数
        self.P = np.ones(num_comb, dtype=np.float32)

        # 信道特性 (模拟真实光纤色散)
        self.channel_response = self._simulate_channel_response()

        # 记录历史数据用于可视化
        self.history = {
            'uniformity': [],
            'reward': [],
            'step': 0
        }

    def _simulate_channel_response(self):
        """模拟光纤信道的频率响应"""
        # 高斯型色散响应
        frequencies = np.linspace(0, 1, self.num_comb)
        response = np.exp(-0.5 * ((frequencies - 0.5) / 0.2) ** 2)
        return response / np.max(response)  # 归一化

    def reset(self):
        """重置环境到初始状态"""
        self.P = np.ones(self.num_comb, dtype=np.float32)
        self.history = {'uniformity': [], 'reward': [], 'step': 0}
        return self._get_state()

    def _get_output_differences(self):
        """计算相邻输出强度差 (考虑信道响应)"""
        # 实际输出 = 衰减设置 * 信道响应 + 噪声
        actual_output = self.P * self.channel_response
        actual_output += np.random.normal(0, 0.05, self.num_comb)  # 5%噪声

        # 创建21个输出点 (20个间隔)
        output_points = np.cumsum(np.insert(actual_output, 0, 0))

        # 计算相邻输出点的差值 (20个差值)
        differences = np.diff(output_points)

        # 应用物理约束
        differences = np.clip(differences, self.MIN_DIFFERENCE, self.MAX_DIFFERENCE)
        return differences

    def _get_state(self):
        """获取当前状态"""
        differences = self._get_output_differences()
        return OpticalState(differences)

    # def step(self, action):
    #     """
    #     执行动作并返回新状态和奖励
    #     """
    #     # 更新衰减参数
    #     P_new = self.P + self.lambda_val * action
    #     self.P = np.clip(P_new, 0.1, 2.0)  # 物理约束: 衰减系数在10%-200%
    #
    #     # 获取新状态
    #     new_state = self._get_state()
    #
    #     # 计算奖励 (考虑状态有效性)
    #     if new_state.is_valid():
    #         # 计算均方值时减去平均值作修正
    #         mse_prime = np.mean(np.square(new_state.get_vector()-np.mean(new_state.get_vector())))
    #         reward = 3 - 15 * sqrt(mse_prime)
    #     else:
    #         # 无效状态惩罚
    #         reward = -100
    #
    #     # 记录历史
    #     self.history['uniformity'].append(new_state.uniformity)
    #     self.history['reward'].append(reward)
    #     self.history['step'] += 1
    #
    #     return new_state, reward, False
    def step(self, action):
        # 更新衰减参数
        P_new = self.P + self.lambda_val * action
        self.P = np.clip(P_new, 0.1, 2.0)

        # 获取新状态
        new_state = self._get_state()

        # 计算奖励（直接奖励均匀性）
        if new_state.is_valid():
            reward = new_state.uniformity  # 匀性越高，奖励越大
        else:
            reward = -100  # 无效状态惩罚

        # 记录历史
        self.history['uniformity'].append(new_state.uniformity)
        self.history['reward'].append(reward)
        self.history['step'] += 1

        return new_state, reward, False

    def plot_history(self, save_path=None):
        """绘制训练历史"""
        plt.figure(figsize=(12, 8))

        # 均匀性变化
        plt.subplot(2, 1, 1)
        plt.plot(self.history['uniformity'], 'b-')
        plt.xlabel('Training Step')
        plt.ylabel('Uniformity')
        plt.title('System Uniformity During Training')
        plt.grid(True)

        # 奖励变化
        plt.subplot(2, 1, 2)
        plt.plot(self.history['reward'], 'g-')
        plt.xlabel('Training Step')
        plt.ylabel('Reward')
        plt.title('Reward During Training')
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


# class DRC_Agent:
#     def __init__(self, state_dim=20, action_dim=20):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.lr = 0.001
#
#         # 创建策略网络
#         self.actor = self.build_actor()
#
#         # 目标网络
#         self.target_actor = self.build_actor()
#         self.target_actor.set_weights(self.actor.get_weights())
#
#     def build_actor(self):
#         """构建四层全连接策略网络"""
#         inputs = Input(shape=(self.state_dim,))
#         x = Dense(64, activation='relu')(inputs)
#         x = Dense(64, activation='relu')(x)
#         x = Dense(64, activation='relu')(x)
#         outputs = Dense(self.action_dim, activation='tanh')(x)  # 输出范围[-1,1]
#
#         model = Model(inputs=inputs, outputs=outputs)
#         model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
#         return model
#
#     def act(self, state):
#         """根据状态选择动作"""
#         state = np.expand_dims(state, axis=0)
#         action = self.actor.predict(state, verbose=0)[0]
#         return action
#
#     def train_target_model(self):
#         """更新目标网络参数"""
#         self.target_actor.set_weights(self.actor.get_weights())
#
#     def save_weights(self, filepath):
#         self.actor.save_weights(filepath)
#
#     def load_weights(self, filepath):
#         self.actor.load_weights(filepath)
#         self.target_actor.load_weights(filepath)


class DRC_Agent:
    def __init__(self, state_dim=20, action_dim=20, buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.001
        self.buffer = deque(maxlen=buffer_size)

        # 策略网络（Actor）
        self.actor = self.build_actor()
        self.actor_optimizer = Adam(learning_rate=self.lr)
        self.actor.compile(optimizer=self.actor_optimizer, loss='mse')

        # 目标策略网络（Actor Target）
        self.actor_target = self.build_actor()
        self.actor_target.set_weights(self.actor.get_weights())

        # 价值网络（Critic）
        self.critic = self.build_critic()
        self.critic_optimizer = Adam(learning_rate=self.lr * 0.1)
        self.critic.compile(optimizer=self.critic_optimizer, loss='mse')

        # 目标价值网络（Critic Target）
        self.critic_target = self.build_critic()
        self.critic_target.set_weights(self.critic.get_weights())

        # 初始化 Critic 优化器
        self.critic_optimizer = Adam(learning_rate=self.lr * 0.1)

    def train_target_model(self):
        """更新目标网络参数"""
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

    def build_actor(self):
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='tanh')(x)
        return Model(inputs, outputs)

    def build_critic(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        x = tf.keras.layers.Concatenate()([state_input, action_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)
        return Model([state_input, action_input], outputs)

    def act(self, state, noise_scale=0.2):
        state = np.expand_dims(state, axis=0)
        action = self.actor.predict(state, verbose=0)[0]
        # 加入探索噪声
        action += np.random.normal(0, noise_scale, self.action_dim)
        action = np.clip(action, -1, 1)
        return action

    # def train(self, batch_size=32):
    #     if len(self.buffer) < batch_size:
    #         return
    #
    #     # 从经验回放中采样
    #     batch = random.sample(self.buffer, batch_size)
    #     states, actions, rewards, next_states = zip(*batch)
    #     states = np.array(states)
    #     actions = np.array(actions)
    #     rewards = np.array(rewards)
    #     next_states = np.array(next_states)
    #
    #     # 计算目标Q值
    #     next_actions = self.actor_target.predict(next_states)
    #     target_q_values = self.critic_target.predict([next_states, next_actions])
    #
    #     # 计算当前Q值
    #     with tf.GradientTape() as tape:
    #         q_values = self.critic([states, actions])
    #         target = rewards + 0.99 * target_q_values  # 折扣因子
    #         critic_loss = tf.reduce_mean(tf.square(q_values - target))
    #     grads = tape.gradient(critic_loss, self.critic.trainable_variables)
    #     self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
    #
    #     # 更新Actor网络
    #     with tf.GradientTape() as tape:
    #         actions_pred = self.actor(states)
    #         actor_loss = -self.critic([states, actions_pred])  # 最大化Q值
    #         actor_loss = tf.reduce_mean(actor_loss)
    #     grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    #     self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
    def train(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return

        # 从经验回放中采样
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        # 计算目标Q值
        next_actions = self.actor_target.predict(next_states)
        target_q_values = self.critic_target.predict([next_states, next_actions])

        # 计算当前Q值
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions])
            target = rewards + 0.99 * target_q_values  # 折扣因子
            critic_loss = tf.reduce_mean(tf.square(q_values - target))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))  # 使用 critic_optimizer

        # 更新Actor网络
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -self.critic([states, actions_pred])  # 最大化Q值
            actor_loss = tf.reduce_mean(actor_loss)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))  # 使用 actor_optimizer

    def save_weights(self, filepath):
        self.actor.save_weights(filepath)
        self.critic.save_weights(filepath + '_critic')

    def load_weights(self, filepath):
        self.actor.load_weights(filepath)
        self.critic.load_weights(filepath + '_critic')


# def generate_synthetic_data(num_samples=2000, state_range=[0, 8], num_comb=20):
#     """
#     生成合成训练数据
#     """
#     # 随机生成状态 (20维向量)
#     states = np.random.uniform(state_range[0], state_range[1], (num_samples, num_comb))
#
#     # 生成对应的动作（假设线性关系）
#     actions = states * 0.5 + np.random.normal(0, 0.01, states.shape)
#     actions = np.clip(actions, -1, 1)
#
#     return states, actions
def generate_synthetic_data(num_samples=2000, state_range=[0, 8], num_comb=20):
    """
    生成合成训练数据
    """
    # 随机生成状态 (20维向量)
    states = np.random.uniform(state_range[0], state_range[1], (num_samples, num_comb))

    # 生成对应的动作（假设线性关系）
    actions = states * 0.5 + np.random.normal(0, 0.01, states.shape)
    actions = np.clip(actions, -1, 1)

    return states, actions



# def pretrain_actor(agent, states, actions, epochs=200, batch_size=32):
#     """预训练策略网络"""
#     # 确保模型已编译
#     if not agent.actor.built:
#         agent.actor.build(input_shape=(agent.state_dim,))
#         agent.actor.compile(optimizer=Adam(learning_rate=agent.lr), loss='mse')
#
#     history = agent.actor.fit(states, actions,
#                               epochs=epochs,
#                               batch_size=batch_size,
#                               verbose=1,
#                               validation_split=0.2)
#     # 绘制预训练损失曲线
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Pretraining Loss History')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('pretrain_loss.png')
#     plt.close()
#     return history

# def drl_training(agent, env, episodes=100, max_steps_per_episode=50):
#     """深度强化学习训练循环"""
#     # 创建结果目录
#     os.makedirs('training_results', exist_ok=True)
#
#     # 训练进度条
#     episode_progress = tqdm(total=episodes, desc="DRL Training", unit="episode")
#
#     # 存储每轮的奖励和均匀性
#     episode_rewards = []
#     episode_uniformities = []
#
#     for episode in range(episodes):
#         state_obj = env.reset()
#         state = state_obj.get_vector()
#         total_reward = 0
#         done = False
#         step_count = 0
#
#         # 每轮内部进度条
#         step_progress = tqdm(total=max_steps_per_episode, desc=f"Episode {episode + 1}", leave=False)
#
#         while not done and step_count < max_steps_per_episode:
#             step_count += 1
#
#             # 选择动作（加入探索噪声）
#             action = agent.act(state) + np.random.normal(0, 0.01, env.num_comb)
#             action = np.clip(action, -1, 1)
#
#             # 执行动作
#             next_state_obj, reward, done = env.step(action)
#             next_state = next_state_obj.get_vector()
#
#             # 仅当状态有效时学习
#             if next_state_obj.is_valid():
#                 with tf.GradientTape() as tape:
#                     predicted_action = agent.actor(np.expand_dims(state, axis=0))
#                     loss = tf.reduce_mean(tf.square(predicted_action - action))
#                 grads = tape.gradient(loss, agent.actor.trainable_variables)
#                 agent.actor.optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))
#
#             state = next_state
#             total_reward += reward
#
#             # 更新内部进度条
#             step_progress.set_postfix(step_reward=f"{reward:.2f}", uniformity=f"{next_state_obj.uniformity:.4f}")
#             step_progress.update(1)
#
#         # 关闭内部进度条
#         step_progress.close()
#
#         # 记录本轮结果
#         episode_rewards.append(total_reward)
#         episode_uniformities.append(next_state_obj.uniformity)
#
#         # 定期更新目标网络
#         if episode % 10 == 0:
#             agent.train_target_model()
#
#             # 保存中间结果
#             env.plot_history(f'training_results/training_step_{env.history["step"]}.png')
#
#             # 绘制奖励和均匀性变化
#             plt.figure(figsize=(12, 6))
#
#             plt.subplot(1, 2, 1)
#             plt.plot(episode_rewards, 'g-o')
#             plt.title('Episode Rewards')
#             plt.xlabel('Episode')
#             plt.ylabel('Total Reward')
#             plt.grid(True)
#
#             plt.subplot(1, 2, 2)
#             plt.plot(episode_uniformities, 'b-o')
#             plt.title('Final Uniformity per Episode')
#             plt.xlabel('Episode')
#             plt.ylabel('Uniformity')
#             plt.ylim(0, 1)
#             plt.grid(True)
#
#             plt.tight_layout()
#             plt.savefig(f'training_results/episode_summary_{episode}.png')
#             plt.close()
#
#         # 更新外部进度条
#         episode_progress.set_postfix(reward=f"{total_reward:.2f}",
#                                      uniformity=f"{next_state_obj.uniformity:.4f}")
#         episode_progress.update(1)
#
#     # 关闭外部进度条
#     episode_progress.close()
#
#     return episode_rewards, episode_uniformities
def pretrain_actor(agent, states, actions, epochs=200, batch_size=32):
    """预训练策略网络"""
    # 确保模型已编译
    if not agent.actor.built:
        agent.actor.build(input_shape=(agent.state_dim,))
        agent.actor.compile(optimizer=Adam(learning_rate=agent.lr), loss='mse')

    history = agent.actor.fit(states, actions,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=1,
                              validation_split=0.2)
    return history

def drl_training(agent, env, episodes=100, max_steps_per_episode=50, buffer_size=10000):
    # 初始化经验回放缓冲区
    agent.buffer = deque(maxlen=buffer_size)

    for episode in range(episodes):
        state_obj = env.reset()
        state = state_obj.get_vector()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            action = agent.act(state, noise_scale=0.2)  # 增大探索噪声
            next_state_obj, reward, done = env.step(action)
            next_state = next_state_obj.get_vector()

            # 存储经验
            agent.buffer.append((state, action, reward, next_state))

            # 训练网络
            agent.train()

            state = next_state
            total_reward += reward

            step_count += 1



def analyze_compensation(agent, env, steps=10):
    """分析补偿效果"""
    # 重置环境
    state_obj = env.reset()
    initial_output = state_obj.raw_differences.copy()

    # 运行优化步骤
    for step in range(steps):
        state = state_obj.get_vector()
        action = agent.act(state)
        state_obj, _, _ = env.step(action)

    final_output = state_obj.raw_differences

    # 计算改善指标
    initial_uniformity = 1 / (1 + np.std(initial_output))
    final_uniformity = state_obj.uniformity
    improvement = (final_uniformity - initial_uniformity) / initial_uniformity * 100

    print(f"初始均匀性: {initial_uniformity:.4f}")
    print(f"最终均匀性: {final_uniformity:.4f}")
    print(f"改善率: {improvement:.2f}%")

    # 绘制补偿效果
    plt.figure(figsize=(10, 6))
    plt.plot(initial_output, 'ro-', label='补偿前')
    plt.plot(final_output, 'go-', label='补偿后')
    plt.xlabel('光梳齿索引')
    plt.ylabel('输出强度差')
    plt.title('色散补偿效果分析')
    plt.legend()
    plt.grid(True)
    plt.savefig('dispersion_compensation_result.png')
    plt.show()


# 主执行流程
if __name__ == "__main__":
    print("初始化光学环境和DRC代理...")
    env = OpticalEnvironment(num_comb=20, lambda_val=0.05)
    agent = DRC_Agent(state_dim=20, action_dim=20)
    print("Critic Optimizer exists:", hasattr(agent, 'critic_optimizer'))
    print("Actor Optimizer exists:", hasattr(agent, 'actor_optimizer'))

    # 检查属性是否存在
    print("Actor Target exists:", hasattr(agent, 'actor_target'))
    print("Critic Target exists:", hasattr(agent, 'critic_target'))

    print("生成合成训练数据...")
    train_states, train_actions = generate_synthetic_data(num_samples=2000)
    test_states, test_actions = generate_synthetic_data(num_samples=500)

    print("开始预训练策略网络...")
    pretrain_actor(agent, train_states, train_actions, epochs=50)

    # 评估预训练效果
    test_loss = agent.actor.evaluate(test_states, test_actions, verbose=0)
    print(f"预训练测试损失: {test_loss:.4f}")

    print("开始深度强化学习训练...")
    start_time = time.time()
    rewards, uniformities = drl_training(agent, env, episodes=100, max_steps_per_episode=30)
    training_time = time.time() - start_time

    print(f"训练完成! 总时间: {training_time:.2f}秒")

    # 保存训练好的模型
    agent.save_weights("drc_optical_compensation.h5")

    # 绘制最终训练结果
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, 'g-o')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(uniformities, 'b-o')
    plt.title('Final Uniformity per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Uniformity')
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('final_training_summary.png')
    plt.show()

    # 测试训练后的系统
    print("\n测试训练后的系统:")
    analyze_compensation(agent, env)

    # 保存最终训练历史
    env.plot_history('final_training_history.png')
