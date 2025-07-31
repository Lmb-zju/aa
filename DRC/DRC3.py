import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class OpticalEnvironment:
    def __init__(self, num_comb=20, lambda_val=0.1):
        self.num_comb = num_comb
        self.lambda_val = lambda_val  # 动作缩放系数
        self.state_range = [0, 8]  # 状态值范围

        # 初始化衰减参数 (全1表示无衰减)
        self.P = np.ones(num_comb, dtype=np.float32)

    def reset(self):
        """重置环境到初始状态"""
        self.P = np.ones(self.num_comb, dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        """根据当前衰减计算状态"""
        # 模拟系统输出 (理想情况应为线性增加)
        output = np.linspace(0, 1, self.num_comb + 1)
        # 计算相邻输出差值
        S_raw = np.diff(output)
        # 归一化并转换到状态空间
        S = S_raw / np.min(S_raw) - 1
        return S

    def step(self, action):
        """
        执行动作并返回新状态和奖励
        :param action: 策略网络输出的动作向量
        :return: (新状态, 奖励, 是否完成)
        """
        # 更新衰减参数
        P_new = self.P + self.lambda_val * action
        self.P = P_new - np.min(P_new)  # 确保非负

        # 获取新状态
        new_state = self._get_state()

        # 计算奖励
        reward = self._calculate_reward(new_state)

        # 总是返回未完成状态（持续优化）
        return new_state, reward, False

    def _calculate_reward(self, state):
        """计算奖励函数: 3 - 15 * MSE(S)"""
        mse = np.mean(np.square(state))
        return 3 - 15 * mse


class DRC_Agent:
    def __init__(self, state_dim=20, action_dim=20):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.001  # 学习率

        # 创建策略网络
        self.actor = self.build_actor()

        # 目标网络（用于稳定训练）
        self.target_actor = self.build_actor()
        self.target_actor.set_weights(self.actor.get_weights())

    def build_actor(self):
        """构建四层全连接策略网络"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='tanh')(x)  # 输出范围[-1,1]

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
        return model

    def act(self, state):
        """根据状态选择动作"""
        state = np.expand_dims(state, axis=0)
        action = self.actor.predict(state, verbose=0)[0]
        return action

    def train_target_model(self):
        """更新目标网络参数"""
        self.target_actor.set_weights(self.actor.get_weights())

    def save_weights(self, filepath):
        """保存模型权重"""
        self.actor.save_weights(filepath)

    def load_weights(self, filepath):
        """加载模型权重"""
        self.actor.load_weights(filepath)
        self.target_actor.load_weights(filepath)


def generate_synthetic_data(num_samples=2000, state_range=[0, 8]):
    """
    生成合成训练数据
    :param num_samples: 样本数量
    :return: (states, actions) 元组
    """
    # 随机生成状态 (20维向量)
    states = np.random.uniform(state_range[0], state_range[1], (num_samples, 20))

    # 生成对应的动作（假设线性关系：动作 = 状态 * 0.5 + 噪声）
    actions = states * 0.5 + np.random.normal(0, 0.1, states.shape)

    # 将动作裁剪到[-1,1]范围
    actions = np.clip(actions, -1, 1)

    return states, actions


def pretrain_actor(agent, states, actions, epochs=200, batch_size=32):
    """
    预训练策略网络
    :param agent: DRC代理
    :param states: 训练状态
    :param actions: 对应动作
    :param epochs: 训练轮数
    :param batch_size: 批大小
    """
    agent.actor.fit(states, actions,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split=0.2)


def drl_training(agent, env, episodes=1000):
    """
    深度强化学习训练循环
    :param agent: DRC代理
    :param env: 光学环境
    :param episodes: 训练回合数
    """
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 选择动作（加入探索噪声）
            action = agent.act(state) + np.random.normal(0, 0.1, env.num_comb)
            action = np.clip(action, -1, 1)

            # 执行动作
            next_state, reward, done = env.step(action)

            # 更新策略网络（简化版）
            # 注：完整DRL应使用经验回放和critic网络
            with tf.GradientTape() as tape:
                predicted_action = agent.actor(np.expand_dims(state, axis=0))
                loss = tf.reduce_mean(tf.square(predicted_action - action))
            grads = tape.gradient(loss, agent.actor.trainable_variables)
            agent.actor.optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))

            state = next_state
            total_reward += reward

        # 定期更新目标网络
        if episode % 10 == 0:
            agent.train_target_model()

        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}")


# ===================== 执行流程 =====================
if __name__ == "__main__":
    # 初始化环境和代理
    env = OpticalEnvironment(num_comb=20, lambda_val=0.1)
    agent = DRC_Agent(state_dim=20, action_dim=20)

    # 生成合成数据
    train_states, train_actions = generate_synthetic_data(num_samples=2000)
    test_states, test_actions = generate_synthetic_data(num_samples=500)

    # 预训练策略网络
    print("开始预训练策略网络...")
    pretrain_actor(agent, train_states, train_actions)

    # 评估预训练效果
    test_loss = agent.actor.evaluate(test_states, test_actions, verbose=0)
    print(f"预训练测试损失: {test_loss:.4f}")

    # 深度强化学习训练
    print("开始深度强化学习训练...")
    drl_training(agent, env, episodes=1000)

    # 保存训练好的模型
    agent.save_weights("drc_actor_weights.h5")
    print("训练完成! 模型权重已保存")
