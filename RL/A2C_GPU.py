import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast


class Config:
    def __init__(self):
        self.hidden_dim = 128
        self.actor_hidden_dim = 64
        self.critic_hidden_dim = 64
        self.lr = 0.001
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_episodes = 5000
        self.max_steps = 500
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.rewards_history = []


# class DRCEnv(gym.Env):
#     def __init__(self, lambda_val=0.1, noise_level=0.1):
#         super(DRCEnv, self).__init__()
#         self.lambda_val = lambda_val
#         self.noise_level = noise_level
#         self.P = np.random.uniform(0, 8, (20,1))
#         self.action_space = spaces.Box(low=-1, high=1, shape=(20,1), dtype=np.float32)
#         self.observation_space = spaces.Box(low=-1, high=1, shape=(19,1), dtype=np.float32)
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.P = np.random.uniform(0, 8, (20, 1))
#         output = self.P.copy()
#         S_initial = [output[i+1] - output[i] for i in range(19)]
#         S_initial = np.array(S_initial)
#         S_initial = S_initial / np.max(S_initial) - 1
#         return S_initial
#
#     def step(self, action):
#         P_update = (self.P + self.lambda_val * action) - np.min(self.P + self.lambda_val * action)
#         self.P = P_update
#         output = self.P.copy()
#         S_new = [output[i+1] - output[i] for i in range(19)]
#         S_new = np.array(S_new)
#         S_new = S_new / np.max(S_new) - 1
#         S_new += np.random.normal(0, self.noise_level, 19)
#
#         mse_S = np.mean((S_new - np.mean(S_new)) ** 2)
#         reward = 3 - 15 * mse_S
#         done = reward > 0
#         return S_new, reward, done, {}
#
#     def render(self, mode='human'):
#         pass
class DRCEnv(gym.Env):
    def __init__(self, lambda_val=0.1, noise_level=0.1):
        super(DRCEnv, self).__init__()
        self.lambda_val = lambda_val
        self.noise_level = noise_level
        self.P = np.random.uniform(0, 8, (20, 1))
        self.action_space = spaces.Box(low=-1, high=1, shape=(20, 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19, 1), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.P = np.random.uniform(0, 8, (20, 1))
        output = self.P.copy()
        S_initial = [output[i+1][0] - output[i][0] for i in range(19)]  # 修正为访问 [0]
        S_initial = np.array(S_initial)
        S_initial = S_initial / np.max(S_initial) - 1
        return S_initial.flatten()  # 转换为一维数组

    def step(self, action):
        P_update = (self.P + self.lambda_val * action) - np.min(self.P + self.lambda_val * action)
        self.P = P_update
        output = self.P.copy()
        S_new = [output[i+1][0] - output[i][0] for i in range(19)]  # 修正为访问 [0]
        S_new = np.array(S_new)
        S_new = S_new / np.max(S_new) - 1
        S_new += np.random.normal(0, self.noise_level, 19)

        mse_S = np.mean((S_new - np.mean(S_new)) ** 2)
        reward = 3 - 15 * mse_S
        done = reward > 0
        return S_new.flatten(), reward, done, {}  # 转换为一维数组


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ActorCritic, self).__init__()
        self.device = config.device

        self.common = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU()
        ).to(self.device)

        self.actor_mean = nn.Sequential(
            nn.Linear(config.hidden_dim, action_dim),
            nn.Tanh()
        ).to(self.device)
        self.actor_std = nn.Parameter(torch.zeros(1, action_dim))  # 标准差参数

        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, 1)
        ).to(self.device)

    def forward(self, state):
        state_T = state.reshape(1, 19)
        common_out = self.common(state_T)
        mean = self.actor_mean(common_out)
        std = self.actor_std.expand_as(mean).exp()  # 标准差
        return mean, std, self.critic(common_out)


class A2C:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.model = ActorCritic(state_dim, action_dim, config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.gamma = config.gamma
        self.scaler = GradScaler()

    def compute_loss(self, states, actions, rewards, dones, next_states, log_probs):
        states = states.to(self.config.device)
        next_states = next_states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device)
        dones = dones.to(self.config.device)
        log_probs = log_probs.to(self.config.device)

        mean, std, values = self.model(states)
        _, _, next_values = self.model(next_states)

        td_targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = td_targets - values

        dist = torch.distributions.Normal(mean, std)
        actor_loss = -advantages * dist.log_prob(actions).mean(dim=1)  # 使用 log_prob
        actor_loss = actor_loss.mean()
        critic_loss = (td_targets - values).pow(2).mean()

        return actor_loss + self.config.value_loss_coef * critic_loss

    def update(self, states, actions, rewards, dones, next_states):
        with autocast():
            loss = self.compute_loss(states, actions, rewards, dones, next_states)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.show()

def train_a2c():
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = DRCEnv(lambda_val=0.1, noise_level=0.1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = A2C(state_dim, action_dim, config)
    agent.model = agent.model.to(config.device)

    for episode in range(config.max_episodes):
        state = env.reset()
        states, actions, log_probs, rewards, dones, next_states = [], [], [], [], [], []
        total_reward = 0

        for step in range(config.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(config.device)
            with torch.no_grad():
                mean, std, _ = agent.model(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().cpu().numpy()
                log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float32, device=config.device)).detach()

            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            total_reward += reward
            state = next_state

            if done:
                break

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        # log_probs = torch.tensor(log_probs)
        log_probs = torch.tensor([item.cpu().detach().numpy() for item in log_probs]).cuda()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        agent.update(states, actions, rewards, dones, next_states, log_probs)
        # agent.update(states, actions, rewards, dones, next_states)

        config.rewards_history.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        if total_reward >= 500:
            print("Solved!")
            break


if __name__ == "__main__":
    train_a2c()
