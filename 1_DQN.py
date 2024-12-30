import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from DQN_Net import DQNetwork


# DQN Агент
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = []
        self.batch_size = 16

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)


    def train(self):
        if len(self.memory) < self.batch_size:
            return

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Обчислення Q-значень
        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# Навчання
def train_dqn(env_name, episodes, epochs, gamma, epsilon, epsilon_decay, lr):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min=0.01)

    rewards_history = []
    q_values_history = []
    loss_history = []

    for epoch in range(epochs):
        epoch_rewards = []
        epoch_losses = []
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.store_transition((state, action, reward, next_state, done))
                loss = agent.train()

                if loss is not None:
                    epoch_losses.append(loss)

                state = next_state
                total_reward += reward

            epoch_rewards.append(total_reward)
        
        # Валідація
        val_state, _ = env.reset()
        val_done = False
        val_q_values = []
        while not val_done:
            with torch.no_grad():
                q_values = agent.q_network(torch.FloatTensor(val_state).unsqueeze(0))
                val_q_values.append(q_values.max().item())
            val_state, _, val_done, _, _ = env.step(np.random.choice(action_dim))

        agent.update_target_network()
        agent.decay_epsilon()

        rewards_history.append(np.mean(epoch_rewards))
        q_values_history.append(np.mean(val_q_values))
        loss_history.append(max(epoch_losses) if epoch_losses else 0)

        print(f"Epoch {epoch + 1}/{epochs}, Avg Reward: {np.mean(epoch_rewards):.2f}, Avg Q: {np.mean(val_q_values):.2f}, Max Loss: {loss_history[-1]:.4f}")

    # Графіки
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Average Reward per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Reward')

    plt.subplot(1, 3, 2)
    plt.plot(q_values_history)
    plt.title('Average Q(s, a) per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Q(s, a)')

    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title('Max Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Max Loss')

    plt.tight_layout()
    plt.savefig("1_DQN Graphs")

    return agent


agent = train_dqn(env_name="CartPole-v1", episodes=150, epochs=100, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, lr=0.001)


# Тренування
def test_dqn(env_name, agent, test_episodes):
    env = gym.make(env_name)
    episode_rewards = []
    episode_lengths = []
    detailed_metrics = {}

    for episode in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        q_values = []
        rewards_per_step = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_value = agent.q_network(state_tensor).cpu().numpy()
            q_values.append(q_value)

            action = np.argmax(q_value)
            next_state, reward, done, _, _ = env.step(action)

            total_reward += reward
            rewards_per_step.append(reward)
            state = next_state
            step_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        detailed_metrics[episode + 1] = {
            "q_values": q_values,
            "rewards_per_step": rewards_per_step,
        }

    env.close()
    return episode_rewards, episode_lengths, detailed_metrics


test_rewards, test_lengths, test_detailed_metrics = test_dqn(env_name="CartPole-v1", agent=agent, test_episodes=100)


random_episode = random.randint(1, 99)
selected_episodes = [1, random_episode, 99]

for episode in selected_episodes:
    q_values_steps = np.array(test_detailed_metrics[episode]["q_values"]).squeeze()
    rewards_steps = test_detailed_metrics[episode]["rewards_per_step"]

    # Графік Q(s, a)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for action_idx in range(q_values_steps.shape[1]):
        plt.plot(q_values_steps[:, action_idx], label=f"Q(s, a={action_idx})")
    plt.xlabel("Step")
    plt.ylabel("Q(s, a)")
    plt.title(f"Q-Values during Episode {episode}")
    plt.legend()
    plt.grid()

    # Графік винагороди
    plt.subplot(2, 1, 2)
    plt.plot(rewards_steps, label="Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Rewards per Step during Episode {episode}")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"2_DQN during Episode {episode}")
