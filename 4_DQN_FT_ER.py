import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
from DQN_Net import DQNetwork

# DQN-FT-ER Агент
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min, memory_size, batch_size, target_update_freq):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_step = 0

        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# Навчання
def train_dqn(env_name, episodes, epochs, gamma, epsilon, epsilon_decay, lr, memory_size, batch_size, target_update_freq):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min=0.01,
                     memory_size=memory_size, batch_size=batch_size, target_update_freq=target_update_freq)

    rewards_history = []
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

        agent.decay_epsilon()

        rewards_history.append(np.mean(epoch_rewards))
        loss_history.append(np.mean(epoch_losses) if epoch_losses else 0)

        print(f"Epoch {epoch + 1}/{epochs}, Avg Reward: {np.mean(epoch_rewards):.2f}, Max Loss: {loss_history[-1]:.4f}")

    # Графіки
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Average Reward per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Reward')

    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title('Max Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Max Loss')

    plt.tight_layout()
    plt.savefig("1_DQN_FT_ER Graphs")

    return agent


agent = train_dqn(env_name="CartPole-v1", episodes=50, epochs=50, gamma=0.99, epsilon=1.0, 
                  epsilon_decay=0.995, lr=0.001, memory_size=25000, batch_size=16, target_update_freq=50)
