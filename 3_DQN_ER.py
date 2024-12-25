import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
from DQN_Net import DQNetwork


# DQN Агент з повторенням досвіду
class DQNAgentER:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min, memory_size, batch_size):
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

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

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
            return

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

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# Навчання
def train_dqn_er(env_name, episodes, epochs, gamma, epsilon, epsilon_decay, lr, memory_size, batch_size, train_start):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgentER(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min=0.01, 
                       memory_size=memory_size, batch_size=batch_size)

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

                if len(agent.memory) > train_start:
                    loss = agent.train()
                    if loss is not None:
                        epoch_losses.append(loss)

                state = next_state
                total_reward += reward

            epoch_rewards.append(total_reward)

        agent.update_target_network()
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
    plt.savefig("1_DQN_ER Graphs")

    return agent


agent_er = train_dqn_er(env_name="CartPole-v1", episodes=50, epochs=50, gamma=0.99, 
                        epsilon=1.0, epsilon_decay=0.995, lr=0.1, 
                        memory_size=25000, batch_size=16, train_start=250)


# Тестування
def test_dqn(env_name, agent, test_episodes=100):
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


rewards_er, lengths_er, metrics_er = test_dqn(env_name="CartPole-v1", agent=agent_er, test_episodes=50)
