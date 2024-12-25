import torch.nn as nn

# Глибинна Q-Мережа
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 60)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(60, 45)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(45, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.out(x)
