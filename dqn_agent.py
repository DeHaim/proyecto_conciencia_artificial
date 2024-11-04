import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_dim, action_dim, hidden_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_network = QNetwork(input_dim, action_dim, hidden_dim)
        self.target_network = QNetwork(input_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Tasa de exploración inicial
        self.epsilon_min = epsilon_min  # Tasa de exploración mínima
        self.epsilon_decay = epsilon_decay  # Decaimiento de la tasa de exploración
        self.memory = deque(maxlen=2000)  # Memoria de experiencia
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.fc3.out_features - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.q_network(torch.FloatTensor(state))).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            action = torch.LongTensor([action])
            done = torch.FloatTensor([done])

            # Q-Learning Target: y = r + γ * max Q(s', a')
            q_values = self.q_network(state)
            next_q_values = self.target_network(next_state)
            q_value = q_values.gather(0, action).squeeze(0)
            next_q_value = reward + self.gamma * torch.max(next_q_values) * (1 - done)

            # Loss
            loss = self.criterion(q_value, next_q_value.detach())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.update_target_network()