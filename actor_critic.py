import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # Red de actor
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # Red de crítico
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, input_dim, action_dim, hidden_dim, lr=0.001):
        self.model = ActorCritic(input_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def update(self, state, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Compute values
        action_probs, state_value = self.model(state)
        _, next_state_value = self.model(next_state)

        # Compute advantage
        advantage = reward + (1 - done) * next