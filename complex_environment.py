import torch
import torch.nn as nn

class ComplexEnvironment(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ComplexEnvironment, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class BiologyClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiologyClass, self).__init__()
        self.env = ComplexEnvironment(input_dim, num_classes, hidden_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.env(x)

    def train_model(self, data_loader, optimizer):
        self.train()
        total_loss = 0
        for data, target in data_loader:
            optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)