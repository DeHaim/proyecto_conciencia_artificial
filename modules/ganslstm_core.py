import torch
import torch.nn as nn
from ruth_full_module_system import TensorHub

class GANSLSTMCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :])

    def generate_tensor(self, input_text):
        input_tensor = torch.randn(1, 10, 64)  # Simulación por ahora
        output = self.forward(input_tensor)
        return TensorHub.register("GANSLSTMCore", output.detach())