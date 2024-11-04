import torch.nn as nn

class EntornoAprendizaje(nn.Module):
    def __init__(self):
        super(EntornoAprendizaje, self).__init__()
        # Cadenas de texto
        self.cadena = nn.Linear(1000, 10)

    def forward(self, x, y):
        # Cadenas de texto
        outputs = self.cadena(x)
        return outputs