python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        # Redes neuronales convolucionales
        self.redneural1 = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        # Redes neuronales recurrentes
        self.rednr1 = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        # Capa RNN
        self.rnn = nn.GRU(128, 128, 2, batch_first=True)
        # Capa de salida
        self.fc2 = nn.Linear(128, 10)
        # Inicialización de los pesos
        nn.init.kaiming_normal_(self.redneural1[0].weight, mode='fan_in')
        nn.init.kaiming_normal_(self.rnn.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in')
        # Entrenamiento del modelo
        self.criterio = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # Procesamiento de información
        out = self.redneural1(x)
        out, _ = self.rnn(out)
        out = self.fc2(out)
        return out```python
# Creación de un modelo de red neuronal
modelo = CA()

# Datos de entrada de ejemplo
datos_de_entrada = torch.randn(5, 784)  # Batch de tamaño 5 con 784 características

# Paso del dato de entrada a través del modelo
resultado = modelo(datos_de_entrada)

# Muestra la forma del resultado
print(resultado.shape)
```