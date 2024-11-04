import torch.nn as nn
from environments.simple_environment import EntornoAprendizaje

class SistemaDeEvaluacion(nn.Module):
    def __init__(self):
        super(SistemaDeEvaluacion, self).__init__()
        # Entorno de aprendizaje
        self.entornoaprendizaje = EntornoAprendizaje()

    def forward(self, x, y):
        # Evaluación del rendimiento del sistema
        outputs = self.entornoaprendizaje(x, y)
        return outputs