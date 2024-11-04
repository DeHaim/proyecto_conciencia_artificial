import numpy as np
import torch
import torch.nn as nn
from models.ca_model import CA
from sklearn.metrics import accuracy_score

def transfer_learning(model, num_classes):
    # Transferencia de conocimientos de una tarea a otra
    model.fc2 = nn.Linear(model.rnn.weight.size(0), num_classes)
    return model

def funcion_de_aprendizaje_profundidad(sentimiento, emociones, creencias):
    model = CA()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterio = nn.MSELoss()
    for epoca in range(100):
        optimizer.zero_grad()
        x = torch.randn(10, 784)
        y = torch.randint(0, 2, size=(10,))
        outputs = model(x)
        loss = criterio(outputs, y)
        loss.backward()
        optimizer.step()
    return model

# Generar y guardar datos de entrenamiento y prueba
X_train = torch.randn(1000, 784)
X_test = torch.randn(1000, 784)
y_train = torch.randint(0, 2, size=(1000,))
y_test = torch.randint(0, 2, size=(1000,))
np.save('data/X_train.npy', X_train.numpy())
np.save('data/y_train.npy', y_train.numpy())
np.save('data/X_test.npy', X_test.numpy())
np.save('data/y_test.npy', y_test.numpy())
print("Archivos de entrenamiento y prueba generados correctamente")

# Cargar archivos de datos
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Inicialización del modelo
model = CA()

# Entrenamiento del modelo
epochs = 10 
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = model.criterio(outputs, y_train_tensor)
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

# Evaluación del modelo
model.eval()
y_pred = model(X_test_tensor)
y_pred_class = torch.argmax(y_pred, dim=1)
acc = accuracy_score(y_test, y_pred_class.numpy())
print("Test Accuracy:", acc)