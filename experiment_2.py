import numpy as np
import torch
import torch.nn as nn
from models.ca_model import CA
from models.q_network import DQNAgent
from sklearn.metrics import accuracy_score

def cargar_datos_aula():
    # Datos simulados de biología básica para una niña de 5 años
    animales = ['Perro', 'Gato', 'Elefante', 'León', 'Pájaro']
    plantas = ['Árbol', 'Flor', 'Hierba', 'Arbusto', 'Cactus']
    
    # Convertir a datos de entrada para el modelo
    datos = animales + plantas
    etiquetas = [0] * len(animales) + [1] * len(plantas)  # 0: Animal, 1: Planta
    
    # Convertir datos a tensores
    X = torch.randn(len(datos), 784)  # Datos simulados
    y = torch.tensor(etiquetas)
    
    return X, y

def experiment_luna_biologia(use_q_network=False):
    X, y = cargar_datos_aula()
    
    if use_q_network:
        agent = DQNAgent(input_dim=784, action_dim=2, hidden_dim=128, lr=0.001)
    else:
        model = CA()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterio = nn.MSELoss()

    for epoca in range(100):
        if use_q_network:
            for i in range(len(X)):
                state = X[i].unsqueeze(0)
                action = agent.select_action(state, epsilon=0.1)
                reward = 1 if action == y[i].item() else 0
                next_state = X[i].unsqueeze(0)
                done = torch.tensor([0]).float()
                agent.train_step(state, action, reward, next_state, done)
        else:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterio(outputs, y)
            loss.backward()
            optimizer.step()
    
    return model if not use_q_network else agent

# Cargar datos de aula
X_train, y_train = cargar_datos_aula()

# Dividir en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Ejecución del experimento en el aula de biología
agent = experiment_luna_biologia(use_q_network=True)

# Evaluación del modelo DQN
accuracy = sum([agent.select_action(X_test[i], epsilon=0.0) == y_test[i].item() for i in range(len(y_test))]) / len(y_test)
print("Precisión de la prueba en el aula de biología:", accuracy)