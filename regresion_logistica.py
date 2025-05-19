import numpy as np

# Cada muestra es un vector con 4 features + 1 label (último valor)
datos = np.array([
    [20, 6, 0, 2, 0.12],
    [30, 4, 1, 3, 0.60],
    [25, 5, 1, 3, 0.80],
    [40, 2, 1, 5, 0.95],
    [22, 7, 0, 5, 0.05],
])

X = datos[:, :4]   # Features
y = datos[:, 4]    # Labels


X = X / np.max(X, axis=0) 
#Se divide cada columna por su valor máximo para que todas las características queden en el rango [0,1].
#Esto es crucial para que el entrenamiento sea más estable y rápido.



# Inicializar pesos
W = np.zeros(4) # Pesos inicializados a 0
b = 0.0 # Bias inicializado a 0
lr = 0.1  # Learning rate es 0,1 que es un valor común para la tasa de aprendizaje en algoritmos de optimización.

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z)) # Función sigmoide para la activación

# Entrenamiento
for epoch in range(1000):
    for i in range(len(X)):
        z = np.dot(W, X[i]) + b # Producto punto entre los pesos y la muestra + bias
        pred = sigmoid(z) # Predicción
        error = y[i] - pred # Error entre la predicción y el valor real
        
        # Gradiente y actualización
        W += lr * error * X[i] # Actualización de pesos
        b += lr * error # Actualización de bias

# Resultado final


print("Pesos entrenados:", W)
print("Bias entrenado:", b)


# Predicción

muestra_nueva = np.array([28, 5, 1, 3])  
muestra_nueva = muestra_nueva / np.max(X, axis=0) # Nueva muestra normalizada

muestra_nueva @ W + b # Producto punto entre la nueva muestra y los pesos + bias

print(sigmoid(muestra_nueva @ W + b))
