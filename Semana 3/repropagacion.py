import numpy as np

# Datos de ejemplo (X = entrada, y = salida deseada)
X = np.array([0.5, 0.8])     # características
y = 1                        # salida esperada

# Pesos iniciales y tasa de aprendizaje
w = np.array([0.4, 0.6])
lr = 0.1

# Paso 1: Forward (predicción)
y_pred = np.dot(X, w)        # salida = sumatoria(X * w)

# Paso 2: Calcular error (loss = (y - y_pred)^2)
error = y - y_pred

# Paso 3: Retropropagación (gradiente)
gradiente = -2 * X * error

# Paso 4: Actualizar pesos
w = w - lr * gradiente

print("Predicción:", y_pred)
print("Error:", error)
print("Nuevos pesos:", w)
