# Paso 1: Importar librerías
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Paso 2: Datos de ejemplo con positivos y negativos
X = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
y = np.array([[-1], [-0.5], [0], [0.5], [1]], dtype=float)  # Relación lineal

# Paso 3: Crear modelo
model = Sequential([
    Dense(units=4, input_shape=[1], activation='tanh'),  # Activación tanh en capa oculta
    Dense(units=1)  # Capa de salida sin activación
])

# Paso 4: Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Paso 5: Entrenar el modelo
model.fit(X, y, epochs=500, verbose=0)

# Paso 6: Hacer una predicción
print("\nPredicción para entrada 1.5:")
print(model.predict(np.array([[1.5]])))