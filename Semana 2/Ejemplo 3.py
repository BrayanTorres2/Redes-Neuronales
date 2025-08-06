# Paso 1: Importar librerías
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Paso 2: Datos de entrada (ficticios para ejemplo)
X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
y = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)  # Relación lineal simple

# Paso 3: Crear modelo
model = Sequential([
    Dense(units=4, input_shape=[1], activation='relu'),   # Capa oculta con ReLU
    Dense(units=1)  # Capa de salida sin activación (regresión lineal)
])

# Paso 4: Compilar modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Paso 5: Entrenar modelo
model.fit(X, y, epochs=500, verbose=0)

# Paso 6: Predecir nuevos valores
print("\nPredicción para entrada 6:")
print(model.predict(np.array([[6.0]])))
