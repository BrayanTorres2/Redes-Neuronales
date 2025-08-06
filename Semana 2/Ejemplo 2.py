# Paso 1: Importar librerías
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Paso 2: Datos de entrada (XOR simplificado para ejemplo)
X = np.array([[0], [1]], dtype=float)
y = np.array([[0], [1]], dtype=float)

# Paso 3: Crear modelo
model = Sequential([
    Dense(units=1, input_shape=[1], activation='sigmoid')  # Activación sigmoide
])

# Paso 4: Compilar modelo
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Paso 5: Entrenar
model.fit(X, y, epochs=1000, verbose=0)

# Paso 6: Predecir
print("\nPredicciones:")
print(model.predict(X))
