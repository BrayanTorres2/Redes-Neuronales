import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Crear el modelo
model = keras.Sequential([
    layers.Dense(1, input_shape=(3,), activation='sigmoid', use_bias=True)
])

# 2. Establecer pesos manualmente
# Pesos: limpieza=3, menú=6, sombrero=-3
# Sesgo: 5
pesos = np.array([[3.0], [6.0], [-3.0]])  # shape (3, 1)
sesgo = np.array([5.0])                  # shape (1,)

# Asignar pesos y sesgo a la capa
model.layers[0].set_weights([pesos, sesgo])

# 3. Entrada de prueba (limpieza, menú, sombrero)
entrada = np.array([[0.9, 0.3, 0.9]])

# 4. Predecir
salida = model.predict(entrada)

# 5. Mostrar resultado
print(f"Valor de activación: {salida[0][0]:.4f}")
if salida[0][0] >= 0.5:
    print("✅ Comer tacos: El restaurante parece una buena opción.")
else:
    print("❌ Mejor no comer tacos: Hay dudas sobre el restaurante.")
