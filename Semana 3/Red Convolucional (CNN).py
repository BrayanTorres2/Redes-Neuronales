# ============================
# Ejemplo 2: Red Convolucional (CNN)
# Dataset: MNIST (imágenes de dígitos 28x28)
# ============================

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Cargar dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X son las imágenes y y son las etiquetas (dígitos).
# 2. Preprocesar (normalizar y redimensionar)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
# Redimensiona las imágenes de entrenamiento para que tengan un canal (escala de grises) y normaliza los valores de píxeles entre 0 y 1.

X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
# Redimensiona y normaliza las imágenes de prueba de la misma forma.

y_train = to_categorical(y_train, num_classes=10)
# Convierte las etiquetas de entrenamiento en vectores one-hot (10 clases, dígitos 0-9).

y_test = to_categorical(y_test, num_classes=10)
# Convierte las etiquetas de prueba en vectores one-hot.

# 3. Definir modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
     # Capa convolucional con 32 filtros de tamaño 3x3 y función de activación ReLU. 
    # input_shape indica el tamaño de las imágenes de entrada.
    MaxPooling2D((2,2)),
    # Capa de pooling que reduce la dimensionalidad tomando el valor máximo en ventanas de 2x2.

    Flatten(),
    # Convierte la matriz 2D resultante en un vector 1D para conectar con las capas densas.

    Dense(64, activation='relu'),
     # Capa densa (fully connected) con 64 neuronas y activación ReLU.

    Dense(10, activation='softmax')  # 10 clases (0-9)
     # Capa de salida con 10 neuronas (una por cada dígito) y activación softmax para clasificación multiclase.
])


# 4. Compilar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Configura el modelo: usa el optimizador Adam, la función de pérdida categorical_crossentropy y la métrica de precisión.
model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)

# 5. Evaluar
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Accuracy en test (CNN - MNIST): {acc:.2f}")
