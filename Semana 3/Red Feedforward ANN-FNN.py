# ============================
# Ejemplo 1: Red Feedforward (ANN)
# Dataset: Iris (clasificación de flores)
# ============================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. Cargar dataset Iris
iris = load_iris()
X = iris.data      # 4 características (largo/pétalo, ancho/pétalo, etc.)
y = to_categorical(iris.target)  # Convertir a formato one-hot

# 2. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Definir modelo ANN
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),  # Capa oculta
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases de flores
])

# 4. Compilar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, verbose=0)

# 5. Evaluar
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Accuracy en test (ANN - Iris): {acc:.2f}")
