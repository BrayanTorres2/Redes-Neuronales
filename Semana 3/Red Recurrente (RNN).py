# ============================
# Ejemplo 3: Red Recurrente (RNN - LSTM)
# Dataset: IMDB (opiniones positivas/negativas)
# ============================

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. Cargar dataset IMDB (10k palabras más comunes)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# 2. Normalizar longitud (padding)
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

# 3. Definir modelo RNN con LSTM
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=200),
    LSTM(64),
    Dense(1, activation='sigmoid')  # Sentimiento (positivo o negativo)
])

# 4. Compilar y entrenar
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size=64, verbose=1)

# 5. Evaluar
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Accuracy en test (RNN - IMDB): {acc:.2f}")
