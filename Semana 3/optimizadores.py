import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Modelo simple
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

# Probar diferentes optimizers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
