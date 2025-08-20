# ============================
# Ejemplo 4: GAN (Generative Adversarial Network)
# Dataset: MNIST (generar dígitos falsos)
# ============================

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

# 1. Cargar datos reales (MNIST)
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0  # Flatten imágenes 28x28 a vector 784

# 2. Definir generador
generator = Sequential([
    Dense(128, input_dim=100),
    LeakyReLU(0.2),
    Dense(784, activation='sigmoid')  # Imagen 28x28 aplanada
])

# 3. Definir discriminador
discriminator = Sequential([
    Dense(128, input_dim=784),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')  # Real (1) o Falso (0)
])
discriminator.compile(optimizer=Adam(0.0002), loss='binary_crossentropy', metrics=['accuracy'])

# 4. Unir GAN (generador + discriminador congelado)
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')

# 5. Entrenar GAN (ejemplo reducido a 1 época)
for epoch in range(1):
    # Seleccionar imágenes reales
    idx = np.random.randint(0, X_train.shape[0], 64)
    real_imgs = X_train[idx]
    real_labels = np.ones((64, 1))

    # Generar imágenes falsas
    noise = np.random.normal(0, 1, (64, 100))
    fake_imgs = generator.predict(noise)
    fake_labels = np.zeros((64, 1))

    # Entrenar discriminador
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

    # Entrenar generador (quiere engañar al discriminador)
    noise = np.random.normal(0, 1, (64, 100))
    g_loss = gan.train_on_batch(noise, np.ones((64, 1)))

    print(f"Época 1 - D_loss_real: {d_loss_real[0]:.4f}, D_loss_fake: {d_loss_fake[0]:.4f}, G_loss: {g_loss:.4f}")
