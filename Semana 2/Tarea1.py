import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paso 1: Cargar datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
df = pd.read_csv(url, names=columns, sep=',\s', engine='python')

# Paso 2: Preprocesamiento
df = df.dropna()
df = df[df['income'].isin(['<=50K', '>50K'])]

# Convertir variables categóricas
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("income", axis=1)
y = df["income"]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Paso 3: Definir una función para crear el modelo con activación variable
def build_model(activation_function):
    model = Sequential([
        Dense(32, input_shape=(X_train.shape[1],), activation=activation_function),
        Dense(16, activation=activation_function),
        Dense(1, activation='sigmoid')  # siempre sigmoide en la salida para clasificación binaria
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Paso 4: Probar diferentes activaciones
for act in ['relu', 'tanh', 'sigmoid']:
    print(f"\nEntrenando con activación: {act}")
    model = build_model(act)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión con {act}: {acc:.4f}")