import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

# Cargar datos
digits = fetch_openml('mnist_784', version=1, as_frame=False)

X = digits.data / 255.0  # Normalizar

#one hot encoding
y = to_categorical(digits.target.astype(int), 10)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Modelo
model = Sequential([
    Dense(512, input_shape=(784,), activation='relu'),
    Dropout(0.2),
    Dense(120, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

# Entrenar
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=2)

# Evaluar
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPérdida: {loss:.4f} | Precisión: {acc:.4f}")

