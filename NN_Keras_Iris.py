import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

iris = load_iris()
print("matrices y vectores ", iris.keys())

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

print("Clases únicas:", np.unique(y))
print("Conteo por clase:\n", y.value_counts())
print("Columnas:\n", X.columns.tolist())
print("\nInfo del DataFrame:")
print(X.info())

#one hot encoding
y = to_categorical(iris.target.astype(int), 3)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
print(f"\nTamaños -> X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"\nTamaños -> y_train: {y_train.shape}, y_test: {y_test.shape}")

#Preparar el módelo
model = Sequential([
    Dense(16, input_shape=(4,), activation='relu'),    
    Dense(8, activation='relu'),    
    Dense(3, activation='softmax')
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Entrenar 
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
# Evaluar
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPérdida: {loss:.4f} | Precisión: {accuracy:.4f}")
