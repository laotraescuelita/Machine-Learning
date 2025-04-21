import numpy as np
import matplotlib.pyplot as plt

# Funciones de activación
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Evita desbordamiento
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Derivadas de funciones de activación
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Función de pérdida
#def cross_entropy_loss(y_true, a2):
    #epsilon = 1e-15  # Evita log(0)
    #a2 = np.clip(a2, epsilon, 1 - epsilon)
    #return -np.sum(y_true * np.log(a2)) / y_true.shape[0]
    
# Clase de la Red Neuronal
class NeuralNetwork:
    def __init__(self, learning_rate, epochs, hidden_size=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.cost = []

    def fit(self, X_train, y_train):
        np.random.seed(42)
        m, n = X_train.shape
        k = y_train.shape[1]  # Número de clases

        # Inicialización de pesos y sesgos
        self.w1 = np.random.randn(n, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = np.random.randn(self.hidden_size, k)
        self.b2 = np.zeros((1, k))

        for epoch in range(self.epochs):
            # **Forward propagation**
            z1 = np.dot(X_train, self.w1) + self.b1
            a1 = relu(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = softmax(z2)

            # **Cálculo de pérdida**           
            epsilon = 1e-15
            cost = -(1/m) * np.sum(y_train * np.log(a2 + epsilon))
            self.cost.append(cost)

            # **Backward propagation**
            dz2 = a2 - y_train
            dw2 = np.dot(a1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m

            dz1 = np.dot(dz2, self.w2.T) * relu_derivative(z1)
            dw1 = np.dot(X_train.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            # **Actualización de pesos y sesgos**
            self.w1 -= self.learning_rate * dw1
            self.b1 -= self.learning_rate * db1
            self.w2 -= self.learning_rate * dw2
            self.b2 -= self.learning_rate * db2

            # Mostrar costo cada 10% de las épocas
            if epoch % (self.epochs // 10) == 0:
                print(f"Época {epoch}: Costo = {cost:.4f}")

    def predict(self, X_test):
        z1 = np.dot(X_test, self.w1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        return softmax(z2)  # Predicciones en probabilidades

    def plot_cost(self):
        plt.plot(np.arange(self.epochs), self.cost)
        plt.xlabel("Épocas")
        plt.ylabel("Costo")
        plt.title("Evolución del Costo")
        plt.show()

    def accuracy(self, X_train, y_true):
        a2 = self.predict(X_train)
        a2_labels = np.argmax(a2, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(a2_labels == y_true_labels)


# **Generación de datos**
np.random.seed(42)
X = 2 * np.random.rand(100, 2)  # 2 características
y = np.random.randint(0, 10, size=(100, 1))
print( y )

# **One-hot encoding de etiquetas**
y_one_hot = np.eye(10)[y.flatten()]
print( y_one_hot )

# **División en entrenamiento y prueba**
X_train, X_test = X[:80], X[80:]
y_train, y_test = y_one_hot[:80], y_one_hot[80:]

# **Entrenamiento de la Red Neuronal**
learning_rate = 0.0001 #0.01
epochs = 10 #10000
hidden_size = 10

nn = NeuralNetwork(learning_rate, epochs, hidden_size)
nn.fit(X_train, y_train)
nn.plot_cost()

# **Predicciones**
a2 = nn.predict(X_test)
a2_labels = np.argmax(a2, axis=1)  # Convertir a clases

precision = nn.accuracy(X_train, y_train)

# **Imprimir algunas predicciones**
print("Predicciones en la primera fila:")
print("Probabilidades:", a2[0])
print("Clase Predicha:", a2_labels[0])
print("Precisión:", precision)

