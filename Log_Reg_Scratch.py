import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegression:
	def __init__(self, lr, epochs):
		self.lr = lr 
		self.epochs = epochs 
		self.weights  = None
		self.bias = None
		self.cost = []

	def sigmoid(self, z ):
		return 1 / (1 + np.exp(-z))

	def fit(self, X_train, y_train):
		
		np.random.seed(42)
		m, n = X_train.shape		

		self.weights = np.random.randn( n , 1)
		self.bias = np.random.rand()

		for _ in range(self.epochs):

			z = np.dot( X_train, self.weights ) +  self.bias 
			
			y_hat = self.sigmoid( z )
			
			epsilon = 1e-15
			cost = -(1/m) * np.sum(y_train * np.log(y_hat + epsilon) + (1 - y_train) * np.log(1 - y_hat + epsilon))			
			self.cost.append(cost)

			d_cost_weights = (1/m) * np.dot(X_train.T, y_hat - y_train) 
			d_cost_bias = (1/m) * np.sum( y_hat - y_train )   

			self.weights -= self.lr * d_cost_weights
			self.bias -= self.lr * d_cost_bias

		return self.weights, self.bias

	def predict(self, X_test):
		y_hat = np.dot(X_test, self.weights) + self.bias
		y_hat_prob = self.sigmoid( y_hat )		
		return (y_hat_prob >= 0.5 ).astype(int)

	def plot_cost(self):
		plt.plot(np.arange(0, self.epochs), self.cost)
		plt.xlabel("Epochs")
		plt.ylabel("Cost")
		plt.title("Cost over Epochs")
		plt.show()

	def accuracy(self, y_true, y_pred):
		return np.mean(y_true == y_pred)


np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = (4 + 3 * X + np.random.randn(100, 1)) > 6  # Binary classification problem

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

lr = 0.05
epochs = 10000

logreg = LogisticRegression(lr, epochs)
logreg.fit(X_train, y_train)
logreg.plot_cost()
# Predicciones y evaluación
y_pred = logreg.predict(X_test)
print("Accuracy:", logreg.accuracy(y_test.flatten(), y_pred.flatten()))
