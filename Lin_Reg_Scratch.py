import matplotlib.pyplot as plt 
import numpy as np 

class LinearRegression:
	def __init__(self, lr, epochs): 
		self.lr = lr
		self.epochs = epochs 
		self.cost = []		

	def fit(self, X_train, y_train):		
		np.random.seed(42)

		m,n = X_train.shape
		
		self.theta = np.random.randn( n , 1 )

		for _ in range(self.epochs):			
			y_hat = np.dot( X_train, self.theta )
			errors = y_hat - y_train 

			cost = (1/(2*m)) * np.sum( np.square( errors ))
			self.cost.append( cost )
			
			dcost_theta = (1/m) * np.dot( X_train.T, errors )			

			self.theta -=  self.lr * dcost_theta
				

	def _predict(self, X_test):
		return np.dot( X_test, self.theta )
		
	def _plot_cost(self):
		plt.plot( np.arange(self.epochs), self.cost )
		plt.xlabel("Epochs")
		plt.ylabel("Cost")
		plt.title("Cost over Epochs")
		plt.show()

	def r_squared(self, y_true, y_hat):
		ss_total = np.sum(np.square(y_true - np.mean(y_true)))
		ss_residual = np.sum(np.square(y_true - y_hat))
		return 1 - (ss_residual / ss_total)



#Ejemplo de regresion lineal sencilla

np.random.seed(42)

X = np.linspace(1, 100, 100).reshape(-1, 1)  # Variable independiente (vector columna)
y = 2 * X + np.random.randn(100, 1) * 5  # Variable dependiente con algo de ruido

X_b = np.c_[np.ones((X.shape[0], 1)), X]

X_train, X_test = X_b[:80], X_b[80:]
y_train, y_test = y[:80], y[80:]

lr = 0.0001
epochs = 10

linreg  = LinearRegression( lr, epochs )
linreg.fit( X_train, y_train )
y_hat = linreg._predict( X_test )
linreg._plot_cost()
print( linreg.r_squared( y_test, y_hat ) )


#Ejemplo de regresion lineal multiple
np.random.seed(42)

X1 = np.linspace(1, 100, 100).reshape(-1, 1)  # Primera variable independiente
X2 = np.random.rand(100, 1) * 50  # Segunda variable independiente (ruido)
X = np.hstack((X1, X2))

X_b = np.c_[np.ones((X.shape[0], 1)), X]

y = 2 * X1 + 3 * X2 + np.random.randn(100, 1) * 5

X_train, X_test = X_b[:80], X_b[80:]
y_train, y_test = y[:80], y[80:]

lr_ = 0.0001
epochs = 10

linreg  = LinearRegression( lr_, epochs )
linreg.fit( X_train, y_train )
y_hat = linreg._predict( X_test )
linreg._plot_cost()
print( linreg.r_squared( y_test, y_hat ) )
