import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=5000, l2=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2  # Ajout du paramètre de régularisation
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, pos_weight=2.0):
        self.m, self.n = X.shape
        self.X = np.hstack((np.ones((self.m, 1)), X))
        self.y = y
        self.weights = np.zeros(self.n + 1)

        for _ in range(self.epochs):
            predictions = self.sigmoid(np.dot(self.X, self.weights))
            # Poids pour chaque échantillon
            sample_weights = np.where(self.y == 1, pos_weight, 1)
            gradient = np.dot(self.X.T, sample_weights * (predictions - self.y)) / self.m
            gradient[1:] += self.l2 * self.weights[1:]
            self.weights -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(np.dot(X, self.weights))
    
    def predict(self, X, threshold=0.4):
        return (self.predict_proba(X) >= threshold).astype(int)
