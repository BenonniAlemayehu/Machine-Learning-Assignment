import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                y_pred = np.dot(X[i], self.weights) + self.bias
                update = self.learning_rate * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Load your CSV dataset
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Explore the dataset and preprocess if necessary

# Assuming 'feature1', 'feature2', ..., 'target' are the column names
X = data[['hypertension']].values
y = data['diabetes'].values

# Convert y to 1D array (if it's not already)
y = np.ravel(y)

# Instantiate the Perceptron model
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)

# Train the model
perceptron.train(X, y)

# Make predictions
predictions = perceptron.predict(X)
