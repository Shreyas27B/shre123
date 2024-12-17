from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# Define the datasets
datasets = {
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    "OR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]))
}

# Function to train and evaluate the perceptron
def train_and_evaluate(X, y, title):
    perceptron = Perceptron(max_iter=1000, eta0=1, random_state=0).fit(X, y)
    predictions = perceptron.predict(X)
    
    # Plot data and predictions
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=100)
    for i, pred in enumerate(predictions):
        plt.text(X[i, 0], X[i, 1], str(pred), color='white', ha='center', va='center')
    plt.title(f'{title} Dataset')
    plt.show()
    
    print(f'{title} Accuracy: {np.mean(predictions == y) * 100:.2f}%\n')

# Evaluate datasets
for name, (X, y) in datasets.items():
    train_and_evaluate(X, y, name)
