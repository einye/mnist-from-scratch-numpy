import numpy as np
import pandas as pd


# Load the MNIST dataset from local data folder containing CSV files
def load_mnist():
    # Load the dataset from CSV files
    train_data = pd.read_csv('data/mnist_train.csv')
    test_data = pd.read_csv('data/mnist_test.csv')
    
    # Separate features and labels
    X_train = train_data.iloc[:, 1:].values / 255.0  # Normalize pixel values
    y_train = train_data.iloc[:, 0].values          # First column is the label
    X_test = test_data.iloc[:, 1:].values / 255.0   # Normalize pixel values
    y_test = test_data.iloc[:, 0].values            # First column is the label
    
    return X_train, X_test, y_train, y_test

# Define the Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    @staticmethod
    def scaled_tanh(x):
        return 1.7159 * np.tanh(2 / 3 * x)

    @staticmethod
    def scaled_tanh_derivative(x):
        return 1.7159 * (2 / 3) * (1 - np.tanh(2 / 3 * x) ** 2)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z = []
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = self.scaled_tanh(z) if i < len(self.weights) - 1 else self.softmax(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y = y.astype(int)  # Convert labels to integer type
        y_one_hot = np.zeros((m, self.a[-1].shape[1]))
        y_one_hot[np.arange(m), y] = 1  # Convert labels to one-hot encoding

        dW = []
        db = []
        delta = self.a[-1] - y_one_hot

        for i in reversed(range(len(self.weights))):
            dW.insert(0, np.dot(self.a[i].T, delta) / m)
            db.insert(0, np.sum(delta, axis=0, keepdims=True) / m)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.scaled_tanh_derivative(self.z[i - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 10 == 0:
                predictions = self.predict(X)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%")

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

# Main code
if __name__ == "__main__":
    # Load the MNIST dataset
    X_train, X_test, y_train, y_test = load_mnist()

    # Initialize the neural network based on the paper's architecture
    input_size = 784  # 28x28 flattened image
    hidden_sizes = [1000, 500]  # Two hidden layers 
    output_size = 10  # 10 digits (0-9)

    nn = NeuralNetwork(input_size, hidden_sizes, output_size)

    # Train the network
    epochs = 100
    learning_rate = 0.01
    nn.train(X_train, y_train, epochs, learning_rate)

    # Test the network
    test_predictions = nn.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
