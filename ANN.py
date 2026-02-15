import numpy as np

class ANN():
    def __init__(self):
        self.weights = []
        self.biases = []
        self.activations_func = []

    # ---------------- ACTIVATION ----------------
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, a):
        return np.where(a > 0, 1, 0)

    # ---------------- ADD LAYER ----------------
    def add_layer(self, input_size, output_size, activation='sigmoid'):
        W = np.random.randn(output_size, input_size)
        b = np.zeros((1, output_size))
        self.weights.append(W)
        self.biases.append(b)
        self.activations_func.append(activation)

    # ---------------- FORWARD ----------------
    def forward(self, X):
        self.activations = [X]
        a = X

        for W, b, act in zip(self.weights, self.biases, self.activations_func):
            z = np.dot(a, W.T) + b  # (samples, neurons)

            if act == 'sigmoid':
                a = self.sigmoid(z)
            elif act == 'relu':
                a = self.relu(z)

            self.activations.append(a)

        return a

    # ---------------- BACKPROP ----------------
    def backward(self, y, lr):
        m = y.shape[0]
        deltas = []

        # Output layer
        error = self.activations[-1] - y
        delta = error * self.sigmoid_derivative(self.activations[-1])
        deltas.append(delta)

        # Hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            error = deltas[-1] @ self.weights[i+1]
            
            if self.activations_func[i] == 'sigmoid':
                delta = error * self.sigmoid_derivative(self.activations[i+1])
            elif self.activations_func[i] == 'relu':
                delta = error * self.relu_derivative(self.activations[i+1])

            deltas.append(delta)

        deltas.reverse()

        # Update weights
        for i in range(len(self.weights)):
            dW = (deltas[i].T @ self.activations[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    # ---------------- TRAIN ----------------
    def train(self, X, y, epochs=1000, lr=0.1):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(y, lr)

            if i % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)




