# ANN From Scratch

A lightweight, pure Python implementation of an Artificial Neural Network (ANN) using generic NumPy operations. This project aims to broaden understanding of the internal mechanics of deep learning models by building them from the ground up without relying on high-level frameworks like TensorFlow or PyTorch.

## Features

- **Custom Layer Architecture**: define the input size and add multiple hidden layers with custom neuron counts.
- **Activation Functions**: Supports multiple activation functions:
  - `sigmoid`
  - `relu`
  - `tanh`
- **Forward Propagation**: Matrix-based forward pass.
- **Backpropagation**: Gradient descent implementation to update weights and biases.
- **Training Loop**: Built-in training method with epoch-based learning.

## Requirements

- Python 3.x
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hemanath-Kumar/ANN_From_scratch.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy scikit-learn
   ```
   *(Note: `scikit-learn` is only used for generating the synthetic dataset in the example script `2.py`)*

## Usage

Here is a simple example of how to use the `ANN` class (based on `2.py`):

```python
import numpy as np
from ANN import ANN

# 1. Initialize the Network
model = ANN()

# 2. Define Input Layer (pass your data to set input size)
# X_train shape should be (features, samples)
X_train = np.random.randn(3, 100) 
y_train = np.random.randint(0, 2, (1, 100))

model.input_layer(X_train)

# 3. Add Hidden Layers
model.hidden_layer(number_of_neurons=3, activation_function='relu')
model.hidden_layer(number_of_neurons=2, activation_function='relu')
model.hidden_layer(number_of_neurons=1, activation_function='sigmoid') # Output layer

# 4. Train the Model
model.train(X_train, y_train, epochs=1000, lr=0.01)

# 5. Make Predictions
predictions = model.predict(X_train)
print(predictions)
```

## Structure

- `ANN.py`: Contains the `ANN` class with all network logic.
- `2.py`: A demonstration script using `sklearn.datasets` to test the network.

## Contributing

Feel free to fork this project and submit pull requests. You can add more activation functions, optimizers (like Adam or SGD with momentum), or different loss functions.

## License

MIT
