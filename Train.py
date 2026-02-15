from ANN import ANN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train= y_train.reshape(-1, 1)

ann = ANN()
ann.add_layer(x_train.shape[1], 3, activation='relu')
ann.add_layer(3, 2, activation='relu')
ann.add_layer(2, 1, activation='sigmoid')

ann.train(x_train, y_train, epochs=5000, lr=0.1)

print("Predictions:")
print(ann.predict(x_test))
