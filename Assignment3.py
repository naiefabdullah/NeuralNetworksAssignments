import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.W1, self.b1, self.W2, self.b2 = self.init_params()

    def init_params(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2

    @staticmethod
    def ReLU(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    @staticmethod
    def ReLU_deriv(Z):
        return Z > 0

    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, Z1, A1, Z2, A2, X, Y):
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / Y.size * dZ2.dot(A1.T)
        db2 = 1 / Y.size * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / Y.size * dZ1.dot(X.T)
        db1 = 1 / Y.size * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

    def gradient_descent(self, X, Y, alpha, iterations):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X, Y)
            self.update_params(dW1, db1, dW2, db2, alpha)

            if i % 10 == 0:
                predictions = self.get_predictions(A2)
                print(f"Iteration {i}: Accuracy {self.get_accuracy(predictions, Y)}")

    @staticmethod
    def get_predictions(A2):
        return np.argmax(A2, 0)

    @staticmethod
    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def test_prediction(self, index, X, Y):
        current_image = X[:, index, None]
        prediction = self.get_predictions(self.forward_prop(current_image)[-1])
        label = Y[index]
        print(f"Prediction: {prediction}, Label: {label}")
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

# Load and preprocess data
data = pd.read_csv('data/train.csv').to_numpy()
np.random.shuffle(data)

# Split data into training and development sets
X_dev = data[0:1000, 1:].T / 255.
Y_dev = data[0:1000, 0]
X_train = data[1000:, 1:].T / 255.
Y_train = data[1000:, 0]

# Initialize and train the neural network
nn = NeuralNetwork()
nn.gradient_descent(X_train, Y_train, alpha=0.1, iterations=500)

# Test predictions on the first ten samples of the training set
for i in range(10):
    nn.test_prediction(i, X_train, Y_train)

# Test accuracy on the development set
dev_predictions = nn.get_predictions(nn.forward_prop(X_dev)[-1])
dev_accuracy = nn.get_accuracy(dev_predictions, Y_dev)
print(f"Development Set Accuracy: {dev_accuracy}")