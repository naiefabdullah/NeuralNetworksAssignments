import pandas as pd
import numpy as np
import cv2

class Neuron:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.bias
        return self.output

class ActivationFunction:
    def forward(self, inputs):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class InputData:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.data = np.array(self.data)
        np.random.shuffle(self.data)

    def preprocess_data(self, start_index, end_index, image_size=20):
        data_subset = self.data[start_index:end_index]
        X = []
        for image in data_subset[:, 1:]:
            reshaped_image = image.reshape(28, 28).astype(np.uint8)
            resized_image = cv2.resize(reshaped_image, (image_size, image_size))
            X.append(resized_image.flatten())
        X = np.array(X) / 255.0
        Y = data_subset[:, 0]
        return X.T, Y

class Training:
    def __init__(self, neuron, activation_function):
        self.neuron = neuron
        self.activation_function = activation_function

    def train(self, X, Y, iterations, alpha):
        for i in range(iterations):
            output = self.neuron.forward(X)
            prediction = self.activation_function.forward(output)
            m = Y.size
            error = -np.sum(Y * np.log(prediction) + (1 - Y) * np.log(1 - prediction)) / m
            dZ = prediction - Y
            dW = 1 / m * np.dot(X, dZ.T)
            db = 1 / m * np.sum(dZ)
            self.neuron.weights -= alpha * dW.T
            self.neuron.bias -= alpha * db
            if i % 10 == 0:
                print(f"Iteration {i}: Loss {error}")

class Testing:
    def __init__(self, neuron, activation_function):
        self.neuron = neuron
        self.activation_function = activation_function

    def test(self, X, Y):
        output = self.neuron.forward(X)
        prediction = self.activation_function.forward(output)
        predictions = np.argmax(prediction, 0)
        accuracy = np.sum(predictions == Y) / Y.size
        return accuracy

def binary_labels(Y, class_label):
    return np.where(Y == class_label, 1, 0)

def test_one_vs_all(models, X, Y):
    predictions = []
    for model in models:
        neuron, activation_function = model
        output = neuron.forward(X)
        prediction = activation_function.forward(output)
        predictions.append(prediction)
    predictions = np.array(predictions).reshape(len(models), -1)
    predicted_classes = np.argmax(predictions, axis=0)
    accuracy = np.mean(predicted_classes == Y)
    return accuracy

def main():
    input_data = InputData('data/train.csv')
    X_train, Y_train = input_data.preprocess_data(0, 42000, image_size=20)
    X_dev, Y_dev = input_data.preprocess_data(0, 1000, image_size=20)
    models = []
    for class_label in range(10):
        print(f"Training for class {class_label}")
        Y_train_binary = binary_labels(Y_train, class_label)
        Y_dev_binary = binary_labels(Y_dev, class_label)
        neuron = Neuron(400, 1)
        activation_function = Sigmoid()
        trainer = Training(neuron, activation_function)
        trainer.train(X_train, Y_train_binary, iterations=500, alpha=0.1)
        models.append((neuron, activation_function))
    accuracy = test_one_vs_all(models, X_dev, Y_dev)
    print(f"Development Set Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
