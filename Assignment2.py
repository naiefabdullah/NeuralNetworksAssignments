import pandas as pd
import numpy as np
import cv2

# Base class for neurons
class Neuron:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases with random values
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.random.rand(output_size, 1) - 0.5

    def forward(self, inputs):
        # Forward pass through the neuron
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.bias
        return self.output

# Base class for activation functions
class ActivationFunction:
    def forward(self, inputs):
        raise NotImplementedError

# Softmax activation function
class Softmax(ActivationFunction):
    def forward(self, inputs):
        # Compute softmax values
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return probabilities

# Class for handling input data
class InputData:
    def __init__(self, filename):
        # Load data from a file and shuffle it
        self.data = pd.read_csv(filename)
        self.data = np.array(self.data)
        np.random.shuffle(self.data)

    def preprocess_data(self, start_index, end_index, image_size=20):
        # Preprocess data: resize and normalize images
        data_subset = self.data[start_index:end_index]
        X = []
        for image in data_subset[:, 1:]:
            try:
                # Ensure image data is in the correct format
                reshaped_image = image.reshape(28, 28).astype(np.uint8)
                resized_image = cv2.resize(reshaped_image, (image_size, image_size))
                X.append(resized_image.flatten())
            except Exception as e:
                print(f"Error processing image: {e}")
                continue  # Skip this image and move to the next

        X = np.array(X) / 255.0
        Y = data_subset[:, 0]
        return X.T, Y

# Class for training the model
class Training:
    def __init__(self, neuron, activation_function):
        self.neuron = neuron
        self.activation_function = activation_function

    def train(self, X, Y, iterations, alpha):
        for i in range(iterations):
            # Forward pass
            output = self.neuron.forward(X)
            prediction = self.activation_function.forward(output)

            # Compute the error (Cross-Entropy Loss)
            m = Y.size
            one_hot_Y = one_hot(Y)
            error = - (one_hot_Y * np.log(prediction)).sum() / m

            # Backward pass
            dZ = prediction - one_hot_Y
            dW = 1 / m * np.dot(dZ, X.T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

            # Update parameters
            self.neuron.weights -= alpha * dW
            self.neuron.bias -= alpha * db

            if i % 10 == 0:
                print(f"Iteration {i}: Loss {error}")

# Class for testing the model
class Testing:
    def __init__(self, neuron, activation_function):
        self.neuron = neuron
        self.activation_function = activation_function

    def test(self, X, Y):
        # Test the model and calculate accuracy
        output = self.neuron.forward(X)
        predictions = self.activation_function.forward(output)
        predictions = np.argmax(predictions, 0)
        accuracy = np.sum(predictions == Y) / Y.size
        return accuracy

# Function to convert labels to one-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Main program
def main():
    # Load and preprocess data with resizing to 20x20
    input_data = InputData('data/train.csv')
    X_train, Y_train = input_data.preprocess_data(1000, 42000, image_size=20)
    X_dev, Y_dev = input_data.preprocess_data(0, 1000, image_size=20)

    # Initialize neuron with 400 input size for 20x20 images
    neuron = Neuron(400, 10)  # Now 400 for flattened 20x20 images
    activation_function = Softmax()

    # Train the model
    trainer = Training(neuron, activation_function)
    trainer.train(X_train, Y_train, iterations=500, alpha=0.1)

    # Test the model
    tester = Testing(neuron, activation_function)
    accuracy = tester.test(X_dev, Y_dev)
    print(f"Development Set Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
