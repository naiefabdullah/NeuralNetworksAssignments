import numpy as np

class Parameters:
    def __init__(self, layer_sizes):
        self.weights = {}
        self.bias = {}
        for i in range(1, len(layer_sizes)):
            self.weights[i] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01
            self.bias[i] = np.zeros((layer_sizes[i], 1))

    def update_parameters(self, grads, learning_rate):
        L = len(self.weights)
        for l in range(1, L + 1):
            self.weights[l] -= learning_rate * grads["dW" + str(l)]
            self.bias[l] -= learning_rate * grads["dB" + str(l)]

class Activation:
    def __init__(self, activation_type):
        self.type = activation_type

    def forward(self, Z):
        if self.type == "linear":
            return Z  # Linear activation function simply returns the input
        if self.type == "relu":
            return np.maximum(0, Z)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.type == "tanh":
            return np.tanh(Z)
        elif self.type == "softmax":
            e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return e_Z / np.sum(e_Z, axis=0, keepdims=True)
        else:
            raise ValueError("Invalid activation function type: {}".format(self.type))

    def backward(self, dA, Z):
        if self.type == "linear":
            return dA  # Derivative of a linear function is 1
        if self.type == "relu":
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            return dZ
        elif self.type == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            return dA * s * (1 - s)
        elif self.type == "tanh":
            t = np.tanh(Z)
            return dA * (1 - t**2)
        else:
            raise ValueError("Invalid activation function type for backward pass: {}".format(self.type))

class Layer:
    def __init__(self, input_size, output_size, activation_type):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))
        self.activation = Activation(activation_type)

    def forward(self, A_prev):
        self.Z = np.dot(self.W, A_prev) + self.b
        self.A = self.activation.forward(self.Z)
        self.A_prev = A_prev
        return self.A

    def backward(self, dA):
        dZ = self.activation.backward(dA, self.Z)
        m = dA.shape[1]
        dW = np.dot(dZ, self.A_prev.T) / m
        dB = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev, dW, dB

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_types):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activation_types[i]) for i in range(len(layer_sizes)-1)]
        self.L = len(self.layers)
        self.is_softmax_output = activation_types[-1] == "softmax"

    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        if self.is_softmax_output:
            cost = -np.mean(np.sum(Y * np.log(AL + 1e-8), axis=0))  # Adding epsilon for numerical stability
        else:
            cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m  # Adding epsilon for numerical stability
        cost = np.squeeze(cost)
        return cost

    def backward_propagation(self, Y):
        grads = {}
        dAL = - (np.divide(Y, self.layers[-1].A) - np.divide(1 - Y, 1 - self.layers[-1].A))
        dA = dAL
        for l in reversed(range(self.L)):
            dA, dW, dB = self.layers[l].backward(dA)
            grads["dW" + str(l+1)] = dW
            grads["dB" + str(l+1)] = dB
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(self.L):
            self.layers[l].W -= learning_rate * grads["dW" + str(l+1)]
            self.layers[l].b -= learning_rate * grads["dB" + str(l+1)]

    def train(self, X, Y, learning_rate=0.001, num_iterations=3000, print_cost=True):
        for i in range(num_iterations):
            AL = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(Y)
            self.update_parameters(grads, learning_rate)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))

# Example usage:
layer_sizes = [2, 4, 3, 1]  # Example: 2 inputs, two hidden layers with 4 and 3 neurons, and 1 output
activation_types = ["relu", "relu", "sigmoid"]  # Relu for hidden layers, Sigmoid for output
network = NeuralNetwork(layer_sizes, activation_types)

# Generate example data
X = np.random.randn(2, 1000)  # Example input
Y = np.random.randint(0, 2, (1, 1000))  # Example output for binary classification

# Train the network
network.train(X, Y, learning_rate=0.0001, num_iterations=2500)