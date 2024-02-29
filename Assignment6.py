import numpy as np

class Activation:
    def __init__(self, activation_type):
        self.type = activation_type

    def forward(self, Z):
        if self.type == "linear":
            return Z
        elif self.type == "relu":
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
            return dA
        elif self.type == "relu":
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
        self.dropout_mask = None
        self.keep_prob = 1

    def forward(self, A_prev, keep_prob=1):
        self.Z = np.dot(self.W, A_prev) + self.b
        self.A = self.activation.forward(self.Z)
        if keep_prob < 1:
            self.dropout_mask = np.random.rand(*self.A.shape) < keep_prob
            self.A *= self.dropout_mask
            self.A /= keep_prob
        self.keep_prob = keep_prob
        self.A_prev = A_prev
        return self.A

    def backward(self, dA, lambd=0.0):
        if self.keep_prob < 1:
            dA *= self.dropout_mask
            dA /= self.keep_prob

        dZ = self.activation.backward(dA, self.Z)
        m = dA.shape[1]
        dW = np.dot(dZ, self.A_prev.T) / m + (lambd / m) * self.W
        dB = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev, dW, dB

class Preprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=1, keepdims=True)
        self.std = np.std(X, axis=1, keepdims=True)

    def transform(self, X):
        X_normalized = (X - self.mean) / (self.std + 1e-8)
        return X_normalized

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_types):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activation_types[i]) for i in range(len(layer_sizes)-1)]
        self.L = len(self.layers)
        self.is_softmax_output = activation_types[-1] == "softmax"
        self.preprocessor = Preprocessor()

    def forward_propagation(self, X, keep_prob=1.0):
        A = X
        for layer in self.layers:
            A = layer.forward(A, keep_prob=keep_prob)
        return A

    def compute_cost(self, AL, Y, lambd=0.0):
        m = Y.shape[1]
        epsilon = 1e-8
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        if self.is_softmax_output:
            cost = -np.mean(np.sum(Y * np.log(AL_clipped), axis=0))
        else:
            cost = -np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped)) / m
        L2_cost = 0
        if lambd > 0:
            L2_cost = (lambd / (2 * m)) * sum([np.sum(np.square(layer.W)) for layer in self.layers])
        cost = cost + L2_cost
        return np.squeeze(cost)

    def backward_propagation(self, Y, lambd=0.0):
        grads = {}
        epsilon = 1e-8
        AL = np.clip(self.layers[-1].A, epsilon, 1 - epsilon)
        if self.is_softmax_output:
            dAL = self.layers[-1].A - Y
        else:
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA = dAL
        for l in reversed(range(self.L)):
            dA, dW, dB = self.layers[l].backward(dA, lambd=lambd)
            grads["dW" + str(l+1)] = dW
            grads["dB" + str(l+1)] = dB
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(self.L):
            self.layers[l].W -= learning_rate * grads["dW" + str(l+1)]
            self.layers[l].b -= learning_rate * grads["dB" + str(l+1)]

    def train(self, X, Y, learning_rate=0.001, num_iterations=3000, print_cost=True, lambd=0.0, keep_prob=1.0):
        X_norm = self.preprocessor.fit_transform(X)
        for i in range(num_iterations):
            AL = self.forward_propagation(X_norm, keep_prob=keep_prob)
            cost = self.compute_cost(AL, Y, lambd=lambd)
            grads = self.backward_propagation(Y, lambd=lambd)
            self.update_parameters(grads, learning_rate)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

# Example usage
layer_sizes = [2, 4, 3, 1]  # Define layer sizes
activation_types = ["relu", "relu", "sigmoid"]  # Define activation functions for each layer
network = NeuralNetwork(layer_sizes, activation_types)

# Generate example data
X = np.random.randn(2, 1000)  # Example input
Y = np.random.randint(0, 2, (1, 1000))  # Example output for binary classification

# Train the network with L2 regularization and dropout
network.train(X, Y, learning_rate=0.01, num_iterations=5000, print_cost=True, lambd=0.7, keep_prob=0.8)
