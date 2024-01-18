import math

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def binary_step(x):
        return 1 if x >= 0 else 0

class Neuron:
    def __init__(self, inputs, activation_func):
        self.inputs = inputs
        self.activation_func = activation_func
        self.bias = 1

    def output(self):
        weighted_sum = sum(input_val * (1 if input_val > 0 else 0) for input_val in self.inputs) + self.bias
        activated_output = self.activation_func(weighted_sum)
        return round(activated_output)

# Example usage
input_handler = lambda: [int(x) for x in input("Enter binary inputs (0 or 1) separated by space: ").split()]

user_inputs = input_handler()
neuron = Neuron(user_inputs, ActivationFunction.sigmoid)  # Example with Sigmoid
print(f"Neuron output: {neuron.output()}")