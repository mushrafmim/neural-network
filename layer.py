import math
import numpy as np
from perceptron import Perceptron


class Layer:

    def __init__(self, n_neurons: int, n_inputs: int, activation: str) -> None:
        self.perceptrons: Perceptron = list()
        self.n_neurons: int = n_neurons
        self.activation = activation
        self.output = np.empty(n_neurons)

        for _ in range(n_neurons):

            self.perceptrons.append(
                Perceptron(
                    W=np.random.uniform(low=-1.0, high=1.0, size=(n_inputs,)),
                    b=np.random.random()
                )
            )

    def find_derivative(self, derivative):
        return self.output[:, np.newaxis] * derivative
        print(derivative)
        row = derivative.shape[0]
        return np.dot(self.output.reshape(self.n_neurons, 1), derivative.reshape(1, row))

    def set_weights_and_biases(self, weights: np.array, biases: np.array):

        for i in range(self.n_neurons):
            self.perceptrons[i].W = weights[i]
            self.perceptrons[i].b = biases[i]

    def forward_pass(self, input_v, input=False):

        output_v = np.empty(self.n_neurons)

        for i in range(self.n_neurons):
            output_v[i] = self.perceptrons[i].calculate(input_v)

        if self.activation == "relu":
            output_v = np.maximum(output_v, 0)

        elif self.activation == "softmax":
            print(output_v)
            output_v = np.exp(output_v)
            output_v = np.divide(output_v, sum(output_v))

        self.output = output_v.copy()

        return output_v
