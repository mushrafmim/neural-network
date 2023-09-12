import math
import numpy as np
from perceptron import Perceptron


class Layer:

    def __init__(self, n_neurons: int, n_inputs: int, activation: str) -> None:
        self.input_v = None
        self.perceptrons: Perceptron = list()
        self.n_neurons: int = n_neurons
        self.activation = activation
        self.Z = None
        self.output = None
        self.Weights = None
        self.biases = None

        for _ in range(n_neurons):

            self.perceptrons.append(
                Perceptron(
                    W=np.random.uniform(low=-1.0, high=1.0, size=(n_inputs,)),
                    b=np.random.random()
                )
            )

    def relu_derivative(self):
        return np.where(self.output > 0, 1.0, 0.0)

    def find_dW(self, derivative, type="hidden"):

        return np.array(self.input_v)[:, np.newaxis] * derivative

    def set_weights_and_biases(self, weights: np.array, biases: np.array):
        self.Weights = weights
        self.biases = biases

        for i in range(self.n_neurons):
            self.perceptrons[i].W = weights[i]
            self.perceptrons[i].b = biases[i]

    def forward_pass(self, input_v, input=False):

        self.input_v = input_v

        output_v = np.empty(self.n_neurons, dtype=np.float64)

        for i in range(self.n_neurons):
            output_v[i] = self.perceptrons[i].calculate(input_v)

        self.Z = output_v.copy()

        if self.activation == "relu":
            output_v = np.maximum(output_v, 0)

        elif self.activation == "softmax":
            output_v = np.exp(output_v)
            output_v = np.divide(output_v, sum(output_v))

        self.output = output_v.copy()

        return output_v
