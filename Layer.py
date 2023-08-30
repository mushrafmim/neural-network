import numpy as np
from Perceptron import Perceptron


class Layer:

    def __init__(self, n_neurons: int, n_inputs: int, activation: str) -> None:
        self.perceptrons: Perceptron = list()
        self.n_neurons: int = n_neurons
        self.activation = activation

        for _ in range(n_neurons):

            self.perceptrons.append(
                Perceptron(
                    w=np.random.uniform(low=0.0, high=1.0, size=(n_inputs,)),
                    b=np.random.random()
                )
            )

    def forward_pass(self, input_v):

        output_v = np.empty(self.n_neurons)

        for i in range(self.n_neurons):
            output_v[i] = self.perceptrons[i].predict(input_v)
            print(f"\tIn Perceptron {i + 1}")

        if self.activation == "relu":
            output_v = np.max(output_v)
        elif self.activation == "softmax":
            output_v = np.exp(output_v)
            output_v = np.divide(output_v / sum(output_v))

        return output_v
