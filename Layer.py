import numpy as np
from Perceptron import Perceptron


class Layer:

    def __init__(self, n_neurons: int, n_inputs: int) -> None:
        self.perceptrons: Perceptron = list()
        self.n_neurons: int = n_neurons

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

        return output_v
