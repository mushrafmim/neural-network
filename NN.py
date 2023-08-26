import numpy as np
from Layer import Layer


class NN:

    def __init__(self, layers: list[Layer]) -> None:
        self.layers: list[Layer] = layers

    def predict(self, X) -> np.array:

        print("In Layer 1")
        output = self.layers[0].forward_pass(X)

        for i in range(1, len(self.layers)):
            print(f"In Layer {i+1}")
            output = self.layers[i].forward_pass(output)

        return output
