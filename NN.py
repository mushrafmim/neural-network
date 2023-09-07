import math
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

        print(output)

        output = np.exp(output)

        return output / sum(output)

    def find_error(self, Y_predicted, Y_i):
        cross_entropy_loss = np.dot(np.log(Y_predicted), Y_i)
        return cross_entropy_loss

    def train(self, X, Y):

        for i in range(len(X)):
            X_i = X[i]
            Y_i = Y[i]

            Y_predicted = self.predict(X_i)
