import numpy as np


class Perceptron:

    def __init__(self, W, b) -> None:
        self.W = W
        self.b = b

    def calculate(self, X: np.ndarray) -> float:

        output = np.dot(self.W, X) + self.b

        return output

    def find_derivative(self, derivative):
        return derivative / self.W
