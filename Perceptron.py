import numpy as np


class Perceptron:

    def __init__(self, w, b) -> None:
        self.w = w
        self.b = b

    def calculate(self, x: np.ndarray) -> float:
        return np.dot(self.w, x) + self.b
