import numpy as np


class Perceptron:

    def __init__(self, w, b) -> None:
        self.w = w
        self.b = b

    def predict(self, x) -> float:
        return max(0, np.dot(self.w, x) + self.b)
