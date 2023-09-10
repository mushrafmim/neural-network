import numpy as np

from nn import NN
from layer import Layer

if __name__ == "__main__":
    nn = NN([
        Layer(14, 1, activation="relu"),
        Layer(100, 14, activation="relu"),
        Layer(40, 100, activation="relu"),
        Layer(4, 40, activation="softmax")
    ])

    # random_input = np.random.choice([-1, 1], size=14)
    data_point = [-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]

    print(nn.backpropogate(data_point, [0, 0, 1, 0]))

    # print(nn.predict(data_point))
