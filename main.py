import numpy as np

from NN import NN
from Layer import Layer

if __name__ == "__main__":
    nn = NN([
        Layer(14, 1),
        Layer(100, 14),
        Layer(40, 100),
        Layer(4, 40)
    ])

    random_input = np.random.choice([-1, 1], size=14)

    print(nn.predict(14))