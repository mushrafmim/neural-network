import numpy as np


def relu_activation(arr: np.ndarray[float]) -> np.ndarray[float]:
    return np.maximum(0, arr)


def softmax_activation(arr: np.ndarray[float]) -> np.ndarray[float]:
    arr = np.exp(arr)
    return np.divide(arr, sum(arr))
