import numpy as np
from numpy.typing import NDArray


def categorical_cross_entropy(y: NDArray, y_hat: NDArray) -> float:
    eps = 1e-15
    return -np.sum(y * np.log(y_hat + eps))


def relu(z: NDArray) -> NDArray:
    return np.maximum(0, z)


def relu_dz(z: NDArray) -> NDArray:
    raise NotImplementedError()


def sigmoid(z: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-z))


def sigmoid_dz(z: NDArray) -> NDArray:
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z: NDArray) -> NDArray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()
