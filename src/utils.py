import numpy as np
import torch
from numpy.typing import NDArray


def categorical_cross_entropy(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    return -np.log(y_hat + 1e-15)[y]


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-z))


def sigmoid_dz(z: torch.Tensor) -> torch.Tensor:
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z: torch.Tensor) -> torch.Tensor:
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()
