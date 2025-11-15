import torch
from torch import Tensor


def categorical_cross_entropy(y: Tensor, y_hat: Tensor) -> Tensor:
    return -torch.log(y_hat + 1e-15)[:,y]


def sigmoid(z: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-z))


def sigmoid_dz(z: Tensor) -> Tensor:
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z: Tensor) -> Tensor:
    exp_z = torch.exp(z - z.max())
    return exp_z / exp_z.sum()
