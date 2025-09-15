import numpy as np
from numpy.typing import NDArray

from src import utils


class TenDigitMLP:
    def __init__(self, lr: float = 1e-3):
        self.lr = lr
        input_size = 28 * 28
        output_size = 10
        self.w1 = np.random.uniform(-0.05, 0.05, (input_size, 128))
        self.b1 = np.random.uniform(-0.05, 0.05, 128)
        self.w2 = np.random.uniform(-0.05, 0.05, (128, 64))
        self.b2 = np.random.uniform(-0.05, 0.05, 64)
        self.w3 = np.random.uniform(-0.05, 0.05, (64, output_size))
        self.b3 = np.random.uniform(-0.05, 0.05, output_size)

    def forward(self, x: NDArray) -> dict[str, NDArray]:
        z1 = x @ self.w1 + self.b1
        a1 = utils.sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = utils.sigmoid(z2)
        z3 = a2 @ self.w3 + self.b3
        y_hat = utils.softmax(z3)
        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
            "y_hat": y_hat,
        }
        return cache

    def backward(self, cache: dict[str, NDArray], y: NDArray):
        da3_dz3 = utils.softmax(cache["z3"]) - y
        dz3_da2 = self.w3
        da2_dz2 = utils.sigmoid_dz(cache["z2"])
        dz2_da1 = self.w2
        da1_dz1 = utils.sigmoid_dz(cache["z1"])

        # Output Layer
        delta3 = da3_dz3
        dL_dw3 = cache["a2"].T @ delta3
        dL_db3 = delta3

        # Hidden layer
        delta2 = delta3 @ dz3_da2 * da2_dz2
        dL_dw2 = cache["a1"].T @ delta2
        dL_db2 = delta2

        # Input layer
        delta1 = delta2 @ dz2_da1 * da1_dz1
        dL_dw1 = cache["x"] @ delta1
        dL_db1 = delta1

        # Gradient Descent step
        self.w1 -= self.lr * dL_dw1
        self.b1 -= self.lr * dL_db1
        self.w2 -= self.lr * dL_dw2
        self.b2 -= self.lr * dL_db2
        self.w3 -= self.lr * dL_dw3
        self.b3 -= self.lr * dL_db3
