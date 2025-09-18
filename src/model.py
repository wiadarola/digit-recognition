import torch

from src import utils


class TenDigitMLP:
    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 64,
        output_dim: int = 10,
        learning_rate: float = 1e-3,
    ):
        self.lr = learning_rate
        self.w1 = torch.rand((input_dim, hidden_dim_1))
        self.b1 = torch.zeros(hidden_dim_1)
        self.w2 = torch.rand((hidden_dim_1, hidden_dim_2))
        self.b2 = torch.zeros(hidden_dim_2)
        self.w3 = torch.rand(hidden_dim_2, output_dim)
        self.b3 = torch.zeros(output_dim)
        self.cache: dict[str, torch.Tensor]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten()
        z1 = x @ self.w1 + self.b1
        a1 = utils.sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = utils.sigmoid(z2)
        z3 = a2 @ self.w3 + self.b3
        y_hat = utils.softmax(z3)

        self.cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}

        return y_hat

    def backward(self, x: torch.Tensor, y: torch.Tensor):
        z1, a1, z2, a2, z3 = self.cache.values()

        # Output gradient
        dL_dz3 = utils.softmax(z3) - y  # (C,)

        # Back propagate
        dL_da2 = dL_dz3 @ self.w3.T  # (H2,)
        dL_dz2 = dL_da2 * utils.sigmoid_dz(z2)  # (H2,)

        dL_da1 = dL_dz2 @ self.w2.T  # (H1,)
        dL_dz1 = dL_da1 * utils.sigmoid_dz(z1)  # (H1,)

        # Weight gradients
        dL_dw3 = torch.outer(a2, dL_dz3)  # (H2, C)
        dL_db3 = dL_dz3  # (C,)

        dL_dw2 = torch.outer(a1, dL_dz2)  # (H1, H2)
        dL_db2 = dL_dz2  # (H2,)

        dL_dw1 = torch.outer(x, dL_dz1)  # (D, H1)
        dL_db1 = dL_dz1  # (H1,)

        # Gradient step
        self.w1 -= self.lr * dL_dw1
        self.b1 -= self.lr * dL_db1
        self.w2 -= self.lr * dL_dw2
        self.b2 -= self.lr * dL_db2
        self.w3 -= self.lr * dL_dw3
        self.b3 -= self.lr * dL_db3
