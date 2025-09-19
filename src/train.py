import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.model import TenDigitMLP
from src.utils import categorical_cross_entropy

# TODO: K-fold cross validation
# TODO: Implement batch size
# TODO: Implement stochastic gradient descent


def main():
    model = TenDigitMLP()
    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    for epoch in range(50):
        losses = [train_one_step(model, x, y) for x, y in tqdm(train_loader)]
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch}: {avg_loss:.4f}")

    losses = [forward_pass(model, x, y) for x, y in tqdm(test_loader, "Testing")]
    avg_loss = np.mean(losses)
    print(f"Test: {avg_loss:.4f}")


def train_one_step(model: TenDigitMLP, x: Tensor, y: Tensor) -> float:
    loss = forward_pass(model, x, y)
    model.backward(x, y)
    return loss


def forward_pass(model: TenDigitMLP, x: Tensor, y: Tensor) -> float:
    y_hat = model.forward(x)
    loss = categorical_cross_entropy(y, y_hat)
    return loss


def load_data(train: bool) -> DataLoader:
    data = MNIST("data/", train=train, download=True, transform=ToTensor())
    dataloader = DataLoader(data, shuffle=True)
    return dataloader


if __name__ == "__main__":
    main()
