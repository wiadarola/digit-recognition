import numpy as np
from numpy.typing import NDArray
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import TenDigitMLP
from src.utils import categorical_cross_entropy


def main():
    train_dataloader, test_dataloader = get_data()

    model = TenDigitMLP()
    for e in tqdm(range(50), "training"):
        losses = [train_one_step(model, y) for y in train_dataloader]
        avg_loss = np.mean(losses)
        print(f"Epoch {e}: {avg_loss}")


def train_one_step(model: TenDigitMLP, y: NDArray):
    cache = model.forward(y)
    loss = categorical_cross_entropy(y, cache["y_hat"])
    model.backward(cache, y)
    return loss


def get_data():
    train_data = MNIST(
        "data/", train=True, download=True, transform=ToTensor())
    test_data = MNIST(
        "data/", train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_data, shuffle=True)
    test_dataloader = DataLoader(test_data, shuffle=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    main()
