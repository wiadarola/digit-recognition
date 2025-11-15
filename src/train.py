import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.model import TenDigitMLP
from derivatives import categorical_cross_entropy


def main():
    model = TenDigitMLP()
    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    for epoch in range(50):
        losses = []
        for x, y in tqdm(train_loader):
            y_hat = model.forward(x)
            loss = categorical_cross_entropy(y, y_hat)
            model.backward(x, y)
            losses.append(loss)
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch}: {avg_loss:.4f}")

    losses = []
    for x, y in tqdm(test_loader):
        y_hat = model.forward(x)
        loss = categorical_cross_entropy(y, y_hat)
        losses.append(loss)
    avg_loss = np.mean(losses)
    print(f"Test Loss: {avg_loss:.4f}")


def load_data(train: bool) -> DataLoader:
    data = MNIST("data/", train=train, download=True, transform=ToTensor())
    dataloader = DataLoader(data, shuffle=train)
    return dataloader


if __name__ == "__main__":
    main()
