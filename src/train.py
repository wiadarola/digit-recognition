import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.model import TenDigitMLP
from derivatives import categorical_cross_entropy
from sklearn.metrics import accuracy_score


def main():
    model = TenDigitMLP()
    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    n_epochs = 50
    pbar = tqdm(total=n_epochs, desc="Training")
    for epoch in range(n_epochs):
        pbar.update(epoch)
        losses = []
        for x, y in tqdm(train_loader, leave=False):
            y_hat = model.forward(x)
            loss = categorical_cross_entropy(y, y_hat)
            model.backward(x, y)
            losses.append(loss)
        pbar.set_description(f"Training (Loss: {np.mean(losses):.4f})")

    losses = []
    accuracies = []
    for x, y in tqdm(test_loader, "Testing", leave=False):
        y_hat = model.forward(x)
        loss = categorical_cross_entropy(y, y_hat)
        losses.append(loss)
        accuracy = accuracy_score(y, y_hat.argmax(1))
        accuracies.append(accuracy)
    print(f"Test Loss: {np.mean(losses):.4f}")
    print(f"Test Accuracy: {np.mean(accuracies):.2f}")


def load_data(train: bool) -> DataLoader:
    data = MNIST("data/", train=train, download=True, transform=ToTensor())
    dataloader = DataLoader(data, shuffle=train, batch_size=8)
    return dataloader


if __name__ == "__main__":
    main()
