import pathlib

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

print(
    f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}"
)


def show_image(data, grey_scale=False):
    """
    show the data as an image

    Args:
        data (tuple): individual data point
    """
    image, label = data
    print(f"Image shape: {image.shape}")
    if grey_scale:
        plt.imshow(
            image.squeeze(),
            cmap="gray"
        )
    else:
        plt.imshow(
            image.squeeze()
        )  # image shape is [1, 28, 28] (colour channels, height, width)
    plt.title(label)
    plt.show()


if __name__ == "__main__":
    data_path = str(pathlib.Path.cwd() / "data")

    # Setup training data
    # 60000 training exmaple
    train_data = datasets.FashionMNIST(
        root=data_path,  # where to download data to?
        train=True,  # get training data
        download=True,  # download data if it doesn't exist on disk
        transform=ToTensor(),  # images come as PIL format, we want to turn into Torch tensors
        target_transform=None,  # you can transform labels as well
    )

    # Setup testing data
    # 10000 training exmaple
    test_data = datasets.FashionMNIST(
        root=data_path,
        train=False,
        download=True,
        transform=ToTensor(),  # get test data
    )

    show_image(train_data[0], True)
