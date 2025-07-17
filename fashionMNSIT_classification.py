import pathlib
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models import FashionMNISTModelV0, FashionMNISTModelV1, FashionMNISTModelV2
from sklearn.metrics import accuracy_score

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
        plt.imshow(image.squeeze(), cmap="gray")
    else:
        plt.imshow(
            image.squeeze()
        )  # image shape is [1, 28, 28] (colour channels, height, width)
    plt.title(label)
    plt.show()


def show_gird_of_images(data, class_name, grey_scale=False):
    """
    show a grid of images

    Args:
        data (list): list of data points
    """
    torch.manual_seed(42)
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(data), size=[1]).item()
        img, label = data[random_idx]
        plt.title(class_name[label])
        fig.add_subplot(rows, cols, i)
        plt.title(class_name[label])
        if grey_scale:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img.squeeze())
        plt.axis(False)

    plt.show()


def train_model(
    model, train_dataloader, device, optimizer, loss_fn, accuracy_fn, epochs=5
):
    """
    Train a model on the FashionMNIST dataset

    Args:
        train_dataloader (DataLoader): iterable of training data
        epochs (int): number of epochs to train for
        learning_rate (float): learning rate for optimizer

    Returns:
        model (nn.Module): trained model
    """

    # Training loop
    for _ in tqdm(range(epochs)):

        train_loss = 0
        train_acc = 0
        # Add a loop to loop through training batches
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            model.train()
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y)
            train_loss += loss  # accumulatively add up the loss per epoch

            y_pred = y_pred.cpu()
            y = y.cpu()
            predicted_classes = y_pred.argmax(dim=1)
            train_acc += accuracy_fn(y_true=y, y_pred=predicted_classes)
            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

    print(
        "Training -> ",
        {
            "model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": train_loss.item(),
            "model_acc": train_acc,
        },
    )
    return model


def test(model, test_dataloader, loss_fn, accuracy_fn):
    """
    Test the model on the FashionMNIST dataset
    """
    loss, acc = 0, 0
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode():
        for X, y in test_dataloader:
            X = X.to(device)

            # Make predictions with the model
            y_pred = model(X)
            y_pred = y_pred.cpu()

            predicted_classes = y_pred.argmax(dim=1)
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=predicted_classes)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(test_dataloader)
        acc /= len(test_dataloader)

    return {
        "model_name": model.__class__.__name__,  # only works when model was created with a class
        "model_loss": loss.item(),
        "model_acc": acc,
    }


if __name__ == "__main__":

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    # class names
    class_name = train_data.classes

    # show single image
    # show_image(train_data[0], True)

    # show a grid of images
    # show_gird_of_images(train_data, class_names, False)

    # hyperparameters
    batch_size = 64
    epochs = 5
    lr = 0.001

    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(
        train_data,  # dataset to turn into iterable
        batch_size=batch_size,  # how many samples per batch?
        shuffle=True,  # shuffle data every epoch?
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # don't necessarily have to shuffle the testing data
    )

    # Model v0
    # setup model, loss function and optimizer
    # image shape is [1, 28, 28] (colour channels, height, width)
    model_v0 = FashionMNISTModelV0(
        input_shape=1 * 28 * 28, hidden_units=128, output_shape=len(class_name)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model_v0.parameters(), lr=lr)

    # model_v0 = train_model(
    #     model=model_v0,
    #     train_dataloader=train_dataloader,
    #     device=device,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     accuracy_fn=accuracy_score,
    #     epochs=epochs,
    # )

    # print(
    #     "Testing -> ", test(model_v0, test_dataloader, loss_fn, accuracy_fn=accuracy_score)
    # )

    # model_v1 = FashionMNISTModelV1(
    #     input_shape=1 * 28 * 28, hidden_units=128, output_shape=len(class_name)
    # ).to(device)
    
    # optimizer = torch.optim.Adam(model_v1.parameters(), lr=lr)
    # model_v1 = train_model(
    #     model=model_v1,
    #     train_dataloader=train_dataloader,
    #     device=device,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     accuracy_fn=accuracy_score,
    #     epochs=epochs,
    # )

    # print(
    #     "Testing -> ", test(model_v1, test_dataloader, loss_fn, accuracy_fn=accuracy_score)
    # )
    
    model_v2 = FashionMNISTModelV2(
        input_shape=1, hidden_units=10, output_shape=len(class_name)
    ).to(device)
    
    optimizer = torch.optim.Adam(model_v2.parameters(), lr=lr)

    model_v2 = train_model(
        model=model_v2,
        train_dataloader=train_dataloader,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_score,
        epochs=epochs,
    )

    print(
        "Testing -> ", test(model_v2, test_dataloader, loss_fn, accuracy_fn=accuracy_score)
    )
