import pathlib
import random
import os

import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from general_functions import (
    make_predictions,
    show_eval_data_gird_of_images,
    show_gird_of_images,
    show_image,
    test,
    train_model,
    plot_loss_acc,
)
from FNIST_models import FashionMNISTModelV0, FashionMNISTModelV1, FashionMNISTModelV2

print(
    f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}"
)


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
    show_image(train_data[0], class_name)

    # show a grid of images
    show_gird_of_images(train_data, class_name)

    # hyperparameters
    batch_size = 128
    epochs = 10
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
    optimizer = torch.optim.Adam(model_v0.parameters(), lr=lr)

    model_file_path = str(pathlib.Path.cwd()) + "\\saved_models\\model_v0.pth"

    # check if model exists
    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model_v0.load_state_dict(torch.load(model_file_path, map_location=device))
    else:
        print(f"Training model model from {model_file_path}")
        model_v0, loss, acc = train_model(
            model=model_v0,
            train_dataloader=train_dataloader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_score,
            epochs=epochs,
        )

        # save the model
        torch.save(model_v0.state_dict(), model_file_path)
        # plot loss and accuracy
        plot_loss_acc(loss, acc, epochs)

    print(
        "Testing -> ",
        test(model_v0, test_dataloader, loss_fn, accuracy_fn=accuracy_score),
    )

    model_v1 = FashionMNISTModelV1(
        input_shape=1 * 28 * 28, hidden_units=128, output_shape=len(class_name)
    ).to(device)

    optimizer = torch.optim.Adam(model_v1.parameters(), lr=lr)

    model_file_path = str(pathlib.Path.cwd()) + "\\saved_models\\model_v1.pth"

    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model_v1.load_state_dict(torch.load(model_file_path, map_location=device))
    else:
        print(f"Training model model from {model_file_path}")
        model_v1, loss, acc = train_model(
            model=model_v1,
            train_dataloader=train_dataloader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_score,
            epochs=epochs,
        )

        # save the model
        torch.save(model_v1.state_dict(), model_file_path)

        # plot loss and accuracy
        plot_loss_acc(loss, acc, epochs)

    print(
        "Testing -> ",
        test(model_v1, test_dataloader, loss_fn, accuracy_fn=accuracy_score),
    )

    model_v2 = FashionMNISTModelV2(
        input_shape=1, hidden_units=10, output_shape=len(class_name)
    ).to(device)

    optimizer = torch.optim.Adam(model_v2.parameters(), lr=lr)

    model_file_path = str(pathlib.Path.cwd()) + "\\saved_models\\model_v2.pth"

    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model_v2.load_state_dict(torch.load(model_file_path, map_location=device))
    else:
        print(f"Training model model from {model_file_path}")
        model_v2, loss, acc = train_model(
            model=model_v2,
            train_dataloader=train_dataloader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_score,
            epochs=epochs,
        )

        # save the model
        torch.save(model_v2.state_dict(), model_file_path)

        # plot loss and accuracy
        plot_loss_acc(loss, acc, epochs)

    print(
        "Testing -> ",
        test(model_v2, test_dataloader, loss_fn, accuracy_fn=accuracy_score),
    )

    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    # Make predictions on test samples with model 2
    pred_probs = make_predictions(model=model_v2, data=test_samples, device=device)

    # Turn the prediction probabilities into prediction labels by taking the argmax()
    pred_classes = pred_probs.argmax(dim=1)

    show_eval_data_gird_of_images(test_samples, class_name, pred_classes, test_labels)
