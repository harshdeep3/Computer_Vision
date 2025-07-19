import pathlib
import random
from torchvision import transforms
import os

import torch
from torch import nn
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from sklearn.metrics import accuracy_score
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


def change_outlayer(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Change the output layer of a model to match the number of classes.

    Args:
        model (nn.Module): The model to change.
        num_classes (int): The number of classes for the new output layer.

    Returns:
        nn.Module: The modified model with the new output layer.
    """

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Change the output layer
    # output layer is (fc): Linear(in_features=512, out_features=1000, bias=True)
    # Recreate the classifier layer and seed it to the target device
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,  # same number of output units as our number of classes
        bias=True,
    )

    return model


if __name__ == "__main__":
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_path = str(pathlib.Path.cwd() / "data")

    # Create a transforms pipeline manually (required for torchvision < 0.13)
    manual_transforms = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # 1. Reshape all images to 224x224 (though some models may require different sizes)
            transforms.ToTensor(),  # 2. Turn image values to between 0 & 1
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],  # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                std=[0.229, 0.224, 0.225],
            ),  # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
        ]
    )

    # Setup training data
    # 60000 training exmaple
    train_data = datasets.CIFAR100(
        root=data_path,  # where to download data to?
        train=True,  # get training data
        download=True,  # download data if it doesn't exist on disk
        transform=ToTensor(),  # images come as PIL format, we want to turn into Torch tensors
        target_transform=None,  # you can transform labels as well
    )

    # Setup testing data
    # 10000 training exmaple
    test_data = datasets.CIFAR100(
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
    show_gird_of_images(train_data, class_name, False)

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

    # Model resnet18
    # setup model, loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # the reset18 model expects outlayer to 1000
    # so we need to change the output layer to accept 100 classes
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = change_outlayer(model, num_classes=len(class_name)).to(device)

    model_file_path = str(pathlib.Path.cwd()) + "\\saved_models\\resnet18_cifar100.pth"

    # check if model exists
    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model.load_state_dict(torch.load(model_file_path, map_location=device))
    else:
        print(f"Training model model from {model_file_path}")
        model, loss, acc = train_model(
            model=model,
            train_dataloader=train_dataloader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_score,
            epochs=epochs,
        )

        # save the model
        torch.save(model.state_dict(), model_file_path)
        # plot loss and accuracy
        plot_loss_acc(loss, acc, epochs)

    print(
        "Testing -> ",
        test(model, test_dataloader, loss_fn, accuracy_fn=accuracy_score),
    )

    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    # Make predictions on test samples with model 2
    pred_probs = make_predictions(model=model, data=test_samples, device=device)

    # Turn the prediction probabilities into prediction labels by taking the argmax()
    pred_classes = pred_probs.argmax(dim=1)

    show_eval_data_gird_of_images(test_samples, class_name, pred_classes, test_labels)
