import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from typing import Tuple


def show_image(data, class_name, grey_scale=False):
    """
    show the data as an image

    Args:
        data (tuple): individual data point
    """
    image, label = data
    print(f"Image shape: {image.shape}")

    image = image.permute(1, 2, 0)  # remove the colour channel dimension
    if grey_scale:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)  # image shape is [1, 28, 28] (colour channels, height, width)
    plt.title(class_name[label])
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
        
        # Convert from [C, H, W] to [H, W, C]
        img = img.permute(1, 2, 0)
        
        fig.add_subplot(rows, cols, i)
        plt.title(class_name[label])
        if grey_scale:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_eval_data_gird_of_images(test_samples, class_name, pred_classes, test_labels, grey_scale=False):
    """
    show a grid of images

    Args:
        data (list): list of data points
    """
    # Plot predictions
    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        # Create a subplot
        plt.subplot(nrows, ncols, i + 1)
        
        sample = sample.permute(1, 2, 0)  # remove the colour channel dimension
        # Plot the target image
        if grey_scale:
            plt.imshow(sample, cmap="gray")
        else:
            plt.imshow(sample)

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_name[pred_classes[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_name[test_labels[i]]

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")  # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r")  # red text if wrong
        plt.axis(False)

    plt.show()


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    device: str,
    optimizer,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    epochs=5,
) -> Tuple[torch.nn.Module, float, float]:
    """
    Train a model on the FashionMNIST dataset

    Args:
        train_dataloader (DataLoader): iterable of training data
        epochs (int): number of epochs to train for
        learning_rate (float): learning rate for optimizer

    Returns:
        model (nn.Module): trained model
    """
    train_acc = []
    train_loss = []
    # Training loop
    for _ in tqdm(range(epochs)):

        epoch_loss = 0
        epoch_acc = 0
        
        all_preds = []
        all_labels = []
        # Add a loop to loop through training batches
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            model.train()
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss  # accumulatively add up the loss per epoch

            y_pred = y_pred.cpu()
            predicted_classes = torch.argmax(y_pred, dim=1)
            
            all_preds.extend(predicted_classes.numpy())
            all_labels.extend(y.cpu().numpy())
            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        epoch_loss /= len(train_dataloader)
        epoch_acc = accuracy_fn(all_labels, all_preds)
        
        train_loss.append(epoch_loss.item())
        train_acc.append(epoch_acc)

    print(
        "Training -> ",
        {
            "model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": train_loss,
            "model_acc": train_acc,
        },
    )
    return model, train_loss, train_acc


def test(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
):
    """
    Test the model on the FashionMNIST dataset
    """
    loss, acc = 0, 0
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for X, y in test_dataloader:
            X = X.to(device)

            # Make predictions with the model
            y_pred = model(X)
            y_pred = y_pred.cpu()

            predicted_classes = y_pred.argmax(dim=1)
            # Accumulate the loss and accuracy values per batch
            all_preds.extend(predicted_classes.numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Calculate loss
            loss += loss_fn(y_pred, y)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(test_dataloader)
        acc = accuracy_fn(all_labels, all_preds)

    return {
        "model_name": model.__class__.__name__,  # only works when model was created with a class
        "model_loss": loss.item(),
        "model_acc": acc,
    }


def make_predictions(model: torch.nn.Module, data: list, device: torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(
                device
            )  # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(
                pred_logit.squeeze(), dim=0
            )  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


def plot_loss_acc(loss, acc, epochs):
    """
    Plot the loss and accuracy of the model over epochs.

    Args:
        loss (list): List of loss values per epoch.
        acc (list): List of accuracy values per epoch.
        epochs (int): Number of epochs.
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), loss, label="Loss", color="blue")
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), acc, label="Accuracy", color="green")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
