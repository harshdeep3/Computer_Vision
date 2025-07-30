import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import SSD300_VGG16_Weights

from getDataFromKaggle import get_data_from_kaggle


def load_data(file_path: str = "data/cat_dog_dataset") -> datasets.ImageFolder:
    """
    Load the dataset and prepare it for training.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    data_dir = file_path + "/PetImages"

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    return dataset


def create_dataloader(
    full_dataset: datasets.ImageFolder,
    training_amount: float = 0.7,
    val_amount: float = 0.1,
    batch_size: int = 32,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    """

    # split the dataset into train, validation and test sets
    train_size = int(training_amount * len(full_dataset))
    val_size = int(val_amount * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def check_one_batch(train_loader: DataLoader):
    """
    Check one batch of data from the DataLoader.
    """
    # Get a batch from the dataloader
    images, labels = next(iter(train_loader))

    print(f"Image shape: {images.size()}")
    print(f"Labels: {labels}")

    # Display the first image in the batch
    image = images[0].numpy().transpose((1, 2, 0))
    plt.imshow(image)

    # 0 is cat, 1 is dog
    plt.title(f"Label: {labels[0].item()}")
    plt.axis("off")
    plt.show()


def detect_objects(model, image, confidence_threshold=0.80):
    """
    Detect objects in an image using the provided model.

    Args:
        model (_type_): _trained object detection model_.
        image (_type_): _image to be processed_.
        confidence_threshold (float, optional): confidence in prediction. Defaults to 0.80.

    Returns:
        _type_: filtered bounding boxes, scores, and labels.
    """
    with torch.no_grad():
        y_pred = model([image])

    bbox, scores, labels = y_pred[0]["boxes"], y_pred[0]["scores"], y_pred[0]["labels"]
    indices = torch.nonzero(scores > confidence_threshold).squeeze(1)

    filtered_bbox = bbox[indices]
    filtered_scores = scores[indices]
    filtered_labels = labels[indices]

    return filtered_bbox, filtered_scores, filtered_labels


def draw_bounding_boxes(image, bbox, labels, class_names):
    """
    Draw bounding boxes on the image.

    Args:
        image (np.array): Image to draw on.
        boxes (torch.Tensor): Bounding boxes.
        scores (torch.Tensor): Scores for each box.
        labels (torch.Tensor): Labels for each box.
        threshold (float): Minimum score to draw the box.
    """
    img_copy = image.copy()

    for i in range(len(bbox)):
        x, y, w, h = bbox[i].cpu().numpy().astype("int")
        cv2.rectangle(img_copy, (x, y), (w, h), (0, 0, 255), 5)

        class_index = labels[i].cpu().numpy().astype("int")
        print(f"Class index: {class_index}")
        class_detected = class_names[class_index - 1]

        cv2.putText(
            img_copy,
            class_detected,
            (x, y + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return img_copy


if __name__ == "__main__":

    # setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = "bhavikjikadara/dog-and-cat-classification-dataset"
    save_path = "data/cat_dog_dataset"
    class_names = SSD300_VGG16_Weights.DEFAULT.meta["categories"]

    if not os.path.exists(save_path):
        get_data_from_kaggle(dataset_name=dataset_name, save_path=save_path)

    full_dataset = load_data(save_path)
    train_loader, val_loader, test_loader = create_dataloader(full_dataset)

    check_one_batch(train_loader)

    # loading the model
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    model.to(device)
    model.eval()
    
    images, labels = next(iter(train_loader))
    
    image = images[0].to(device)
    label = labels[0].to(device)

    bbox, scores, labels = detect_objects(
        model, image, confidence_threshold=0.25
    )
    
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    
    draw_bounding_boxes = draw_bounding_boxes(image_np, bbox, labels, class_names)
    
    cv2.imshow("Bounding Boxes", draw_bounding_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
