# Computer Vision Projects

This repository documents my learning journey in **Computer Vision** using **Python**, **PyTorch**, and **OpenCV**. It includes experiments with different datasets and techniques such as classification, transfer learning, and object detection.

---

## 🔍 Project Structure

### 1. Fashion MNIST Classification

This section follows:
- [Learn PyTorch YouTube Tutorial](https://www.youtube.com/watch?v=V_xro1bcAuA&ab_channel=freeCodeCamp.org)
- [Learn PyTorch Website](https://www.learnpytorch.io/)
- [Jupyter Notebook by Daniel Bourke](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb)

🧠 **Goal**: Understand the fundamentals of image classification using neural networks and the Fashion MNIST dataset.

---

### 2. CIFAR-100 & Transfer Learning

In the `transfer_learning` notebook, I explore transfer learning using the **CIFAR-100** dataset.

🔄 **What is Transfer Learning?**  
Transfer learning allows us to reuse patterns (weights) learned by a model trained on a large dataset (like ImageNet) and fine-tune it on our own task, which helps when we have less data.

🔗 Based on: [PyTorch Transfer Learning Guide](https://www.learnpytorch.io/06_pytorch_transfer_learning/)

Use cases covered:
- Leveraging pretrained models
- Fine-tuning and freezing layers
- Improving accuracy on small datasets

---

## 🔲 In progress: Object Detection & Bounding Boxes

The next step in this project is to expand into **object detection** using bounding box prediction.

🧱 **Planned Features:**
- Dataset: COCO / Pascal VOC or custom annotated images
- Models: SSD, Faster R-CNN (via `torchvision.models.detection`)
- Visualizations using OpenCV and `matplotlib`
- Metrics: Intersection over Union (IoU), mAP

🎯 **Goal**: Train and evaluate models that not only classify images but also localize objects within them by drawing bounding boxes.

---

## 📦 Tech Stack

- Python
- PyTorch
- OpenCV
- Matplotlib
- TorchVision Datasets & Models
- Jupyter Notebooks

---
