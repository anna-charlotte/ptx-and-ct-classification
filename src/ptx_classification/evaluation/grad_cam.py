from pathlib import Path
from typing import List, Optional

import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ptx_classification.models import DenseNet121Model, MultiLabelModel, ResNet18Model
from ptx_classification.utils import get_date_and_time


def cam(model: MultiLabelModel, img: torch.Tensor, class_label: str) -> Image:
    target_layers = []
    if model.model.__class__.__name__ == DenseNet121Model.__name__:
        target_layers.append(model.model.model.features.denseblock4.denselayer16.conv2)
    elif model.model.__class__.__name__ == ResNet18Model.__name__:
        target_layers.append(model.model.model.layer4[-1])

    img_tensor = img.unsqueeze(0)

    # Instantiate GradCAM
    grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # Get index of target
    index = model.class_labels.index(class_label)
    targets = [ClassifierOutputTarget(index)]

    # Generate heatmap
    grayscale_cam = grad_cam(input_tensor=img_tensor, targets=targets)

    # Display heatmap
    visualization = show_cam_on_image(
        img=np.transpose(img.cpu().detach().numpy(), axes=([1, 2, 0])) / 255,
        mask=grayscale_cam[0, :],
        use_rgb=True,
    )
    pil_image = PIL.Image.fromarray(visualization)
    return pil_image


def plot_grad_cam(
    models: List[MultiLabelModel],
    model_names: List[str],
    imgs: List[torch.Tensor],
    img_id: str = "",
    save_to: Path = None,
    show: bool = False,
    ground_truth_image: Image = None,
    ground_truth_title: Optional[str] = None,
) -> None:
    assert (
        len(models) == len(model_names) == len(imgs)
    ), f"len(models) = {len(models)}, len(model_names) = {len(model_names)}, len(imgs) = {len(imgs)}"

    images_to_plot = []
    titles = []
    if ground_truth_image is not None and ground_truth_title is not None:
        images_to_plot.append(ground_truth_image)
        titles.append(ground_truth_title)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device)

    class_name_abbreviations = {"Pneumothorax": "PTX", "Chest Tube": "CT"}
    # get cam images for each
    for i, model in enumerate(models):
        for class_label in reversed(model.class_labels):
            grad_cam_img = cam(model=model, img=imgs[i].to(device), class_label=class_label)
            images_to_plot.append(grad_cam_img)
            pred = model.predict(imgs[i][None, :].to(device))
            titles.append(
                f"{class_name_abbreviations[class_label]} prediction ({pred.cpu().detach().numpy():.2f}) by {model_names[i].replace('Pneumothorax', 'PTX').replace('Chest Tube', 'CT')}"
            )

    # create figure
    width = 5 * len(images_to_plot)
    fig = plt.figure(figsize=(width, 5))

    # setting values to rows and column variables
    rows = 1
    columns = len(images_to_plot)

    for i, imgs in enumerate(images_to_plot):
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i + 1)

        # showing image
        plt.imshow(images_to_plot[i])
        plt.axis("off")
        plt.title(titles[i])
    if save_to is not None:
        if img_id == "":
            img_id = get_date_and_time()
        plt.savefig(save_to / f"grad_cam_{img_id}.png")
    if show:
        plt.show()
