from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ptx_classification.evaluation.grad_cam import cam, plot_grad_cam
from ptx_classification.models import MultiLabelModel, load_multi_label_model_from_checkpoint
from ptx_classification.transforms import get_transforms

model_path = Path(
    "/Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/ray_results/training_function_57e43_00008_8_data_aug_p=0.75_labels_to_use=ChestTube_model=model_class_ResNet18Model_max_epochs_100_lr_5e-05_transform_image_net_batch_size_32_loss_FocalWith/checkpoint_best_model/epoch=21-step=9262.ckpt"
)
model: MultiLabelModel = load_multi_label_model_from_checkpoint(model_path=model_path)
class_labels = model.class_labels
# class_labels = ["Pneumothorax"]
# model = MultiLabelModel(
#     model=DenseNet121Model(pretrained=True, num_classes=len(labels)),
#     lr=1e-04,
#     transform=get_transforms("image_net"),
#     labels=class_labels,
#     loss=BCEWithLogitsLoss(),
# )

pil_img = Image.open(
    "/Users/x/Desktop/bioinformatik/thesis/data_samples/chexpert_sample/ct_positive_large/patient01465_study1_view1_frontal.jpg",
)

np_img = np.array(pil_img)
rgb_img = np.array([np_img, np_img, np_img])
tensor_img = torch.from_numpy((rgb_img))


def test_cam() -> None:
    for class_label in class_labels:
        print(f"class_label = {class_label}")
        img = cam(model=model, img=tensor_img, class_label=class_label)


# def test_plot_grad_cam():
plot_grad_cam(models=[model], model_names=["model_1"], imgs=[tensor_img])
