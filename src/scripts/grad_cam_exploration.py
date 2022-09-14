from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ptx_classification.models import MultiLabelModel, ResNet18Model
from ptx_classification.transforms import get_transforms


def cam(model, class_index, img_tensor):
    target_layers = []
    target_layers.append(model.model.model.layer4[-1])

    # Instantiate GradCAM
    grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # Get index of target
    targets = [ClassifierOutputTarget(class_index)]

    # Generate heatmap
    grayscale_cam = grad_cam(input_tensor=img_tensor, targets=targets)

    # Display heatmap
    visualization = show_cam_on_image(
        img=np.transpose(img_tensor[0].cpu().detach().numpy(), axes=([1, 2, 0])) / 255,
        mask=grayscale_cam[0, :],
        use_rgb=True,
    )
    pil_image = Image.fromarray(visualization)
    return pil_image


def main():
    submodel = ResNet18Model(pretrained=True, num_classes=1000)
    imagenet_transformation = get_transforms("image_net")
    model = MultiLabelModel(
        model=submodel,
        lr=0.1,
        transform=imagenet_transformation,
        labels=[str(i) for i in range(1000)],
    )

    cat_image_dir = Path("/Users/x/Desktop/cat_images")
    cat_images = cat_image_dir.glob("*")
    save_to = cat_image_dir / "grad_cam_3"

    class_names = {
        "Tabby_cat": 281,
        "tiger_cat": 282,
        "persian_cat": 283,
        "siamese_cat": 284,
        "egyptian_cat": 285,
        "mountain_lion": 286,
        "catamount": 287,
    }
    for i, img_path in enumerate(cat_images):
        print(f"i = {i}")
        if (
            ".DS_Store" not in str(img_path)
            and str(img_path) != "/Users/x/Desktop/cat_images/grad_cam_3"
        ):
            for resize in [(224, 224), (256, 256), (512, 512)]:
                print(f"img_path = {img_path}")
                print(f"resize = {resize}")
                img_name = f"{str(img_path).split('/')[-1]}_resize_{resize}"
                img = Image.open(img_path)
                img = img.resize(resize, resample=Image.Resampling.BILINEAR)
                img.save(save_to / f"{img_name}_original.jpg")

                img = np.transpose(np.array(img), axes=[2, 0, 1])
                tensor_img = torch.from_numpy(img)[None, :]
                pred = model(tensor_img)
                for class_name, index in class_names.items():
                    cam_img = cam(model, index, tensor_img)
                    cam_img.save(
                        save_to / f"{img_name}_{class_name}_proba={pred[0, index]:.3f}.jpg"
                    )


main()
