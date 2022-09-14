import numpy as np
import torch
from PIL import Image

from ptx_classification.transforms import (
    cast_tensor_to_float32,
    cast_tensor_to_uint8,
    create_augmentation_transform,
    ten_crop,
)


def test_augmentation_transform() -> None:
    augmentation_transform = create_augmentation_transform(p=1.0)

    pil_img = Image.open("/Users/x/Desktop/chestxray_sample_image.png")  # .convert("RGB")
    pil_img = pil_img.resize((256, 256), resample=Image.Resampling.BILINEAR)
    # pil_img.show()

    img_2d = np.expand_dims(np.array(pil_img), axis=0)
    img = torch.from_numpy(np.concatenate([img_2d, img_2d, img_2d]))
    transformed_img = augmentation_transform.forward(img)
    assert (
        transformed_img.size() == img.size() == (3, 256, 256)
    ), f"transformed_img.size() = {transformed_img.size()}, img.size() = {img.size()}"

    np_img_3d = np.array(img.cpu().detach().numpy())
    np_img = np.transpose(np_img_3d, axes=([1, 2, 0])).astype("uint8")
    pil_img = Image.fromarray(np_img)
    # pil_img.show()


def test_ten_crop() -> None:
    img = torch.randint(low=0, high=256, size=(3, 256, 256))
    print(f"\nimg.size() = {img.size()}")
    ten_crop_transform = ten_crop(224)
    ten_cropped_imgs = ten_crop_transform(img)
    print(f"ten_cropped_imgs.size() = {ten_cropped_imgs.size()}")


def test_cast_tensor_to_uint8() -> None:
    t_float32 = torch.tensor([1, 2, 3, 4, 255], dtype=torch.float32)
    t_uint8 = cast_tensor_to_uint8(t_float32)
    assert t_float32.dtype == torch.float32
    assert t_uint8.dtype == torch.uint8


def test_cast_tensor_to_float32() -> None:
    t_uint8 = torch.tensor([1, 2, 3, 4, 255], dtype=torch.uint8)
    t_float32 = cast_tensor_to_float32(t_uint8)
    assert t_uint8.dtype == torch.uint8
    assert t_float32.dtype == torch.float32
