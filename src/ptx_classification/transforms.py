from typing import Optional

import torch
import torch.nn
from torchvision import transforms as T
from torchvision.transforms import ColorJitter, RandomApply, transforms


def create_augmentation_transform(p: float) -> Optional[torch.nn.Module]:
    if p == 0.0:
        return None
    else:
        return RandomApply(
            p=p,
            transforms=[
                T.RandomHorizontalFlip(p=0.2),
                RandomApply(
                    p=0.6,
                    transforms=[
                        T.RandomAffine(degrees=(-25, 25), translate=(0.2, 0.2), scale=(0.7, 1.0))
                    ],
                ),
                RandomApply(
                    p=0.6, transforms=[ColorJitter(brightness=(0.6, 0.7), contrast=(0.5, 0.7))]
                ),
                RandomApply(p=1.0, transforms=[ColorJitter(brightness=(0.5, 1.1))]),
            ],
        )


def ten_crop(crop_size: int) -> T.Compose:
    assert crop_size >= 0, f"crop_size = {crop_size}"
    return T.Compose(
        [
            # transforms.Resize(256),
            T.TenCrop(crop_size),
            T.Lambda(lambda crops: torch.stack([crop for crop in crops])),
        ]
    )


def cast_tensor_to_uint8(t: torch.Tensor) -> torch.Tensor:
    return t.type(dtype=torch.uint8)


def cast_tensor_to_float32(t: torch.Tensor) -> torch.Tensor:
    return t.type(dtype=torch.float32)


def imagenet_transformations() -> torch.nn.Module:
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def transform_image_from_zero_to_one() -> torch.nn.Module:
    return T.Normalize(mean=[0, 0, 0], std=[255, 255, 255])


def get_transforms(transformation: Optional[str] = None) -> Optional[torch.nn.Module]:
    transform = None
    if transformation == "image_net":
        transform = transforms.Compose(
            [
                cast_tensor_to_float32,
                transform_image_from_zero_to_one(),
                imagenet_transformations(),
            ]
        )
    return transform
