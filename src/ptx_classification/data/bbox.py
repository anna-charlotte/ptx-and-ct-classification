from typing import Optional

from PIL import Image, ImageDraw


def draw_bbox(
    image: Image,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    fill: Optional[str] = None,
    outline: Optional[str] = None,
    width: int = 1,
) -> Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle(((x_min, y_min), (x_max, y_max)), fill=fill, outline=outline, width=width)
    return image
