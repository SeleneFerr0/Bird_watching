"""Image loading shared by single-image inference workflows."""

import os

from PIL import Image


def load_image(sample: Image.Image | str | os.PathLike[str], channels: int = 3) -> Image.Image:
    """Return an independent image in the model's color mode, closing files we open.

    Coordinates retain the stored pixel orientation, matching directory inference.
    The caller's PIL image is never modified or closed.
    """
    if channels not in (1, 3):
        raise ValueError("Image loading supports one or three input channels")

    mode = "RGB" if channels == 3 else "L"
    if isinstance(sample, (str, os.PathLike)):
        with Image.open(sample) as image:
            return image.convert(mode)
    if isinstance(sample, Image.Image):
        return sample.convert(mode)

    raise TypeError("Expected an image path or a PIL Image")
