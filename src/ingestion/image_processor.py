"""Local image preprocessing utilities. No external API is used."""

from PIL import Image


class ImageProcessor:
    def prepare(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")
