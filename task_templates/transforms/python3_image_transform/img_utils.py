import base64
from io import BytesIO
import numpy as np
from PIL import Image


def img_array_to_b64(image_data: np.array) -> bytes:
    """encode an image still in array format as a b64 string for use in DataRobot"""
    image = Image.fromarray(image_data)
    return img_to_b64(image)


def img_to_b64(image: Image.Image) -> bytes:
    """Encode an image as a b64 string for use in DataRobot."""
    if image.format == "LA":
        # grayscale image
        output_type = "PNG"
    elif image.format != "RBG":
        # color image, make sure it is RGB
        image = image.convert("RGB")
        output_type = "JPEG"
    with BytesIO() as image_ram_buffer:
        image.save(image_ram_buffer, output_type)
        image_ram_buffer.seek(0)
        image_bytes = image_ram_buffer.read()
        b64 = base64.b64encode(image_bytes)
        return b64


def b64_img_to_array(b64: str, dtype=np.uint8) -> np.array:
    """Load a b64 encoded image to be used in the transform as a numpy array"""
    return np.array(Image.open(BytesIO(base64.b64decode(b64))), dtype=dtype)


def b64_to_img(b64: str) -> Image.Image:
    """Load a b64 encoded image as a PIL Image object"""
    return Image.open(BytesIO(base64.b64decode(b64)))


def img_to_grayscale(img: Image.Image) -> Image.Image:
    """convert an image to grayscale"""
    return img.convert("LA")
