import base64
import io
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.pipeline import Pipeline

from object_detection_model import efficientdet
from object_detection_utils import preprocess_image, postprocess_boxes

IMG_SIZE = 150
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
PHI = 1
WEIGHTED_BIFPN = True
MODEL_PATH = 'efficientdet-d1.h5'
IMAGE_SIZES = (512, 640, 768, 896, 1024, 1280, 1408)
IMAGE_SIZE = IMAGE_SIZES[PHI]
NUM_CLASSES = 90 # from coco
SCORE_THRESHOLD = 0.3


def img_preprocessing(image: Image) -> np.ndarray:
    """ given a PIL.Image object resize, convert to RGB and return as np.array """
    image, scale = preprocess_image(image, image_size=IMAGE_SIZE)
    return image, scale

def get_img_obj_from_base64_str(b64_img_str: str) -> Image:
    """ given a base64 encoded image str get the PIL.Image object """
    b64_img = base64.b64decode(b64_img_str)
    image_bytes = io.BytesIO(b64_img)
    return Image.open(image_bytes)

def get_base64_str_from_PIL_img(pillowed_img: Image) -> str:
    """ given a PIL.Image object return base64 encoded str of the image object """
    buffer = io.BytesIO()
    pillowed_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue())

def load_and_preprocess_image_data(x_data: np.ndarray) -> pd.DataFrame:
    """ Apply the preprocessing methods on the data before prediction for the model to work on """
    try:
        image = get_img_obj_from_base64_str(x_data)
    except:
        image = get_imputation_img()
    return img_preprocessing(image)

def apply_image_data_preprocessing(x_data: np.ndarray) -> np.ndarray:
    """ Image data preprocessing before fit """
    x_data, scale = load_and_preprocess_image_data(x_data)
    return x_data, scale

def get_imputation_img() -> str:
    """ Black image in base64 str for data imputation filling """
    black_PIL_img = Image.fromarray(np.zeros(IMG_SHAPE, dtype="float32"), "RGB")
    return black_PIL_img

def predict_with_preprocessing(model, b64_image_string: str) -> np.ndarray:
    """ Apply necessary preprocessing to conver b64 image string to image values, preprocessing to
    the image values and finally predict bounding boxes and labels with the preprocessed image
    values
    """
    image, scale = apply_image_data_preprocessing(b64_image_string)
    w, h = image.shape[:2]
    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
    # select indices which have a score above the threshold
    indices = np.where(scores[:] > SCORE_THRESHOLD)[0]
    # select those detections
    boxes = boxes[indices]
    labels = labels[indices]
    return np.hstack([boxes, np.expand_dims(labels, 1)])

def load_image_object_detection_inference_pipeline(input_dir: str) -> Pipeline:
    """ Load keras based image object detection model used to predict bounding boxes and labels """
    _, model = efficientdet(phi=PHI,
                            weighted_bifpn=WEIGHTED_BIFPN,
                            num_classes=NUM_CLASSES,
                            score_threshold=SCORE_THRESHOLD)
    model.load_weights(os.path.join(input_dir, MODEL_PATH), by_name=True)
    return model
